"""Projeto JOI - Health check endpoint.

The /health endpoint provides a lightweight liveness/readiness probe for
container orchestration (Docker, Kubernetes) and monitoring systems.

Design:
- Liveness (/health): fast, no external dependencies checked. Returns 200
  if the process is alive and the event loop is responsive.
- Readiness (/health/ready): checks dependencies (DB, Redis, LLM providers).
  Returns 200 only if all critical dependencies are reachable.

The distinction matters in production: a dead database should not kill the
pod (liveness), but should stop it from receiving traffic (readiness).
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger

router = APIRouter(prefix="/health", tags=["health"])
logger = get_logger(__name__)

# Startup timestamp for uptime calculation
_started_at = time.monotonic()


class LivenessResponse(BaseModel):
    """Response model for liveness probe."""

    status: Literal["ok"] = "ok"
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Current environment")
    uptime_seconds: float = Field(..., description="Process uptime in seconds")


class DependencyStatus(BaseModel):
    """Status of a single external dependency."""

    name: str
    status: Literal["ok", "degraded", "down"]
    latency_ms: float | None = None
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Response model for readiness probe."""

    status: Literal["ok", "degraded", "down"]
    version: str
    uptime_seconds: float
    dependencies: list[DependencyStatus]


@router.get(
    "",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Lightweight check that the process is alive. No external dependencies checked.",
)
async def liveness() -> LivenessResponse:
    """Liveness probe — always returns 200 if the process can serve requests."""
    return LivenessResponse(
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(time.monotonic() - _started_at, 3),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Checks all critical dependencies. Returns 503 if any critical dep is down.",
)
async def readiness() -> JSONResponse:
    """Readiness probe — checks DB, Redis, and LLM providers."""
    deps: list[DependencyStatus] = []

    # Check Groq (optional in dev, required in prod)
    deps.append(await _check_groq())

    # Check Ollama (optional, used for fallback)
    if settings.feature_fallback_local and settings.ollama_enabled:
        deps.append(await _check_ollama())

    # Check Redis (optional at this stage, will be required later)
    deps.append(await _check_redis())

    # Determine overall status
    any_down = any(d.status == "down" for d in deps)
    any_degraded = any(d.status == "degraded" for d in deps)

    if any_down:
        overall: Literal["ok", "degraded", "down"] = "down"
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE
    elif any_degraded:
        overall = "degraded"
        http_status = status.HTTP_200_OK  # degraded but serviceable
    else:
        overall = "ok"
        http_status = status.HTTP_200_OK

    response = ReadinessResponse(
        status=overall,
        version=settings.app_version,
        uptime_seconds=round(time.monotonic() - _started_at, 3),
        dependencies=deps,
    )

    logger.info(
        "health.readiness.checked",
        status=overall,
        deps_count=len(deps),
        down_count=sum(1 for d in deps if d.status == "down"),
    )

    return JSONResponse(
        status_code=http_status,
        content=response.model_dump(),
    )


async def _check_groq() -> DependencyStatus:
    """Check Groq API reachability (without revealing the API key)."""
    if not settings.has_groq_key:
        return DependencyStatus(
            name="groq",
            status="degraded",
            detail="GROQ_API_KEY not configured (primary LLM unavailable)",
        )

    start = time.monotonic()
    try:
        # Lightweight check: list models endpoint (no tokens consumed)
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {settings.groq_api_key.get_secret_value()}"},
            )
        latency_ms = round((time.monotonic() - start) * 1000, 2)

        if resp.status_code == 200:
            return DependencyStatus(
                name="groq",
                status="ok",
                latency_ms=latency_ms,
                detail="API reachable, key valid",
            )
        return DependencyStatus(
            name="groq",
            status="down",
            latency_ms=latency_ms,
            detail=f"HTTP {resp.status_code}",
        )
    except httpx.TimeoutException:
        return DependencyStatus(name="groq", status="down", detail="timeout (5s)")
    except Exception as exc:
        return DependencyStatus(name="groq", status="down", detail=f"connection error: {type(exc).__name__}")


async def _check_ollama() -> DependencyStatus:
    """Check Ollama local server reachability."""
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.ollama_host}/api/tags")
        latency_ms = round((time.monotonic() - start) * 1000, 2)

        if resp.status_code == 200:
            return DependencyStatus(
                name="ollama",
                status="ok",
                latency_ms=latency_ms,
                detail="local server reachable",
            )
        return DependencyStatus(name="ollama", status="down", detail=f"HTTP {resp.status_code}")
    except httpx.ConnectError:
        return DependencyStatus(
            name="ollama",
            status="degraded",  # not critical, just unavailable
            detail="not running (fallback disabled)",
        )
    except Exception as exc:
        return DependencyStatus(name="ollama", status="down", detail=f"{type(exc).__name__}")


async def _check_redis() -> DependencyStatus:
    """Check Redis reachability. Non-blocking at this stage."""
    # We don't import redis here to avoid requiring it at module load time
    # Once Redis is integrated with the app, this will use the shared pool.
    try:
        import redis.asyncio as aioredis  # type: ignore[import-untyped]

        start = time.monotonic()
        client = aioredis.from_url(settings.redis_url, socket_connect_timeout=2.0)
        try:
            pong = await client.ping()
            latency_ms = round((time.monotonic() - start) * 1000, 2)
            return DependencyStatus(
                name="redis",
                status="ok" if pong else "down",
                latency_ms=latency_ms,
            )
        finally:
            await client.aclose()
    except Exception as exc:
        return DependencyStatus(
            name="redis",
            status="degraded",
            detail=f"not reachable: {type(exc).__name__}",
        )


# Re-export Literal for use in function signatures above
from typing import Literal  # noqa: E402
