"""Projeto JOI - LLM status and observability endpoint.

Provides visibility into the LLM router's state:
- Which providers are configured
- Circuit breaker states (CLOSED/OPEN/HALF_OPEN)
- Provider health (reachable or not)
- Request statistics (successes, failures, fallbacks)

This endpoint is critical for:
- Debugging ("why is my chat slow today?")
- Monitoring (alert when circuit opens)
- Capacity planning (track fallback usage over time)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.logging import get_logger

router = APIRouter(prefix="/llm", tags=["llm"])
logger = get_logger(__name__)


# ─── Response schemas ─────────────────────────────────────────────────────


class ProviderStatus(BaseModel):
    """Status of a single LLM provider."""

    name: str = Field(..., description="Provider name (groq/ollama/mock)")
    healthy: bool = Field(..., description="Whether the provider is reachable")
    circuit_state: str = Field(
        ...,
        description="Circuit breaker state: closed/open/half_open",
    )
    is_primary: bool = Field(..., description="Whether this is the primary provider")
    is_fallback: bool = Field(..., description="Whether this is the fallback provider")


class RouterStats(BaseModel):
    """Aggregated statistics for the LLM router."""

    total_requests: int = Field(ge=0)
    primary_successes: int = Field(ge=0)
    fallback_successes: int = Field(ge=0)
    total_failures: int = Field(ge=0)
    privacy_mode_requests: int = Field(ge=0)
    fallback_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of requests that used the fallback provider",
    )


class LLMStatusResponse(BaseModel):
    """Complete LLM subsystem status."""

    providers: list[ProviderStatus]
    stats: RouterStats
    router_configured: bool = Field(
        ..., description="Whether the LLMRouter is properly configured"
    )


# ─── Dependency ───────────────────────────────────────────────────────────


def get_router() -> Any:
    """Dependency: return the LLMRouter instance.

    This function is overridden by the chat router's dependency injection.
    In production, it returns the singleton LLMRouter. In tests, it can
    be overridden with a mock router.
    """
    # Lazy import to avoid circular dependency
    from app.api.routes.chat import get_llm_router

    return get_llm_router()


# ─── Endpoint ─────────────────────────────────────────────────────────────


@router.get(
    "/status",
    response_model=LLMStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get LLM router status and statistics",
    description=(
        "Returns the current state of all LLM providers, their circuit breakers, "
        "and aggregated request statistics. Use this for monitoring and debugging."
    ),
)
async def get_llm_status(
    router_instance: Any = Depends(get_router),
) -> JSONResponse:
    """Handle GET /api/v1/llm/status."""
    # If no router is configured (e.g., early in startup), return a minimal response
    if router_instance is None:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=LLMStatusResponse(
                providers=[],
                stats=RouterStats(
                    total_requests=0,
                    primary_successes=0,
                    fallback_successes=0,
                    total_failures=0,
                    privacy_mode_requests=0,
                    fallback_rate=0.0,
                ),
                router_configured=False,
            ).model_dump(),
        )

    # Get health from the router
    try:
        health = await router_instance.health_check()
    except Exception as exc:
        logger.error("llm.status.check_failed", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": f"Failed to check LLM status: {exc}"},
        )

    # Build provider status list
    providers: list[ProviderStatus] = []

    if "primary" in health:
        p = health["primary"]
        providers.append(
            ProviderStatus(
                name=p.get("provider", "unknown"),
                healthy=p.get("healthy", False),
                circuit_state=p.get("circuit_state", "unknown"),
                is_primary=True,
                is_fallback=False,
            )
        )

    if health.get("fallback"):
        f = health["fallback"]
        providers.append(
            ProviderStatus(
                name=f.get("provider", "unknown"),
                healthy=f.get("healthy", False),
                circuit_state=f.get("circuit_state", "unknown"),
                is_primary=False,
                is_fallback=True,
            )
        )

    # Build stats
    stats_data = health.get("stats", {})
    total_requests = stats_data.get("total_requests", 0)
    primary_successes = stats_data.get("primary_successes", 0)
    fallback_successes = stats_data.get("fallback_successes", 0)

    # Calculate fallback rate (avoid division by zero)
    successful = primary_successes + fallback_successes
    fallback_rate = (fallback_successes / successful) if successful > 0 else 0.0

    response = LLMStatusResponse(
        providers=providers,
        stats=RouterStats(
            total_requests=total_requests,
            primary_successes=primary_successes,
            fallback_successes=fallback_successes,
            total_failures=stats_data.get("total_failures", 0),
            privacy_mode_requests=stats_data.get("privacy_mode_requests", 0),
            fallback_rate=round(fallback_rate, 4),
        ),
        router_configured=True,
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response.model_dump(),
    )
