"""Projeto JOI - FastAPI application entry point.

Run locally with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Or programmatically:
    python -m app.main

Design principles:
- Fail fast on misconfiguration (validate at startup, not on first request)
- Structured request logging (every request gets a correlation ID)
- Graceful shutdown (drain in-flight requests on SIGTERM)
- CORS properly configured (not wildcard in production)
- Health endpoints exposed without authentication
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.routes import health
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: runs startup and shutdown events.

    Startup:
    - Log configuration (redacting secrets)
    - Validate critical settings
    - Pre-warm any heavy resources

    Shutdown:
    - Close database pools, HTTP clients, etc.
    - Log final metrics
    """
    # ─── Startup ──────────────────────────────────────────────────────────
    logger.info(
        "app.startup.begin",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        groq_configured=settings.has_groq_key,
        ollama_enabled=settings.ollama_enabled,
    )

    # Validate production-readiness
    if settings.is_production:
        if not settings.has_groq_key:
            logger.critical("app.startup.failed", reason="GROQ_API_KEY missing in production")
            raise RuntimeError("GROQ_API_KEY is required in production")
        logger.info("app.startup.production_validated")

    # Warn about missing optional components in dev
    if settings.is_development and not settings.has_groq_key:
        logger.warning(
            "app.startup.warning",
            reason="GROQ_API_KEY not set — primary LLM unavailable. "
                   "Set it in .env to enable Groq integration. "
                   "Get a key at https://console.groq.com",
        )

    logger.info("app.startup.complete", host=settings.host, port=settings.port)
    yield
    # ─── Shutdown ─────────────────────────────────────────────────────────
    logger.info("app.shutdown.begin")
    # Future: close DB pools, Redis clients, Ollama sessions, etc.
    logger.info("app.shutdown.complete")


def create_app() -> FastAPI:
    """Application factory.

    Using a factory function (instead of module-level app) enables:
    - Multiple instances in tests with different configurations
    - Cleaner separation between app creation and app running
    - Easier mocking of settings during tests
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Entidade digital companheira inspirada na JOI de Blade Runner 2049. "
            "Núcleo cognitivo conectado à API Groq, com fallback local via Ollama, "
            "memória persistente multi-camada e persona evolutiva."
        ),
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ─── Middleware (order matters: outer to inner) ──────────────────────

    # Gzip: compress responses > 500 bytes
    app.add_middleware(GZipMiddleware, minimum_size=500)

    # CORS: explicit origins, never wildcard in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-User-ID"],
    )

    # ─── Routes ──────────────────────────────────────────────────────────
    app.include_router(health.router, prefix="/api/v1")

    # Root endpoint: redirects to docs (dev only)
    if not settings.is_production:
        from fastapi.responses import RedirectResponse

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse(url="/docs")

    # ─── Request logging middleware ──────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log every request with method, path, status, duration."""
        import time
        import uuid

        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        start = time.monotonic()

        # Attach request_id to structlog context (auto-included in logs)
        from structlog.contextvars import bind_contextvars, clear_contextvars
        bind_contextvars(request_id=request_id)

        try:
            response = await call_next(request)
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            logger.info(
                "request.completed",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=duration_ms,
            )
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception:
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            logger.error(
                "request.failed",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
                exc_info=True,
            )
            raise
        finally:
            clear_contextvars()

    return app


# Module-level app instance (used by uvicorn: app.main:app)
app = create_app()


if __name__ == "__main__":
    # Run programmatically: python -m app.main
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
        access_log=False,  # we handle our own request logging
    )
