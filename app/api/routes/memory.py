"""Projeto JOI - Memory status endpoint.

Provides visibility into the memory subsystem:
- Health of each backend (working, episodic, semantic, emotional)
- Statistics (turn count, episode count, fact count)
- Per-user memory inspection (admin/debug only in production)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.logging import get_logger

router = APIRouter(prefix="/memory", tags=["memory"])
logger = get_logger(__name__)


class MemoryBackendStatus(BaseModel):
    """Health status of a single memory backend."""

    name: str = Field(..., description="Backend name (working/episodic/semantic/emotional)")
    healthy: bool = Field(..., description="Whether the backend is operational")
    backend_type: str = Field(..., description="Implementation class name")


class MemoryStats(BaseModel):
    """Statistics about memory usage."""

    episodes_stored: int = Field(ge=0, description="Total episodes in episodic memory")
    facts_stored: int = Field(ge=0, description="Total facts in semantic memory")


class MemoryStatusResponse(BaseModel):
    """Complete memory subsystem status."""

    backends: list[MemoryBackendStatus]
    stats: MemoryStats
    manager_initialized: bool = Field(
        ..., description="Whether the MemoryManager singleton is initialized"
    )


# ─── Dependency ───────────────────────────────────────────────────────────


def get_memory_manager() -> Any:
    """Dependency: return the MemoryManager instance (or None)."""
    from app.memory.manager import get_memory_manager as _get

    try:
        return _get()
    except Exception as exc:
        logger.warning("memory.manager.init_failed", error=str(exc))
        return None


# ─── Endpoint ─────────────────────────────────────────────────────────────


@router.get(
    "/status",
    response_model=MemoryStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get memory subsystem status",
    description="Returns health of all memory backends and usage statistics.",
)
async def get_memory_status(
    manager: Any = Depends(get_memory_manager),
) -> JSONResponse:
    """Handle GET /api/v1/memory/status."""
    if manager is None:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=MemoryStatusResponse(
                backends=[],
                stats=MemoryStats(episodes_stored=0, facts_stored=0),
                manager_initialized=False,
            ).model_dump(),
        )

    # Check health of all backends
    health = await manager.health_check()

    # Build backend status list
    backends: list[MemoryBackendStatus] = []
    for name, healthy in health.items():
        backend_obj = getattr(manager, f"_{name}", None)
        backend_type = type(backend_obj).__name__ if backend_obj else "unknown"
        backends.append(
            MemoryBackendStatus(
                name=name,
                healthy=healthy,
                backend_type=backend_type,
            )
        )

    # Get stats (use a default user_id for global stats — actual per-user
    # stats would require a user_id parameter)
    stats = MemoryStats(episodes_stored=0, facts_stored=0)
    try:
        # Count episodes and facts for all users (sum across users)
        # For simplicity, we use the SQLite backends' count methods with
        # a special "all" user_id that returns total counts
        # This is a simplification — in production, we'd have proper aggregate queries
        pass  # TODO: implement aggregate stats when we add admin endpoints
    except Exception as exc:
        logger.warning("memory.stats.failed", error=str(exc))

    response = MemoryStatusResponse(
        backends=backends,
        stats=stats,
        manager_initialized=True,
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response.model_dump(),
    )
