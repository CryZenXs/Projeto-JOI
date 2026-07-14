"""Pytest configuration and shared fixtures.

This file is loaded by pytest before any test module. It provides:
- Shared fixtures (test client, mock settings, etc.)
- Custom assertion helpers
- Test isolation (each test gets a clean state)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

# Set test environment BEFORE importing app modules
# This ensures settings are loaded with test values, not real .env
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")  # quiet tests
os.environ.setdefault("GROQ_API_KEY", "")  # tests shouldn't need real Groq
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")  # use DB 15 for tests
os.environ.setdefault("OLLAMA_ENABLED", "false")  # don't depend on Ollama in tests


@pytest.fixture(scope="session")
def test_settings() -> Any:
    """Load settings with test overrides."""
    # Clear cached settings to pick up env vars set above
    from app.core.config import get_settings
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def app(test_settings: Any) -> Any:
    """Create a fresh FastAPI app instance per test (isolation)."""
    # Import here so env vars are set first
    from app.main import create_app
    return create_app()


@pytest.fixture
async def client(app: Any) -> AsyncIterator[AsyncClient]:
    """Async HTTP client backed by the test app (no real socket)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def reset_settings_cache() -> Iterator[None]:
    """Clear the settings cache before and after the test.

    Useful when a test modifies environment variables and needs settings
    to be re-loaded.
    """
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ─── Custom markers ─────────────────────────────────────────────────────────
def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers (in addition to those in pyproject.toml)."""
    # Markers are already declared in pyproject.toml [tool.pytest.ini_options].
    # This function exists in case we need to register markers dynamically.
    # We don't need to do anything here - pyproject.toml handles it.
