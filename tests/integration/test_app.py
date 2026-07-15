"""Integration tests for the FastAPI app.

These tests verify the app as a whole (not isolated units).
They use the ASGI transport (no real socket) so they're fast.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.integration
class TestAppStartup:
    """Tests that the app starts correctly and serves requests."""

    @pytest.mark.asyncio
    async def test_app_responds(self, client: AsyncClient) -> None:
        """The app should respond to at least one endpoint."""
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_openapi_available_in_dev(self, client: AsyncClient) -> None:
        """OpenAPI schema should be available in development."""
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"]["title"] == "Projeto JOI"
        assert "version" in data["info"]

    @pytest.mark.asyncio
    async def test_docs_available_in_dev(self, client: AsyncClient) -> None:
        """Swagger UI should be available in development."""
        resp = await client.get("/docs")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_cors_headers_set(self, client: AsyncClient) -> None:
        """CORS headers should be present on responses."""
        # OPTIONS preflight
        resp = await client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Should not be 400 (CORS rejection)
        assert resp.status_code in (200, 204)


@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling and 404s."""

    @pytest.mark.asyncio
    async def test_unknown_route_returns_404(self, client: AsyncClient) -> None:
        """Unknown routes should return 404 with structured error."""
        resp = await client.get("/api/v1/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_method_not_allowed_returns_405(self, client: AsyncClient) -> None:
        """Wrong HTTP method should return 405."""
        resp = await client.delete("/api/v1/health")
        assert resp.status_code == 405
