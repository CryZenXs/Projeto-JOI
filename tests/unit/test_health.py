"""Unit tests for /api/v1/health endpoints.

Tests verify:
- Liveness probe returns 200 and basic info
- Readiness probe correctly reports dependency status
- Readiness returns 503 when critical deps are down
- Response schema matches OpenAPI contract
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient


class TestLiveness:
    """Tests for GET /api/v1/health (liveness probe)."""

    @pytest.mark.asyncio
    async def test_liveness_returns_200(self, client: AsyncClient) -> None:
        """Liveness probe should always return 200 OK."""
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_liveness_response_schema(self, client: AsyncClient) -> None:
        """Liveness response should match the documented schema."""
        resp = await client.get("/api/v1/health")
        data = resp.json()

        assert data["status"] == "ok"
        assert "version" in data
        assert isinstance(data["version"], str)
        assert "environment" in data
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_liveness_uptime_increases(self, client: AsyncClient) -> None:
        """Uptime should increase over time (proves process is alive)."""
        import asyncio

        r1 = await client.get("/api/v1/health")
        await asyncio.sleep(0.05)  # 50ms
        r2 = await client.get("/api/v1/health")

        assert r2.json()["uptime_seconds"] >= r1.json()["uptime_seconds"]


class TestReadiness:
    """Tests for GET /api/v1/health/ready (readiness probe)."""

    @pytest.mark.asyncio
    async def test_readiness_returns_200_or_503(self, client: AsyncClient) -> None:
        """Readiness should return 200 (ok/degraded) or 503 (down)."""
        resp = await client.get("/api/v1/health/ready")
        assert resp.status_code in (200, 503)

    @pytest.mark.asyncio
    async def test_readiness_response_schema(self, client: AsyncClient) -> None:
        """Readiness response should match the documented schema."""
        resp = await client.get("/api/v1/health/ready")
        data = resp.json()

        assert "status" in data
        assert data["status"] in ("ok", "degraded", "down")
        assert "version" in data
        assert "uptime_seconds" in data
        assert "dependencies" in data
        assert isinstance(data["dependencies"], list)

        # Each dependency should have name, status, and optional detail
        for dep in data["dependencies"]:
            assert "name" in dep
            assert "status" in dep
            assert dep["status"] in ("ok", "degraded", "down")

    @pytest.mark.asyncio
    async def test_readiness_includes_groq_dep(self, client: AsyncClient) -> None:
        """Readiness should check Groq as a dependency."""
        resp = await client.get("/api/v1/health/ready")
        deps = resp.json()["dependencies"]
        dep_names = [d["name"] for d in deps]
        assert "groq" in dep_names

    @pytest.mark.asyncio
    async def test_readiness_groq_degraded_without_key(self, client: AsyncClient) -> None:
        """Without GROQ_API_KEY, Groq dep should be 'degraded' (not 'down')."""
        # The test client uses test_settings which has no GROQ_API_KEY
        resp = await client.get("/api/v1/health/ready")
        deps = resp.json()["dependencies"]
        groq_dep = next(d for d in deps if d["name"] == "groq")
        assert groq_dep["status"] == "degraded"
        assert "not configured" in (groq_dep.get("detail") or "").lower()


class TestRootRedirect:
    """Tests for the root endpoint (dev-only redirect to docs)."""

    @pytest.mark.asyncio
    async def test_root_redirects_to_docs_in_dev(self, client: AsyncClient) -> None:
        """In development, / should redirect to /docs."""
        resp = await client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307)
        assert "/docs" in resp.headers.get("location", "")


class TestRequestTracing:
    """Tests for the request logging middleware (X-Request-ID)."""

    @pytest.mark.asyncio
    async def test_request_id_returned_in_header(self, client: AsyncClient) -> None:
        """Every response should include an X-Request-ID header."""
        resp = await client.get("/api/v1/health")
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) > 0

    @pytest.mark.asyncio
    async def test_request_id_echoed_when_provided(self, client: AsyncClient) -> None:
        """If client provides X-Request-ID, server should echo it back."""
        custom_id = "test-request-id-12345"
        resp = await client.get(
            "/api/v1/health",
            headers={"X-Request-ID": custom_id},
        )
        assert resp.headers["X-Request-ID"] == custom_id
