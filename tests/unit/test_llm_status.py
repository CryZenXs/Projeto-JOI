"""Tests for /api/v1/llm/status endpoint."""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient


class TestLLMStatusEndpoint:
    """Tests for GET /api/v1/llm/status."""

    @pytest.mark.asyncio
    async def test_status_returns_200(self, client: AsyncClient) -> None:
        """Status endpoint should return 200 OK."""
        resp = await client.get("/api/v1/llm/status")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_status_response_schema(self, client: AsyncClient) -> None:
        """Response should match the documented schema."""
        resp = await client.get("/api/v1/llm/status")
        data = resp.json()

        assert "providers" in data
        assert isinstance(data["providers"], list)
        assert "stats" in data
        assert "router_configured" in data

    @pytest.mark.asyncio
    async def test_status_includes_stats_fields(self, client: AsyncClient) -> None:
        """Stats should include all required fields."""
        resp = await client.get("/api/v1/llm/status")
        stats = resp.json()["stats"]

        assert "total_requests" in stats
        assert "primary_successes" in stats
        assert "fallback_successes" in stats
        assert "total_failures" in stats
        assert "privacy_mode_requests" in stats
        assert "fallback_rate" in stats
        assert 0.0 <= stats["fallback_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_status_providers_have_required_fields(self, client: AsyncClient) -> None:
        """Each provider should have name, healthy, circuit_state, is_primary, is_fallback."""
        resp = await client.get("/api/v1/llm/status")
        providers = resp.json()["providers"]

        if providers:  # may be empty if router not configured
            for p in providers:
                assert "name" in p
                assert "healthy" in p
                assert "circuit_state" in p
                assert "is_primary" in p
                assert "is_fallback" in p
                assert p["circuit_state"] in ("closed", "open", "half_open", "unknown")

    @pytest.mark.asyncio
    async def test_status_after_request_increments_counter(
        self,
        client: AsyncClient,
    ) -> None:
        """After a chat request, total_requests should increment."""
        # Get initial stats
        resp1 = await client.get("/api/v1/llm/status")
        initial_total = resp1.json()["stats"]["total_requests"]

        # Make a chat request
        await client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        # Get stats again
        resp2 = await client.get("/api/v1/llm/status")
        final_total = resp2.json()["stats"]["total_requests"]

        assert final_total > initial_total
