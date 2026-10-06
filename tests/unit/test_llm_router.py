"""Tests for app.llm.router.LLMRouter.

Tests verify:
- Primary provider success (no fallback needed)
- Fallback triggered when primary fails with retryable error
- Fallback NOT triggered on non-retryable errors (when no fallback configured)
- Privacy mode forces fallback provider only
- PRIMARY_ONLY mode skips fallback
- Circuit breaker integration (OPEN state triggers fallback)
- Streaming with fallback
- Health check endpoint
- Stats tracking
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.circuit_breaker import CircuitBreaker, CircuitState
from app.llm.base import LLMMessage, LLMRequest, LLMRole, TokenChunk, TokenUsage
from app.llm.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.llm.mock_client import MockLLMClient
from app.llm.router import LLMRouter, RoutingMode


@pytest.fixture
def simple_request() -> LLMRequest:
    """Minimal valid LLMRequest."""
    return LLMRequest(
        messages=[LLMMessage(role=LLMRole.USER, content="Hello")],
        request_id="test-req-001",
    )


@pytest.fixture
def primary_mock() -> MockLLMClient:
    """Mock primary client (succeeds)."""
    return MockLLMClient(
        default_model="primary-model",
        responses=["Primary response"],
    )


@pytest.fixture
def fallback_mock() -> MockLLMClient:
    """Mock fallback client (succeeds)."""
    return MockLLMClient(
        default_model="fallback-model",
        responses=["Fallback response"],
    )


@pytest.fixture
def router(primary_mock: MockLLMClient, fallback_mock: MockLLMClient) -> LLMRouter:
    """Router with primary and fallback mocks."""
    return LLMRouter(
        primary=primary_mock,
        fallback=fallback_mock,
        max_retries=2,
        retry_base_delay_seconds=0.01,  # fast for tests
    )


# ─── Complete (non-streaming) tests ──────────────────────────────────────


class TestRouterComplete:
    """Test the complete() method (non-streaming)."""

    @pytest.mark.asyncio
    async def test_primary_success_no_fallback(
        self,
        router: LLMRouter,
        simple_request: LLMRequest,
    ) -> None:
        """When primary succeeds, fallback should NOT be called."""
        response = await router.complete(simple_request)

        assert response.content == "Primary response"
        assert response.provider == "mock"
        # Fallback mock should not have been called
        assert fallback_mock_unused(router)

    @pytest.mark.asyncio
    async def test_fallback_on_rate_limit(
        self,
        fallback_mock: MockLLMClient,
        simple_request: LLMRequest,
    ) -> None:
        """When primary hits rate limit, fallback should be used."""
        # Use fresh primary mock (no pre-loaded responses)
        primary_mock = MockLLMClient(default_model="primary-model")
        primary_mock.queue_error(
            LLMRateLimitError(provider="mock", retry_after_seconds=1.0)
        )

        router = LLMRouter(
            primary=primary_mock,
            fallback=fallback_mock,
            max_retries=1,
            retry_base_delay_seconds=0.01,
        )

        response = await router.complete(simple_request)

        assert response.content == "Fallback response"
        assert response.provider == "mock"

    @pytest.mark.asyncio
    async def test_fallback_on_timeout(
        self,
        fallback_mock: MockLLMClient,
        simple_request: LLMRequest,
    ) -> None:
        """When primary times out, fallback should be used."""
        primary_mock = MockLLMClient(default_model="primary-model")
        primary_mock.queue_error(
            LLMTimeoutError(provider="mock", timeout_seconds=30.0)
        )

        router = LLMRouter(
            primary=primary_mock,
            fallback=fallback_mock,
            max_retries=1,
            retry_base_delay_seconds=0.01,
        )

        response = await router.complete(simple_request)

        assert response.content == "Fallback response"

    @pytest.mark.asyncio
    async def test_fallback_on_connection_error(
        self,
        fallback_mock: MockLLMClient,
        simple_request: LLMRequest,
    ) -> None:
        """When primary has connection error, fallback should be used."""
        primary_mock = MockLLMClient(default_model="primary-model")
        primary_mock.queue_error(
            LLMConnectionError(provider="mock", cause=ConnectionError("network down"))
        )

        router = LLMRouter(
            primary=primary_mock,
            fallback=fallback_mock,
            max_retries=1,
            retry_base_delay_seconds=0.01,
        )

        response = await router.complete(simple_request)

        assert response.content == "Fallback response"

    @pytest.mark.asyncio
    async def test_no_fallback_on_auth_error_without_fallback(
        self,
        simple_request: LLMRequest,
    ) -> None:
        """Auth error should propagate when no fallback is configured."""
        primary_mock = MockLLMClient(default_model="primary-model")
        primary_mock.queue_error(
            LLMAuthenticationError(provider="mock")
        )

        router = LLMRouter(
            primary=primary_mock,
            fallback=None,
            max_retries=1,
            retry_base_delay_seconds=0.01,
        )

        with pytest.raises(LLMAuthenticationError):
            await router.complete(simple_request)

    @pytest.mark.asyncio
    async def test_privacy_mode_forces_fallback(
        self,
        router: LLMRouter,
        fallback_mock: MockLLMClient,
        simple_request: LLMRequest,
    ) -> None:
        """In privacy mode, primary should be skipped, fallback used."""
        # Clear primary's responses so it would echo if called
        # (we want to verify it's NOT called)
        response = await router.complete(simple_request, mode=RoutingMode.PRIVACY)

        assert response.content == "Fallback response"
        # Primary should not have been called
        assert router._primary.call_count == 0  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_primary_only_mode_no_fallback(
        self,
        fallback_mock: MockLLMClient,
        simple_request: LLMRequest,
    ) -> None:
        """In PRIMARY_ONLY mode, fallback should NOT be used even on failure."""
        primary_mock = MockLLMClient(default_model="primary-model")
        primary_mock.queue_error(
            LLMRateLimitError(provider="mock", retry_after_seconds=1.0)
        )

        router = LLMRouter(
            primary=primary_mock,
            fallback=fallback_mock,
            max_retries=1,
            retry_base_delay_seconds=0.01,
        )

        # In PRIMARY_ONLY mode, when primary fails, should raise (not fall back)
        # MaxRetriesExceeded wraps the original LLMRateLimitError
        from app.core.retry import MaxRetriesExceeded
        with pytest.raises((LLMError, MaxRetriesExceeded)):
            await router.complete(simple_request, mode=RoutingMode.PRIMARY_ONLY)

        # Verify fallback was NOT called
        assert fallback_mock.call_count == 0

    @pytest.mark.asyncio
    async def test_retry_then_success(
        self,
        simple_request: LLMRequest,
    ) -> None:
        """When primary fails once then succeeds, no fallback needed."""
        # Use a fresh mock (not the fixture which pre-loads responses)
        primary_mock = MockLLMClient(default_model="primary-model")
        # First call: rate limit error
        primary_mock.queue_error(
            LLMRateLimitError(provider="mock", retry_after_seconds=0.01)
        )
        # Second call (retry): success
        primary_mock.queue_response("Success after retry")

        router = LLMRouter(
            primary=primary_mock,
            fallback=None,
            max_retries=3,
            retry_base_delay_seconds=0.01,
        )

        response = await router.complete(simple_request)

        assert response.content == "Success after retry"

    @pytest.mark.asyncio
    async def test_both_providers_fail_raises(
        self,
        fallback_mock: MockLLMClient,
        simple_request: LLMRequest,
    ) -> None:
        """When both primary and fallback fail, should raise the error."""
        primary_mock = MockLLMClient(default_model="primary-model")
        # Primary always fails
        for _ in range(5):
            primary_mock.queue_error(
                LLMRateLimitError(provider="mock", retry_after_seconds=0.01)
            )
        # Fallback also fails
        fallback_mock.queue_error(
            LLMProviderError("Ollama unavailable", provider="mock")
        )

        router = LLMRouter(
            primary=primary_mock,
            fallback=fallback_mock,
            max_retries=1,
            retry_base_delay_seconds=0.01,
        )

        with pytest.raises(LLMError):
            await router.complete(simple_request)


# ─── Streaming tests ─────────────────────────────────────────────────────


class TestRouterStream:
    """Test the stream() method."""

    @pytest.mark.asyncio
    async def test_stream_primary_success(
        self,
        router: LLMRouter,
        simple_request: LLMRequest,
    ) -> None:
        """Streaming should work with primary provider."""
        simple_request.stream = True
        # Override primary mock with stream chunks
        router._primary.queue_response("Hello world from primary")  # type: ignore[attr-defined]

        chunks: list[TokenChunk] = []
        async for chunk in router.stream(simple_request):
            chunks.append(chunk)

        # Should have content chunks + final done chunk
        assert len(chunks) > 0
        # Last chunk should have finish_reason
        assert chunks[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_fallback_on_connection_error(
        self,
        primary_mock: MockLLMClient,
        fallback_mock: MockLLMClient,
        simple_request: LLMRequest,
    ) -> None:
        """If primary fails before streaming starts, fallback should be used."""
        # Primary fails with connection error (pre-stream)
        primary_mock.queue_error(
            LLMConnectionError(provider="mock", cause=ConnectionError("no server"))
        )
        # Fallback has chunks to stream
        fallback_mock.queue_response("Fallback stream content")

        router = LLMRouter(
            primary=primary_mock,
            fallback=fallback_mock,
            max_retries=1,
            retry_base_delay_seconds=0.01,
        )

        simple_request.stream = True
        chunks: list[TokenChunk] = []
        async for chunk in router.stream(simple_request):
            chunks.append(chunk)

        # Should have received chunks from fallback
        assert len(chunks) > 0
        content = "".join(c.content for c in chunks if c.content)
        assert "Fallback" in content


# ─── Health check and stats ──────────────────────────────────────────────


class TestRouterHealthCheck:
    """Test the health_check() method."""

    @pytest.mark.asyncio
    async def test_health_check_returns_both_providers(
        self,
        router: LLMRouter,
    ) -> None:
        """Health check should return status for both providers."""
        health = await router.health_check()

        assert "primary" in health
        assert "fallback" in health
        assert health["primary"]["provider"] == "mock"
        assert health["fallback"]["provider"] == "mock"
        # Both mocks are healthy by default
        assert health["primary"]["healthy"] is True
        assert health["fallback"]["healthy"] is True

    @pytest.mark.asyncio
    async def test_health_check_includes_circuit_state(
        self,
        router: LLMRouter,
    ) -> None:
        """Health check should include circuit breaker states."""
        health = await router.health_check()

        assert "circuit_state" in health["primary"]
        assert health["primary"]["circuit_state"] == "closed"

    @pytest.mark.asyncio
    async def test_health_check_includes_stats(
        self,
        router: LLMRouter,
        simple_request: LLMRequest,
    ) -> None:
        """Health check should include request stats."""
        # Make a successful request first
        await router.complete(simple_request)

        health = await router.health_check()

        assert "stats" in health
        assert health["stats"]["total_requests"] == 1
        assert health["stats"]["primary_successes"] == 1


class TestRouterStats:
    """Test the stats tracking."""

    @pytest.mark.asyncio
    async def test_stats_track_primary_success(
        self,
        router: LLMRouter,
        simple_request: LLMRequest,
    ) -> None:
        """Primary success should increment primary_successes counter."""
        await router.complete(simple_request)

        health = await router.health_check()
        stats = health["stats"]

        assert stats["total_requests"] == 1
        assert stats["primary_successes"] == 1
        assert stats["fallback_successes"] == 0
        assert stats["total_failures"] == 0

    @pytest.mark.asyncio
    async def test_stats_track_fallback_success(
        self,
        fallback_mock: MockLLMClient,
        simple_request: LLMRequest,
    ) -> None:
        """Fallback success should increment fallback_successes counter."""
        primary_mock = MockLLMClient(default_model="primary-model")
        primary_mock.queue_error(
            LLMRateLimitError(provider="mock", retry_after_seconds=0.01)
        )

        router = LLMRouter(
            primary=primary_mock,
            fallback=fallback_mock,
            max_retries=1,
            retry_base_delay_seconds=0.01,
        )

        await router.complete(simple_request)

        health = await router.health_check()
        stats = health["stats"]

        assert stats["total_requests"] == 1
        assert stats["fallback_successes"] == 1
        assert stats["primary_successes"] == 0

    @pytest.mark.asyncio
    async def test_stats_track_privacy_mode(
        self,
        router: LLMRouter,
        simple_request: LLMRequest,
    ) -> None:
        """Privacy mode requests should be tracked separately."""
        await router.complete(simple_request, mode=RoutingMode.PRIVACY)

        health = await router.health_check()
        stats = health["stats"]

        assert stats["privacy_mode_requests"] == 1
        assert stats["fallback_successes"] == 1  # privacy uses fallback


# ─── Circuit breaker integration ─────────────────────────────────────────


class TestRouterCircuitBreaker:
    """Test circuit breaker integration in the router."""

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(
        self,
        fallback_mock: MockLLMClient,
        simple_request: LLMRequest,
    ) -> None:
        """Circuit should open after enough failures."""
        primary_mock = MockLLMClient(default_model="primary-model")

        # Use a fast circuit breaker
        primary_breaker = CircuitBreaker(
            name="test-primary",
            failure_threshold=3,
            recovery_timeout=60.0,
            expected_exceptions=(LLMError,),
        )

        router = LLMRouter(
            primary=primary_mock,
            fallback=fallback_mock,
            primary_breaker=primary_breaker,
            max_retries=1,
            retry_base_delay_seconds=0.01,
        )

        # Make several requests to trip the circuit
        # queue_error is one-shot, so we need to queue before each call
        for _ in range(3):
            primary_mock.queue_error(
                LLMConnectionError(provider="mock", cause=ConnectionError("down"))
            )
            try:
                await router.complete(simple_request)
            except Exception:
                pass

        # Circuit should now be open (or half-open if recovery passed)
        assert primary_breaker.state in (CircuitState.OPEN, CircuitState.HALF_OPEN)


# ─── Helpers ─────────────────────────────────────────────────────────────


def fallback_mock_unused(router: LLMRouter) -> bool:
    """Check that the fallback mock was not called."""
    fallback = router._fallback
    if fallback is None:
        return True
    # MockLLMClient tracks call_count
    return getattr(fallback, "call_count", 0) == 0
