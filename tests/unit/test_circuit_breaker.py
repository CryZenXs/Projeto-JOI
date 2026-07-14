"""Tests for app.core.circuit_breaker.CircuitBreaker.

Verifies the state machine transitions:
- CLOSED → OPEN after `failure_threshold` consecutive failures
- OPEN → HALF_OPEN after `recovery_timeout`
- HALF_OPEN → CLOSED on success
- HALF_OPEN → OPEN on failure (with reset timeout)
- CircuitOpenError raised when calling while OPEN
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


@pytest.fixture
def fast_breaker() -> CircuitBreaker:
    """Circuit breaker with fast recovery for testing."""
    return CircuitBreaker(
        name="test",
        failure_threshold=3,
        recovery_timeout=0.2,  # 200ms
        expected_exceptions=(ValueError, ConnectionError),
    )


async def _success() -> str:
    """Async function that succeeds."""
    await asyncio.sleep(0)
    return "ok"


async def _fail_with(exc: Exception) -> None:
    """Async function that raises a specific exception."""
    await asyncio.sleep(0)
    raise exc


class TestCircuitBreakerStates:
    """Test the state machine transitions."""

    @pytest.mark.asyncio
    async def test_starts_closed(self, fast_breaker: CircuitBreaker) -> None:
        assert fast_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_stays_closed_on_success(self, fast_breaker: CircuitBreaker) -> None:
        result = await fast_breaker.call(_success)
        assert result == "ok"
        assert fast_breaker.state == CircuitState.CLOSED
        assert fast_breaker.stats["total_successes"] == 1

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(
        self, fast_breaker: CircuitBreaker
    ) -> None:
        """Circuit should OPEN after `failure_threshold` consecutive failures."""
        for i in range(fast_breaker.failure_threshold):
            with pytest.raises(ValueError):
                await fast_breaker.call(lambda: _fail_with(ValueError("fail")))
            # Circuit should still be closed until threshold reached
            if i < fast_breaker.failure_threshold - 1:
                assert fast_breaker.state == CircuitState.CLOSED

        # After threshold reached, circuit should be OPEN
        assert fast_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(
        self, fast_breaker: CircuitBreaker
    ) -> None:
        """A success after some failures should reset the failure counter."""
        # Two failures (below threshold of 3)
        with pytest.raises(ValueError):
            await fast_breaker.call(lambda: _fail_with(ValueError("fail")))
        with pytest.raises(ValueError):
            await fast_breaker.call(lambda: _fail_with(ValueError("fail")))
        assert fast_breaker.stats["failure_count"] == 2

        # Success resets
        await fast_breaker.call(_success)
        assert fast_breaker.stats["failure_count"] == 0
        assert fast_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_unexpected_exception_doesnt_trip_circuit(
        self, fast_breaker: CircuitBreaker
    ) -> None:
        """Exceptions NOT in expected_exceptions should NOT count as failures."""
        # RuntimeError is NOT in expected_exceptions (ValueError, ConnectionError)
        with pytest.raises(RuntimeError):
            await fast_breaker.call(lambda: _fail_with(RuntimeError("unexpected")))

        assert fast_breaker.state == CircuitState.CLOSED
        assert fast_breaker.stats["failure_count"] == 0
        # But it should still be counted as a total call
        assert fast_breaker.stats["total_calls"] == 1


class TestCircuitOpenBehavior:
    """Test what happens when the circuit is OPEN."""

    @pytest.mark.asyncio
    async def test_raises_circuit_open_error(self, fast_breaker: CircuitBreaker) -> None:
        """Calls while OPEN should raise CircuitOpenError immediately."""
        # Trip the circuit
        for _ in range(fast_breaker.failure_threshold):
            with pytest.raises(ValueError):
                await fast_breaker.call(lambda: _fail_with(ValueError("fail")))
        assert fast_breaker.state == CircuitState.OPEN

        # Subsequent call should fail fast with CircuitOpenError
        with pytest.raises(CircuitOpenError) as exc_info:
            await fast_breaker.call(_success)

        assert exc_info.value.provider == "test"
        assert exc_info.value.retry_after_seconds > 0
        assert fast_breaker.stats["total_rejected"] >= 1

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(
        self, fast_breaker: CircuitBreaker
    ) -> None:
        """After recovery_timeout, circuit should allow one trial call."""
        # Trip the circuit
        for _ in range(fast_breaker.failure_threshold):
            with pytest.raises(ValueError):
                await fast_breaker.call(lambda: _fail_with(ValueError("fail")))
        assert fast_breaker.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(fast_breaker.recovery_timeout + 0.05)

        # Next call should be allowed (HALF_OPEN) and succeed
        result = await fast_breaker.call(_success)
        assert result == "ok"
        assert fast_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(
        self, fast_breaker: CircuitBreaker
    ) -> None:
        """If the trial call in HALF_OPEN fails, circuit goes back to OPEN."""
        # Trip the circuit
        for _ in range(fast_breaker.failure_threshold):
            with pytest.raises(ValueError):
                await fast_breaker.call(lambda: _fail_with(ValueError("fail")))
        assert fast_breaker.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(fast_breaker.recovery_timeout + 0.05)

        # Trial call fails
        with pytest.raises(ValueError):
            await fast_breaker.call(lambda: _fail_with(ValueError("still failing")))

        # Should be back to OPEN
        assert fast_breaker.state == CircuitState.OPEN


class TestCircuitBreakerReset:
    """Test manual reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_returns_to_closed(self, fast_breaker: CircuitBreaker) -> None:
        """reset() should force the circuit back to CLOSED state."""
        # Trip the circuit
        for _ in range(fast_breaker.failure_threshold):
            with pytest.raises(ValueError):
                await fast_breaker.call(lambda: _fail_with(ValueError("fail")))
        assert fast_breaker.state == CircuitState.OPEN

        # Manual reset
        fast_breaker.reset()

        assert fast_breaker.state == CircuitState.CLOSED
        assert fast_breaker.stats["failure_count"] == 0


class TestCircuitBreakerStats:
    """Test the observability stats."""

    @pytest.mark.asyncio
    async def test_stats_track_all_counters(self, fast_breaker: CircuitBreaker) -> None:
        """Stats should accurately track calls, successes, failures, rejections."""
        await fast_breaker.call(_success)  # success
        with pytest.raises(ValueError):
            await fast_breaker.call(lambda: _fail_with(ValueError("fail")))  # failure

        stats = fast_breaker.stats
        assert stats["total_calls"] == 2
        assert stats["total_successes"] == 1
        assert stats["total_failures"] == 1
        assert stats["total_rejected"] == 0
        assert stats["name"] == "test"
        assert stats["state"] == "closed"
