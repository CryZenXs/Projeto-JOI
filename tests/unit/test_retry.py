"""Tests for app.core.retry.retry_with_backoff."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.core.retry import MaxRetriesExceeded, retry_with_backoff
from app.llm.errors import LLMError, LLMRateLimitError, LLMTimeoutError


class TestRetryWithBackoff:
    """Test the retry_with_backoff function."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self) -> None:
        """If func succeeds immediately, no retry needed."""
        call_count = 0

        async def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry_with_backoff(
            succeed,
            retry_on=(LLMError,),
            max_attempts=3,
            base_delay_seconds=0.01,
            provider="test",
        )

        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_retryable_error(self) -> None:
        """Should retry when a retryable exception is raised."""
        call_count = 0

        async def fail_then_succeed() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMRateLimitError(provider="test", retry_after_seconds=0.01)
            return "success"

        result = await retry_with_backoff(
            fail_then_succeed,
            retry_on=(LLMRateLimitError, LLMTimeoutError),
            max_attempts=5,
            base_delay_seconds=0.01,
            provider="test",
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_max_retries_after_all_attempts(self) -> None:
        """Should raise MaxRetriesExceeded after exhausting all attempts."""
        call_count = 0

        async def always_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise LLMTimeoutError(provider="test", timeout_seconds=1.0)

        with pytest.raises(MaxRetriesExceeded) as exc_info:
            await retry_with_backoff(
                always_fail,
                retry_on=(LLMTimeoutError,),
                max_attempts=3,
                base_delay_seconds=0.01,
                provider="test",
            )

        assert exc_info.value.attempts == 3
        assert exc_info.value.provider == "test"
        assert isinstance(exc_info.value.last_exception, LLMTimeoutError)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_does_not_retry_on_non_retryable_error(self) -> None:
        """Non-retryable exceptions should propagate immediately."""
        call_count = 0

        async def raise_non_retryable() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            await retry_with_backoff(
                raise_non_retryable,
                retry_on=(LLMTimeoutError,),  # ValueError not in retry_on
                max_attempts=3,
                base_delay_seconds=0.01,
                provider="test",
            )

        assert call_count == 1  # No retry attempted

    @pytest.mark.asyncio
    async def test_respects_retry_after_hint(self) -> None:
        """Should use retry_after_seconds from the exception if present."""
        delays: list[float] = []

        async def fail_with_retry_after() -> None:
            raise LLMRateLimitError(
                provider="test",
                retry_after_seconds=0.05,  # 50ms hint
            )

        original_sleep = asyncio.sleep

        async def mock_sleep(seconds: float) -> None:
            delays.append(seconds)
            await original_sleep(0)  # don't actually wait

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(asyncio, "sleep", mock_sleep)

            with pytest.raises(MaxRetriesExceeded):
                await retry_with_backoff(
                    fail_with_retry_after,
                    retry_on=(LLMRateLimitError,),
                    max_attempts=2,
                    base_delay_seconds=1.0,  # would normally be 1s, 2s
                    provider="test",
                )

        # Should have used the retry_after hint (0.05) instead of exponential (1.0)
        assert len(delays) == 1  # 2 attempts = 1 delay
        assert delays[0] <= 0.1  # close to retry_after hint (with jitter)

    @pytest.mark.asyncio
    async def test_max_attempts_one_disables_retries(self) -> None:
        """max_attempts=1 should not retry at all."""
        call_count = 0

        async def always_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise LLMTimeoutError(provider="test", timeout_seconds=1.0)

        with pytest.raises(MaxRetriesExceeded):
            await retry_with_backoff(
                always_fail,
                retry_on=(LLMTimeoutError,),
                max_attempts=1,
                base_delay_seconds=0.01,
                provider="test",
            )

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_delay_cap(self) -> None:
        """Delay should not exceed max_delay_seconds."""
        delays: list[float] = []

        async def always_fail() -> None:
            raise LLMTimeoutError(provider="test", timeout_seconds=1.0)

        async def mock_sleep(seconds: float) -> None:
            delays.append(seconds)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(asyncio, "sleep", mock_sleep)

            with pytest.raises(MaxRetriesExceeded):
                await retry_with_backoff(
                    always_fail,
                    retry_on=(LLMTimeoutError,),
                    max_attempts=5,
                    base_delay_seconds=10.0,  # would be 10, 20, 40, 80
                    max_delay_seconds=15.0,  # capped at 15
                    provider="test",
                )

        # All delays should be <= max_delay_seconds (with jitter ±25%)
        for delay in delays:
            assert delay <= 15.0 * 1.25  # max with jitter

    def test_max_retries_exceeded_str_representation(self) -> None:
        """MaxRetriesExceeded should have a useful string representation."""
        exc = MaxRetriesExceeded(
            "All attempts failed",
            attempts=3,
            last_exception=LLMTimeoutError(provider="groq", timeout_seconds=30.0),
            provider="groq",
        )

        s = str(exc)
        assert "groq" in s
        assert "attempts=3" in s
        assert "LLMTimeoutError" in s
