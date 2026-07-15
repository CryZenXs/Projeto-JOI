"""Projeto JOI - Retry with exponential backoff.

Provides a generic async retry utility for transient failures.
Used by the LLMRouter to retry requests that fail with rate limit
or timeout errors before falling back to the next provider.

Design:
- Exponential backoff with optional jitter (avoids thundering herd)
- Configurable max attempts and base delay
- Only retries on specified exception types (don't retry auth errors)
- Respects Retry-After header when present (for 429 responses)
- Logs each retry attempt for observability

Usage:
    from app.core.retry import retry_with_backoff

    try:
        result = await retry_with_backoff(
            func=lambda: groq_client.complete(request),
            retry_on=(LLMRateLimitError, LLMTimeoutError, LLMConnectionError),
            max_attempts=3,
            base_delay_seconds=1.0,
        )
    except MaxRetriesExceeded as exc:
        # All retries failed, fall back to next provider
        ...
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# Lazy import to avoid circular dependency:
# app.core.retry -> app.llm.errors -> app.llm.__init__ -> app.llm.router -> app.core.retry
def _get_llm_error_class():
    from app.llm.errors import LLMError
    return LLMError


class MaxRetriesExceeded(Exception):
    """Raised when all retry attempts have been exhausted.

    The original exception is preserved in `__cause__` for inspection.
    The caller should catch this and fall back to the next strategy
    (e.g., switch to a different LLM provider).

    Note: This class is dynamically registered as a subclass of LLMError
    at import time to allow circuit breakers to catch it. See
    _register_as_llm_error() below.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_exception: Exception,
        provider: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception
        self.provider = provider
        self.__cause__ = last_exception

    def __str__(self) -> str:
        return (
            f"[{self.provider}] {self.args[0]} "
            f"(attempts={self.attempts}, last_error={type(self.last_exception).__name__})"
        )


async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    *,
    retry_on: tuple[type[Exception], ...],
    max_attempts: int = 3,
    base_delay_seconds: float = 1.0,
    max_delay_seconds: float = 30.0,
    jitter: bool = True,
    provider: str = "unknown",
    request_id: str | None = None,
) -> T:
    """Execute an async function with retry and exponential backoff.

    Args:
        func: Zero-argument async callable to execute.
        retry_on: Tuple of exception types that should trigger a retry.
            Exceptions NOT in this tuple are re-raised immediately.
        max_attempts: Maximum number of attempts (including the first).
            Setting this to 1 disables retries.
        base_delay_seconds: Initial delay before the first retry.
            Subsequent delays double (with optional jitter).
        max_delay_seconds: Upper bound on delay between retries.
        jitter: If True, add random jitter (±25%) to delays to avoid
            thundering herd when multiple clients retry simultaneously.
        provider: Name of the provider (for logging).
        request_id: Optional correlation ID (for logging).

    Returns:
        The result of `func()` if it succeeds within max_attempts.

    Raises:
        MaxRetriesExceeded: If all attempts fail with retryable exceptions.
        Any non-retryable exception raised by `func()` (re-raised immediately).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = await func()
            if attempt > 1:
                logger.info(
                    "retry.succeeded",
                    provider=provider,
                    attempt=attempt,
                    request_id=request_id,
                )
            return result

        except retry_on as exc:
            last_exception = exc

            # Check if the exception carries a retry_after hint (429 responses)
            retry_after = getattr(exc, "retry_after_seconds", None)

            if attempt >= max_attempts:
                logger.warning(
                    "retry.exhausted",
                    provider=provider,
                    attempts=attempt,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    request_id=request_id,
                )
                raise MaxRetriesExceeded(
                    f"All {max_attempts} attempts failed for {provider}",
                    attempts=attempt,
                    last_exception=exc,
                    provider=provider,
                ) from exc

            # Calculate delay
            if retry_after is not None and retry_after > 0:
                # Respect the server's hint (e.g., 429 Retry-After header)
                delay = min(retry_after, max_delay_seconds)
            else:
                # Exponential backoff: base * 2^(attempt-1)
                delay = base_delay_seconds * (2 ** (attempt - 1))
                delay = min(delay, max_delay_seconds)

            # Add jitter to avoid thundering herd
            if jitter and delay > 0:
                jitter_factor = random.uniform(0.75, 1.25)  # ±25%
                delay = delay * jitter_factor

            logger.warning(
                "retry.attempt_failed",
                provider=provider,
                attempt=attempt,
                max_attempts=max_attempts,
                delay_seconds=round(delay, 2),
                error_type=type(exc).__name__,
                error_message=str(exc),
                request_id=request_id,
                retry_after_hint=retry_after,
            )

            await asyncio.sleep(delay)

        # Note: non-retryable exceptions are NOT caught here, so they
        # propagate immediately to the caller without retry.

    # Should not reach here, but just in case
    raise MaxRetriesExceeded(
        f"Unexpected exit from retry loop for {provider}",
        attempts=max_attempts,
        last_exception=last_exception or RuntimeError("No exception captured"),
        provider=provider,
    )
