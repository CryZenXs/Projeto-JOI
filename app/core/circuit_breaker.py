"""Projeto JOI - Circuit Breaker pattern for LLM resilience.

The circuit breaker protects against cascading failures when a provider
(Groq, Ollama) goes down. Instead of letting every request fail slowly,
it "opens" after N consecutive failures and short-circuits subsequent
calls to the fallback provider immediately.

States:
    CLOSED    → Normal operation. Requests go through. Failures counted.
    OPEN      → Tripped. All requests fail fast with CircuitOpenError.
                After recovery_timeout, moves to HALF_OPEN.
    HALF_OPEN → One trial request allowed. If it succeeds → CLOSED.
                If it fails → back to OPEN (with doubled timeout).

This implementation is asyncio-safe and stateless across processes
(in-memory only; for multi-process deployments, use Redis-backed version).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, TypeVar

from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """The three states of a circuit breaker."""

    CLOSED = "closed"  # Normal: requests pass through
    OPEN = "open"  # Tripped: requests fail fast
    HALF_OPEN = "half_open"  # Trial: one request allowed


class CircuitOpenError(Exception):
    """Raised when calling a service whose circuit is OPEN.

    This is NOT an LLMError — it's a control-flow signal that the router
    should immediately try the fallback provider instead of waiting.
    """

    def __init__(self, provider: str, retry_after_seconds: float) -> None:
        self.provider = provider
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Circuit OPEN for {provider!r} — retry in {retry_after_seconds:.1f}s"
        )


class CircuitBreaker:
    """Async circuit breaker for protecting LLM provider calls.

    Usage:
        breaker = CircuitBreaker(name="groq", failure_threshold=5, recovery_timeout=60)

        try:
            result = await breaker.call(lambda: groq_client.complete(request))
        except CircuitOpenError:
            # Fallback to Ollama
            result = await ollama_client.complete(request)

    Thread-safety: This implementation uses an asyncio.Lock to protect
    state transitions. Safe to share across coroutines in the same event
    loop. NOT safe across processes (use Redis-backed version for that).
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        expected_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        """Initialize the circuit breaker.

        Args:
            name: Human-readable name (e.g., "groq", "ollama") for logging.
            failure_threshold: Number of consecutive failures before opening.
            recovery_timeout: Seconds to wait before trying HALF_OPEN.
            half_open_max_calls: Concurrent calls allowed in HALF_OPEN state.
            expected_exceptions: Exception types that count as "failures".
                Other exceptions are re-raised without affecting the circuit.
        """
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout = max(1.0, recovery_timeout)
        self.half_open_max_calls = max(1, half_open_max_calls)
        self.expected_exceptions = expected_exceptions

        # State (mutable)
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

        # Stats (for observability)
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._total_rejected = 0  # calls rejected while OPEN

    # ─── Public API ───────────────────────────────────────────────────────

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute an async function through the circuit breaker.

        Args:
            func: A zero-argument async callable (e.g., lambda: client.complete(req)).

        Returns:
            The result of `func()` if it succeeds.

        Raises:
            CircuitOpenError: If the circuit is OPEN (don't retry — fallback).
            Any exception raised by `func` (if it's in expected_exceptions,
            the circuit counts it as a failure).
        """
        async with self._lock:
            await self._before_call()
            self._total_calls += 1

        # Execute outside the lock to allow concurrency
        try:
            result = await func()
        except self.expected_exceptions as exc:
            await self._on_failure(exc)
            raise
        except Exception as exc:
            # Unexpected exception — re-raise without affecting circuit
            logger.warning(
                "circuit.unexpected_exception",
                circuit=self.name,
                error_type=type(exc).__name__,
            )
            raise
        else:
            await self._on_success()
            return result

    @property
    def state(self) -> CircuitState:
        """Current circuit state (read-only)."""
        return self._state

    @property
    def stats(self) -> dict[str, Any]:
        """Observability stats (for /health endpoint and metrics)."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "total_rejected": self._total_rejected,
        }

    def reset(self) -> None:
        """Force-reset the circuit to CLOSED state (for testing/manual ops)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        logger.info("circuit.reset", circuit=self.name)

    # ─── Internal state machine ──────────────────────────────────────────

    async def _before_call(self) -> None:
        """Check state and decide whether to allow the call."""
        if self._state == CircuitState.CLOSED:
            return  # always allow

        if self._state == CircuitState.OPEN:
            # Check if recovery period has elapsed
            if self._should_transition_to_half_open():
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(
                    "circuit.transition",
                    circuit=self.name,
                    from_state="open",
                    to_state="half_open",
                )
            else:
                self._total_rejected += 1
                retry_after = self._retry_after_seconds()
                logger.warning(
                    "circuit.rejected",
                    circuit=self.name,
                    retry_after=retry_after,
                )
                raise CircuitOpenError(provider=self.name, retry_after_seconds=retry_after)

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                self._total_rejected += 1
                raise CircuitOpenError(
                    provider=self.name,
                    retry_after_seconds=self.recovery_timeout,
                )
            self._half_open_calls += 1

    async def _on_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._total_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    # Recovery successful — close the circuit
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._last_failure_time = None
                    logger.info(
                        "circuit.transition",
                        circuit=self.name,
                        from_state="half_open",
                        to_state="closed",
                    )
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success (consecutive failures only)
                self._failure_count = 0

    async def _on_failure(self, exc: Exception) -> None:
        """Record a failed call."""
        async with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Trial failed — back to OPEN with full timeout
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning(
                    "circuit.transition",
                    circuit=self.name,
                    from_state="half_open",
                    to_state="open",
                    reason="trial call failed",
                    error=type(exc).__name__,
                )
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "circuit.transition",
                        circuit=self.name,
                        from_state="closed",
                        to_state="open",
                        failure_count=self._failure_count,
                        threshold=self.failure_threshold,
                    )

    def _should_transition_to_half_open(self) -> bool:
        """Check if enough time has passed since last failure."""
        if self._last_failure_time is None:
            return True
        elapsed = time.monotonic() - self._last_failure_time
        return elapsed >= self.recovery_timeout

    def _retry_after_seconds(self) -> float:
        """How long until the circuit might allow a trial call."""
        if self._last_failure_time is None:
            return 0.0
        elapsed = time.monotonic() - self._last_failure_time
        remaining = self.recovery_timeout - elapsed
        return max(0.0, remaining)
