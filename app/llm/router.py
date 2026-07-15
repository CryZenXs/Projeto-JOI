"""Projeto JOI - LLM Router with automatic fallback.

The LLMRouter is the single entry point for all LLM requests. It:
1. Tries the primary provider (Groq) first
2. On failure, retries with exponential backoff
3. If all retries fail, falls back to the secondary provider (Ollama)
4. Each provider has its own circuit breaker for resilience
5. Supports "privacy mode" — force Ollama for sensitive conversations

This is the brain that decides which LLM handles each request. The
caller (chat endpoint, persona engine, etc.) never needs to know which
provider actually served the request — the router abstracts this away.

Architecture:
    Request ──► LLMRouter ──► [Privacy?] ──► Ollama only
                      │
                      └──► [Normal] ──► Groq (with retry)
                                          │
                                          ├─ Success ──► Return
                                          │
                                          └─ Fail ──► Ollama
                                                         │
                                                         ├─ Success ──► Return
                                                         │
                                                         └─ Fail ──► Raise
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from app.core.config import settings
from app.core.logging import get_logger
from app.core.retry import MaxRetriesExceeded, retry_with_backoff
from app.llm.base import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    TokenChunk,
)
from app.llm.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = get_logger(__name__)


# ─── Router configuration ────────────────────────────────────────────────


class RoutingMode(str, Enum):
    """How the router should handle this request."""

    NORMAL = "normal"  # Try primary, fall back to secondary on failure
    PRIVACY = "privacy"  # Force secondary (local) provider only
    PRIMARY_ONLY = "primary_only"  # Force primary only (no fallback)


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider in the router."""

    name: str  # "groq" or "ollama"
    client: LLMClient
    circuit_breaker: CircuitBreaker
    priority: int  # lower = higher priority (1 = primary)


@dataclass
class RoutingDecision:
    """Record of how a request was routed (for observability)."""

    request_id: str
    mode: RoutingMode
    primary_provider: str
    final_provider: str  # which provider actually served the request
    fallback_used: bool
    attempts: list[dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0

    def add_attempt(
        self,
        provider: str,
        attempt: int,
        success: bool,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        """Record a single attempt within this routing decision."""
        self.attempts.append({
            "provider": provider,
            "attempt": attempt,
            "success": success,
            "latency_ms": round(latency_ms, 2),
            "error": error,
        })


class LLMRouter:
    """Routes LLM requests across multiple providers with fallback.

    The router is the SINGLE entry point for all LLM calls in the app.
    It handles:
    - Provider selection (primary vs fallback vs privacy mode)
    - Per-provider circuit breakers (prevent cascading failures)
    - Retry with exponential backoff (for transient errors)
    - Observability (logs every routing decision)

    Usage:
        router = LLMRouter(groq_client, ollama_client)
        response = await router.complete(request)
        # or:
        async for chunk in router.stream(request):
            print(chunk.content)
    """

    def __init__(
        self,
        primary: LLMClient,
        fallback: LLMClient | None = None,
        primary_breaker: CircuitBreaker | None = None,
        fallback_breaker: CircuitBreaker | None = None,
        max_retries: int = 2,
        retry_base_delay_seconds: float = 1.0,
    ) -> None:
        """Initialize the router with primary and fallback providers.

        Args:
            primary: Primary LLM client (usually GroqClient).
            fallback: Fallback LLM client (usually OllamaClient). Can be None
                if no fallback is desired (e.g., during testing).
            primary_breaker: Circuit breaker for the primary provider.
                If None, a default one is created.
            fallback_breaker: Circuit breaker for the fallback provider.
            max_retries: Number of retry attempts per provider before failing over.
            retry_base_delay_seconds: Initial delay for exponential backoff.
        """
        self._primary = primary
        self._fallback = fallback

        # Create default circuit breakers if not provided
        self._primary_breaker = primary_breaker or CircuitBreaker(
            name=f"{primary.provider_name}-primary",
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exceptions=(
                LLMError,  # catches all LLM exceptions
            ),
        )

        self._fallback_breaker = fallback_breaker
        if fallback and not fallback_breaker:
            self._fallback_breaker = CircuitBreaker(
                name=f"{fallback.provider_name}-fallback",
                failure_threshold=3,
                recovery_timeout=30.0,
                expected_exceptions=(LLMError,),
            )

        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay_seconds

        # Stats for observability
        self._total_requests = 0
        self._primary_successes = 0
        self._fallback_successes = 0
        self._total_failures = 0
        self._privacy_mode_requests = 0

        logger.info(
            "router.initialized",
            primary=primary.provider_name,
            fallback=fallback.provider_name if fallback else None,
            max_retries=max_retries,
        )

    # ─── Public API ───────────────────────────────────────────────────────

    async def complete(
        self,
        request: LLMRequest,
        mode: RoutingMode = RoutingMode.NORMAL,
    ) -> LLMResponse:
        """Generate a complete (non-streaming) response."""
        decision = RoutingDecision(
            request_id=request.request_id or f"req_{int(time.monotonic() * 1000)}",
            mode=mode,
            primary_provider=self._primary.provider_name,
            final_provider="",
            fallback_used=False,
        )
        start = time.monotonic()

        try:
            response = await self._route_complete(request, mode, decision)
            decision.final_provider = response.provider
            decision.total_latency_ms = (time.monotonic() - start) * 1000
            # Determine which provider served this request
            # In PRIVACY mode, the fallback is used directly (not as a "fallback")
            if mode == RoutingMode.PRIVACY:
                serving_provider = self._fallback
            else:
                serving_provider = self._fallback if decision.fallback_used else self._primary
            self._record_success(response.provider, mode, provider=serving_provider)
            return response
        except LLMError as exc:
            decision.total_latency_ms = (time.monotonic() - start) * 1000
            self._total_failures += 1
            logger.error(
                "router.request.failed",
                request_id=decision.request_id,
                mode=mode.value,
                error=str(exc),
                attempts=len(decision.attempts),
            )
            raise

        finally:
            self._log_decision(decision)

    async def stream(
        self,
        request: LLMRequest,
        mode: RoutingMode = RoutingMode.NORMAL,
    ) -> AsyncIterator[TokenChunk]:
        """Stream response chunks with automatic fallback.

        Note: Streaming fallback is more complex than non-streaming.
        If the primary fails DURING streaming (after some chunks have
        been emitted), we cannot transparently switch to fallback —
        the client would receive a partial response. In this case,
        we emit an error chunk.

        We only fall back if the primary fails BEFORE streaming starts
        (i.e., during the initial connection/handshake).
        """
        decision = RoutingDecision(
            request_id=request.request_id or f"req_{int(time.monotonic() * 1000)}",
            mode=mode,
            primary_provider=self._primary.provider_name,
            final_provider="",
            fallback_used=False,
        )
        start = time.monotonic()

        # Determine which provider to use
        provider, breaker = self._select_provider(mode, decision)

        if provider is None:
            # Both providers unavailable
            decision.total_latency_ms = (time.monotonic() - start) * 1000
            self._total_failures += 1
            self._log_decision(decision)
            raise LLMProviderError(
                "No LLM provider available (all circuits open or no fallback configured)",
                provider="router",
            )

        # Try streaming from the selected provider
        try:
            # Check circuit breaker before starting
            if breaker and breaker.state == CircuitState.OPEN:
                raise CircuitOpenError(
                    provider=provider.provider_name,
                    retry_after_seconds=breaker.recovery_timeout,
                )

            chunk_count = 0
            async for chunk in provider.stream(request):
                chunk_count += 1
                yield chunk

            # Success
            decision.final_provider = provider.provider_name
            decision.total_latency_ms = (time.monotonic() - start) * 1000
            decision.add_attempt(
                provider=provider.provider_name,
                attempt=1,
                success=True,
                latency_ms=decision.total_latency_ms,
            )
            self._record_success(provider.provider_name, mode, provider=provider)

        except (CircuitOpenError, LLMConnectionError, LLMTimeoutError) as exc:
            # Pre-stream failure: we can fall back
            decision.add_attempt(
                provider=provider.provider_name,
                attempt=1,
                success=False,
                latency_ms=(time.monotonic() - start) * 1000,
                error=str(exc),
            )

            # Try fallback if available and not already using it
            if (
                self._fallback
                and provider is not self._fallback
                and mode != RoutingMode.PRIMARY_ONLY
            ):
                logger.warning(
                    "router.stream.fallback",
                    request_id=decision.request_id,
                    failed_provider=provider.provider_name,
                    fallback_provider=self._fallback.provider_name,
                    error=str(exc),
                )
                decision.fallback_used = True
                decision.final_provider = self._fallback.provider_name
                decision.total_latency_ms = (time.monotonic() - start) * 1000

                async for chunk in self._fallback.stream(request):
                    yield chunk

                self._record_success(self._fallback.provider_name, mode, provider=self._fallback)
            else:
                decision.total_latency_ms = (time.monotonic() - start) * 1000
                self._total_failures += 1
                raise

        except LLMError:
            decision.total_latency_ms = (time.monotonic() - start) * 1000
            self._total_failures += 1
            raise

        finally:
            self._log_decision(decision)

    async def health_check(self) -> dict[str, Any]:
        """Check health of all providers and return status summary."""
        result: dict[str, Any] = {
            "primary": {
                "provider": self._primary.provider_name,
                "healthy": False,
                "circuit_state": self._primary_breaker.state.value,
            },
            "fallback": None,
            "stats": self._get_stats(),
        }

        result["primary"]["healthy"] = await self._primary.health_check()

        if self._fallback:
            result["fallback"] = {
                "provider": self._fallback.provider_name,
                "healthy": False,
                "circuit_state": self._fallback_breaker.state.value
                if self._fallback_breaker
                else "unknown",
            }
            result["fallback"]["healthy"] = await self._fallback.health_check()

        return result

    # ─── Internal routing logic ──────────────────────────────────────────

    async def _route_complete(
        self,
        request: LLMRequest,
        mode: RoutingMode,
        decision: RoutingDecision,
    ) -> LLMResponse:
        """Execute the routing logic for non-streaming requests."""
        provider, breaker = self._select_provider(mode, decision)

        if provider is None:
            raise LLMProviderError(
                "No LLM provider available for this request mode",
                provider="router",
            )

        # Define retryable exceptions
        retryable = (
            LLMRateLimitError,
            LLMTimeoutError,
            LLMConnectionError,
        )

        # Attempt 1: Try primary provider with retries
        # The circuit breaker wraps the retry logic. When retries are exhausted,
        # we unwrap MaxRetriesExceeded to get the underlying LLMError (which the
        # circuit breaker counts as a failure).
        attempt_start = time.monotonic()
        try:
            async def _call_with_retry() -> LLMResponse:
                try:
                    return await retry_with_backoff(
                        func=lambda: provider.complete(request),
                        retry_on=retryable,
                        max_attempts=self._max_retries,
                        base_delay_seconds=self._retry_base_delay,
                        provider=provider.provider_name,
                        request_id=decision.request_id,
                    )
                except MaxRetriesExceeded as exc:
                    # Unwrap: re-raise the underlying LLMError so the circuit
                    # breaker counts it as a failure
                    raise exc.last_exception from exc

            if breaker:
                response = await breaker.call(_call_with_retry)
            else:
                response = await _call_with_retry()

            decision.add_attempt(
                provider=provider.provider_name,
                attempt=1,
                success=True,
                latency_ms=(time.monotonic() - attempt_start) * 1000,
            )
            return response

        except CircuitOpenError as exc:
            # Circuit is open — try fallback if available
            decision.add_attempt(
                provider=provider.provider_name,
                attempt=1,
                success=False,
                latency_ms=(time.monotonic() - attempt_start) * 1000,
                error=str(exc),
            )

            if (
                self._fallback
                and provider is not self._fallback
                and mode != RoutingMode.PRIMARY_ONLY
            ):
                logger.warning(
                    "router.fallback.triggered",
                    request_id=decision.request_id,
                    failed_provider=provider.provider_name,
                    fallback_provider=self._fallback.provider_name,
                    reason=str(exc),
                )
                decision.fallback_used = True
                return await self._try_fallback(request, decision)

            # No fallback available or allowed
            raise

        except LLMError as exc:
            # Non-retryable error (auth, provider error) — try fallback?
            decision.add_attempt(
                provider=provider.provider_name,
                attempt=1,
                success=False,
                latency_ms=(time.monotonic() - attempt_start) * 1000,
                error=str(exc),
            )

            if (
                self._fallback
                and provider is not self._fallback
                and mode != RoutingMode.PRIMARY_ONLY
                and not isinstance(exc, LLMAuthenticationError)
            ):
                # Don't fall back on auth errors (same key would fail on fallback too,
                # but actually Ollama doesn't use keys — so we COULD fall back.
                # For safety, we do fall back on everything except auth.)
                logger.warning(
                    "router.fallback.triggered",
                    request_id=decision.request_id,
                    failed_provider=provider.provider_name,
                    fallback_provider=self._fallback.provider_name,
                    reason=str(exc),
                )
                decision.fallback_used = True
                return await self._try_fallback(request, decision)

            raise

    async def _try_fallback(
        self,
        request: LLMRequest,
        decision: RoutingDecision,
    ) -> LLMResponse:
        """Try the fallback provider (Ollama) when primary fails."""
        if not self._fallback:
            raise LLMProviderError(
                "No fallback provider configured",
                provider="router",
            )

        attempt_start = time.monotonic()
        try:
            if self._fallback_breaker:
                response = await self._fallback_breaker.call(
                    lambda: self._fallback.complete(request)  # type: ignore[union-attr]
                )
            else:
                response = await self._fallback.complete(request)  # type: ignore[union-attr]

            decision.add_attempt(
                provider=self._fallback.provider_name,
                attempt=2,
                success=True,
                latency_ms=(time.monotonic() - attempt_start) * 1000,
            )
            decision.final_provider = self._fallback.provider_name
            return response

        except LLMError as exc:
            decision.add_attempt(
                provider=self._fallback.provider_name,
                attempt=2,
                success=False,
                latency_ms=(time.monotonic() - attempt_start) * 1000,
                error=str(exc),
            )
            raise

    def _select_provider(
        self,
        mode: RoutingMode,
        decision: RoutingDecision,
    ) -> tuple[LLMClient | None, CircuitBreaker | None]:
        """Select which provider to use based on the routing mode.

        Returns:
            Tuple of (client, circuit_breaker) or (None, None) if no
            provider is available for this mode.
        """
        if mode == RoutingMode.PRIVACY:
            # Force fallback (local) provider only
            decision.primary_provider = "(skipped — privacy mode)"
            if self._fallback:
                return self._fallback, self._fallback_breaker
            logger.warning(
                "router.privacy_mode_no_fallback",
                request_id=decision.request_id,
            )
            return None, None

        # Normal or PRIMARY_ONLY: use primary
        return self._primary, self._primary_breaker

    # ─── Observability ───────────────────────────────────────────────────

    def _record_success(self, provider_name: str, mode: RoutingMode, provider: LLMClient | None = None) -> None:
        """Record a successful request for stats.

        Uses object identity (is) when possible to distinguish providers
        that share the same provider_name (e.g., two MockLLMClient instances
        both named "mock").
        """
        self._total_requests += 1
        if mode == RoutingMode.PRIVACY:
            self._privacy_mode_requests += 1

        # Check object identity first (most reliable)
        if provider is not None:
            if provider is self._primary:
                self._primary_successes += 1
                return
            if self._fallback is not None and provider is self._fallback:
                self._fallback_successes += 1
                return

        # Fall back to name matching
        if provider_name == self._primary.provider_name:
            self._primary_successes += 1
        elif self._fallback and provider_name == self._fallback.provider_name:
            self._fallback_successes += 1

    def _get_stats(self) -> dict[str, Any]:
        """Get router statistics for observability."""
        return {
            "total_requests": self._total_requests,
            "primary_successes": self._primary_successes,
            "fallback_successes": self._fallback_successes,
            "total_failures": self._total_failures,
            "privacy_mode_requests": self._privacy_mode_requests,
            "primary_circuit": self._primary_breaker.stats,
            "fallback_circuit": self._fallback_breaker.stats if self._fallback_breaker else None,
        }

    def _log_decision(self, decision: RoutingDecision) -> None:
        """Log the routing decision for observability."""
        logger.info(
            "router.decision",
            request_id=decision.request_id,
            mode=decision.mode.value,
            primary_provider=decision.primary_provider,
            final_provider=decision.final_provider,
            fallback_used=decision.fallback_used,
            total_latency_ms=round(decision.total_latency_ms, 2),
            attempts=len(decision.attempts),
        )
