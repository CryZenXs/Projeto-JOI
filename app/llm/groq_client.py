"""Projeto JOI - Groq Cloud LLM client implementation.

Wraps the official `groq` Python SDK with:
- Async-first design (matches FastAPI's event loop)
- Streaming via async iterators (NOT callbacks)
- Structured error classification (auth/rate-limit/timeout/provider)
- Circuit breaker integration
- Latency measurement (TTFT - Time To First Token)
- Token usage tracking for budget enforcement
- Structured logging with request correlation

The Groq SDK was chosen over raw httpx because:
- It handles SSE parsing correctly (edge cases with partial chunks)
- It provides typed objects (ChatCompletion, ChatCompletionChunk)
- It receives updates when Groq adds new features (tool calls, vision, etc.)
- We still wrap it to avoid vendor lock-in at the call site
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import groq
from groq import AsyncGroq
from groq.types.chat import ChatCompletion, ChatCompletionChunk

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.base import (
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMRole,
    TokenChunk,
    TokenUsage,
)
from app.llm.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    classify_http_error,
)

if TYPE_CHECKING:
    from app.core.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)


class GroqClient(LLMClient):
    """Async client for Groq Cloud API.

    Wraps the official groq-python SDK. Designed to be injected into
    the LLMRouter (Part 1.3) alongside an OllamaClient for fallback.

    Usage:
        client = GroqClient(api_key="gsk_...", default_model="llama-3.1-70b-versatile")
        response = await client.complete(request)
        # or:
        async for chunk in client.stream(request):
            print(chunk.content, end="", flush=True)
    """

    provider_name: str = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize the Groq client.

        Args:
            api_key: Groq API key. Defaults to settings.groq_api_key.
            default_model: Default model ID. Defaults to settings.groq_model_primary.
            timeout: Request timeout in seconds. Defaults to settings.groq_timeout_seconds.
            max_retries: SDK-level retries for transient errors. Defaults to settings.groq_max_retries.
            circuit_breaker: Optional circuit breaker for resilience.
        """
        super().__init__(default_model=default_model or settings.groq_model_primary)

        self._api_key = api_key or settings.groq_api_key.get_secret_value()
        if not self._api_key:
            raise LLMAuthenticationError(
                "GROQ_API_KEY not configured. Get one at https://console.groq.com",
                provider=self.provider_name,
            )

        self._timeout = timeout or settings.groq_timeout_seconds
        self._max_retries = max_retries if max_retries is not None else settings.groq_max_retries
        self._circuit = circuit_breaker

        # Initialize the underlying SDK client
        self._client = AsyncGroq(
            api_key=self._api_key,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

        logger.info(
            "llm.client.initialized",
            provider=self.provider_name,
            default_model=self.default_model,
            timeout_s=self._timeout,
            max_retries=self._max_retries,
            circuit_breaker_enabled=self._circuit is not None,
        )

    # ─── Public API ───────────────────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a complete (non-streaming) response from Groq."""
        if self._circuit:
            return await self._circuit.call(lambda: self._do_complete(request))
        return await self._do_complete(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[TokenChunk]:
        """Stream response chunks from Groq.

        Note: circuit breaker protection for streaming is more nuanced.
        We check the circuit BEFORE starting the stream, but once streaming
        has begun, individual chunk errors don't trip the circuit (they're
        often transient mid-stream).
        """
        if self._circuit and self._circuit.state.value == "open":
            from app.core.circuit_breaker import CircuitOpenError
            raise CircuitOpenError(
                provider=self.provider_name,
                retry_after_seconds=self._circuit.recovery_timeout,
            )

        async for chunk in self._do_stream(request):
            yield chunk

    async def health_check(self) -> bool:
        """Verify Groq API is reachable and the key is valid.

        Uses the lightweight /models endpoint (no tokens consumed).
        """
        try:
            await self._client.models.list()
            return True
        except groq.AuthenticationError:
            return False
        except Exception as exc:
            logger.warning(
                "llm.health_check.failed",
                provider=self.provider_name,
                error=type(exc).__name__,
            )
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client. Call on app shutdown."""
        await self._client.close()

    # ─── Internal: actual Groq API calls ──────────────────────────────────

    async def _do_complete(self, request: LLMRequest) -> LLMResponse:
        """Make the actual non-streaming API call to Groq."""
        model = self._resolve_model(request)
        start = time.monotonic()
        request_id = request.request_id or f"req_{int(start * 1000)}"

        logger.info(
            "llm.request.start",
            provider=self.provider_name,
            model=model,
            request_id=request_id,
            stream=False,
            messages_count=len(request.messages),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        try:
            response: ChatCompletion = await self._client.chat.completions.create(
                model=model,
                messages=self._convert_messages(request.messages),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stream=False,
                stop=request.stop,
                user=request.user_id,
            )
        except groq.AuthenticationError as exc:
            raise LLMAuthenticationError(
                f"Groq authentication failed: {exc}",
                provider=self.provider_name,
            ) from exc
        except groq.RateLimitError as exc:
            raise LLMRateLimitError(
                f"Groq rate limit: {exc}",
                provider=self.provider_name,
            ) from exc
        except groq.APITimeoutError as exc:
            raise LLMTimeoutError(
                f"Groq timeout after {self._timeout}s",
                provider=self.provider_name,
                timeout_seconds=self._timeout,
            ) from exc
        except groq.APIConnectionError as exc:
            raise LLMConnectionError(
                f"Cannot connect to Groq: {exc}",
                provider=self.provider_name,
                cause=exc,
            ) from exc
        except groq.APIStatusError as exc:
            raise classify_http_error(
                exc.status_code,
                provider=self.provider_name,
                response_body=str(exc.body) if exc.body else None,
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                f"Unexpected Groq error: {type(exc).__name__}: {exc}",
                provider=self.provider_name,
            ) from exc

        latency_ms = (time.monotonic() - start) * 1000

        # Extract content and usage
        if not response.choices:
            raise LLMProviderError(
                "Groq returned no choices",
                provider=self.provider_name,
                status_code=200,
            )

        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = self._normalize_finish_reason(choice.finish_reason)

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

        logger.info(
            "llm.request.complete",
            provider=self.provider_name,
            model=model,
            request_id=request_id,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            finish_reason=finish_reason,
        )

        return LLMResponse(
            content=content,
            model=response.model or model,
            usage=usage,
            finish_reason=finish_reason,
            response_id=response.id,
            latency_ms=latency_ms,
            provider=self.provider_name,
        )

    async def _do_stream(self, request: LLMRequest) -> AsyncIterator[TokenChunk]:
        """Stream chunks from Groq, yielding TokenChunk objects."""
        model = self._resolve_model(request)
        start = time.monotonic()
        request_id = request.request_id or f"req_{int(start * 1000)}"
        ttft_ms: float | None = None
        chunk_count = 0

        logger.info(
            "llm.request.start",
            provider=self.provider_name,
            model=model,
            request_id=request_id,
            stream=True,
            messages_count=len(request.messages),
        )

        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=self._convert_messages(request.messages),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stream=True,
                stop=request.stop,
                user=request.user_id,
            )
        except groq.AuthenticationError as exc:
            raise LLMAuthenticationError(
                f"Groq authentication failed: {exc}",
                provider=self.provider_name,
            ) from exc
        except groq.RateLimitError as exc:
            raise LLMRateLimitError(
                f"Groq rate limit: {exc}",
                provider=self.provider_name,
            ) from exc
        except groq.APITimeoutError as exc:
            raise LLMTimeoutError(
                f"Groq timeout after {self._timeout}s",
                provider=self.provider_name,
                timeout_seconds=self._timeout,
            ) from exc
        except groq.APIConnectionError as exc:
            raise LLMConnectionError(
                f"Cannot connect to Groq: {exc}",
                provider=self.provider_name,
                cause=exc,
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                f"Unexpected Groq error starting stream: {type(exc).__name__}: {exc}",
                provider=self.provider_name,
            ) from exc

        # Consume the stream
        try:
            async for raw_chunk in stream:
                chunk_count += 1
                if ttft_ms is None:
                    ttft_ms = (time.monotonic() - start) * 1000

                parsed = self._parse_stream_chunk(raw_chunk)
                if parsed is not None:
                    yield parsed
        except groq.APIStatusError as exc:
            raise classify_http_error(
                exc.status_code,
                provider=self.provider_name,
                response_body=str(exc.body) if exc.body else None,
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                f"Error during streaming: {type(exc).__name__}: {exc}",
                provider=self.provider_name,
            ) from exc

        total_ms = (time.monotonic() - start) * 1000
        logger.info(
            "llm.stream.complete",
            provider=self.provider_name,
            model=model,
            request_id=request_id,
            ttft_ms=round(ttft_ms, 2) if ttft_ms else None,
            total_ms=round(total_ms, 2),
            chunk_count=chunk_count,
        )

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, str]]:
        """Convert our LLMMessage list to Groq's expected format.

        Groq uses OpenAI-compatible message format:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        return [
            {
                "role": msg.role.value,
                "content": msg.content,
            }
            for msg in messages
        ]

    def _parse_stream_chunk(self, chunk: ChatCompletionChunk) -> TokenChunk | None:
        """Parse a Groq stream chunk into our TokenChunk format.

        Returns None for empty chunks (keep-alives from Groq).
        """
        if not chunk.choices:
            # Some chunks have usage but no choices (final stats chunk)
            if chunk.usage:
                return TokenChunk(
                    content="",
                    finish_reason="stop",
                    usage=TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    ),
                )
            return None

        delta = chunk.choices[0].delta
        finish_reason = self._normalize_finish_reason(chunk.choices[0].finish_reason)

        return TokenChunk(
            content=delta.content or "",
            finish_reason=finish_reason,
        )

    @staticmethod
    def _normalize_finish_reason(
        reason: str | None,
    ) -> LLMResponse.model_fields["finish_reason"].annotation:  # type: ignore[name-defined]
        """Normalize Groq's finish_reason to our enum.

        Groq uses: "stop", "length", "tool_calls", "content_filter"
        We use:   "stop", "length", "tool_call", "content_filter"
        """
        if reason is None:
            return None
        if reason == "tool_calls":
            return "tool_call"
        if reason in ("stop", "length", "content_filter"):
            return reason  # type: ignore[return-value]
        return "stop"  # default for unknown reasons
