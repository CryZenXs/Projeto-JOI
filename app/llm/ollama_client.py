"""Projeto JOI - Ollama local LLM client implementation.

Wraps the official `ollama` Python SDK with the same LLMClient interface
as GroqClient. This enables transparent fallback: when Groq fails, the
router can switch to Ollama without the caller knowing.

Design notes:
- Uses the official ollama-python SDK (async API)
- Streaming via async generators (matches GroqClient interface)
- Same error classification as GroqClient (LLMAuthenticationError, etc.)
- Lower default timeout than Groq (local server is faster to fail)
- No API key needed (Ollama runs locally without auth)

When to use Ollama vs Groq:
- Ollama: privacy mode, offline, rate limit fallback, dev/testing
- Groq: primary (lower latency, larger models, more tokens/s)
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from ollama import AsyncClient

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
    LLMConnectionError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    classify_http_error,
)

logger = get_logger(__name__)


class OllamaClient(LLMClient):
    """Async client for local Ollama server.

    Ollama runs entirely on the user's machine (no cloud, no API key).
    This makes it ideal for:
    - Privacy mode (sensitive conversations don't leave the device)
    - Offline operation (when internet is unavailable)
    - Fallback when Groq is down or rate-limited

    Usage:
        client = OllamaClient(default_model="llama3.1:8b")
        response = await client.complete(request)
        # or:
        async for chunk in client.stream(request):
            print(chunk.content, end="", flush=True)
    """

    provider_name: str = "ollama"

    def __init__(
        self,
        host: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the Ollama client.

        Args:
            host: Ollama server URL. Defaults to settings.ollama_host.
            default_model: Default model ID. Defaults to settings.ollama_model.
            timeout: Request timeout in seconds. Defaults to settings.ollama_timeout_seconds.
        """
        super().__init__(default_model=default_model or settings.ollama_model)

        self._host = host or settings.ollama_host
        self._timeout = timeout or settings.ollama_timeout_seconds

        # Initialize the underlying SDK client
        self._client = AsyncClient(host=self._host, timeout=self._timeout)

        logger.info(
            "llm.client.initialized",
            provider=self.provider_name,
            default_model=self.default_model,
            host=self._host,
            timeout_s=self._timeout,
        )

    # ─── Public API ───────────────────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a complete (non-streaming) response from Ollama."""
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
        )

        try:
            response: dict[str, Any] = await self._client.chat(
                model=model,
                messages=self._convert_messages(request.messages),
                stream=False,
                options={
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                    "top_p": request.top_p,
                    "stop": request.stop if request.stop else None,
                },
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Ollama timeout after {self._timeout}s",
                provider=self.provider_name,
                timeout_seconds=self._timeout,
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self._host}: {exc}",
                provider=self.provider_name,
                cause=exc,
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise classify_http_error(
                exc.response.status_code,
                provider=self.provider_name,
                response_body=exc.response.text,
            ) from exc
        except Exception as exc:
            # Ollama SDK may raise its own exceptions; wrap them
            raise LLMProviderError(
                f"Unexpected Ollama error: {type(exc).__name__}: {exc}",
                provider=self.provider_name,
            ) from exc

        latency_ms = (time.monotonic() - start) * 1000

        # Parse response (Ollama's response structure)
        message = response.get("message", {})
        content = message.get("content", "")
        finish_reason = "stop"
        if response.get("done_reason") == "length":
            finish_reason = "length"

        # Ollama provides usage stats in different fields
        prompt_eval_count = response.get("prompt_eval_count", 0) or 0
        eval_count = response.get("eval_count", 0) or 0
        usage = TokenUsage(
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_tokens=prompt_eval_count + eval_count,
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
            model=response.get("model", model),
            usage=usage,
            finish_reason=finish_reason,  # type: ignore[arg-type]
            response_id=f"ollama_{int(start * 1000)}",
            latency_ms=latency_ms,
            provider=self.provider_name,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[TokenChunk]:
        """Stream response chunks from Ollama."""
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
            stream = await self._client.chat(
                model=model,
                messages=self._convert_messages(request.messages),
                stream=True,
                options={
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                    "top_p": request.top_p,
                    "stop": request.stop if request.stop else None,
                },
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Ollama timeout after {self._timeout}s",
                provider=self.provider_name,
                timeout_seconds=self._timeout,
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self._host}: {exc}",
                provider=self.provider_name,
                cause=exc,
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                f"Unexpected Ollama error starting stream: {type(exc).__name__}: {exc}",
                provider=self.provider_name,
            ) from exc

        # Consume the stream
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = "stop"

        try:
            async for raw_chunk in stream:
                chunk_count += 1
                if ttft_ms is None:
                    ttft_ms = (time.monotonic() - start) * 1000

                parsed = self._parse_stream_chunk(raw_chunk)
                if parsed is not None:
                    if parsed.usage:
                        prompt_tokens = parsed.usage.prompt_tokens
                        completion_tokens = parsed.usage.completion_tokens
                    if parsed.finish_reason:
                        finish_reason = parsed.finish_reason
                    yield parsed
        except httpx.HTTPStatusError as exc:
            raise classify_http_error(
                exc.response.status_code,
                provider=self.provider_name,
                response_body=exc.response.text,
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

    async def health_check(self) -> bool:
        """Check if Ollama server is reachable.

        Uses the lightweight /api/tags endpoint (no model invocation).
        """
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._host}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client. Call on app shutdown."""
        # ollama-python's AsyncClient doesn't expose a close method directly
        # The httpx client inside it will be garbage collected
        pass

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, str]]:
        """Convert our LLMMessage list to Ollama's expected format.

        Ollama uses the same OpenAI-compatible format as Groq:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        return [
            {
                "role": msg.role.value,
                "content": msg.content,
            }
            for msg in messages
        ]

    def _parse_stream_chunk(self, chunk: dict[str, Any]) -> TokenChunk | None:
        """Parse an Ollama stream chunk into our TokenChunk format.

        Ollama stream chunks have this structure:
        {
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "token text"},
            "done": false  # true on final chunk
        }

        The final chunk (done=true) also contains:
        {
            "total_duration": ...,
            "prompt_eval_count": N,
            "eval_count": M,
            "done_reason": "stop" | "length"
        }
        """
        if not chunk:
            return None

        message = chunk.get("message", {})
        content = message.get("content", "")
        is_done = chunk.get("done", False)

        if is_done:
            # Final chunk: extract usage stats
            prompt_tokens = chunk.get("prompt_eval_count", 0) or 0
            completion_tokens = chunk.get("eval_count", 0) or 0
            done_reason = chunk.get("done_reason", "stop")
            finish_reason = "length" if done_reason == "length" else "stop"

            return TokenChunk(
                content=content,  # usually empty on final chunk
                finish_reason=finish_reason,  # type: ignore[arg-type]
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

        # Intermediate chunk: just content
        if not content:
            return None  # skip empty chunks

        return TokenChunk(
            content=content,
            finish_reason=None,
        )
