"""Projeto JOI - Mock LLM client for testing.

This client doesn't make any network calls — it returns canned responses
based on configurable scripts. This makes tests:
- Deterministic (same input → same output, every time)
- Fast (no network latency)
- Free (no API tokens consumed)
- Offline (no connectivity required)

Usage in tests:
    mock = MockLLMClient(responses=["Hello!", "How are you?"])
    response = await mock.complete(LLMRequest(messages=[...]))
    assert response.content == "Hello!"

For streaming:
    mock = MockLLMClient(stream_chunks=["Hello", " ", "world", "!"])
    async for chunk in mock.stream(request):
        print(chunk.content, end="", flush=True)
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from app.llm.base import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    LLMRole,
    TokenChunk,
    TokenUsage,
)
from app.llm.errors import LLMError


class MockLLMClient(LLMClient):
    """In-memory LLM client that returns pre-configured responses.

    Supports two modes:
    1. **Scripted responses**: pass a list of responses; each call returns the next.
    2. **Echo mode**: if no responses are configured, echoes back the last user message.

    Also supports error simulation for testing the router/circuit breaker:
    - Set `error_to_raise` to make the next call raise an exception.
    """

    provider_name: str = "mock"

    def __init__(
        self,
        default_model: str = "mock-model",
        responses: list[str] | None = None,
        stream_chunks: list[list[str]] | None = None,
        error_to_raise: LLMError | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        super().__init__(default_model=default_model)
        self._responses = list(responses) if responses else []
        self._stream_chunks = list(stream_chunks) if stream_chunks else []
        self._error_to_raise = error_to_raise
        self._latency_ms = latency_ms
        self._call_count: int = 0
        self._last_request: LLMRequest | None = None

    # ─── Public API ───────────────────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the next scripted response or echo the user's message."""
        self._last_request = request
        self._call_count += 1

        # Simulate latency (useful for testing timeout logic)
        if self._latency_ms > 0:
            import asyncio
            await asyncio.sleep(self._latency_ms / 1000.0)

        # Simulate error
        if self._error_to_raise:
            err = self._error_to_raise
            self._error_to_raise = None  # one-shot
            raise err

        # Determine response content
        if self._responses:
            content = self._responses.pop(0)
        else:
            # Echo mode: return the last user message
            content = self._echo_last_user_message(request)

        # Estimate token counts (rough: 1 word ≈ 1.3 tokens)
        prompt_tokens = sum(len(m.content.split()) for m in request.messages) * 2
        completion_tokens = max(1, len(content.split()) * 2)

        return LLMResponse(
            content=content,
            model=request.model or self.default_model or "mock-model",
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason="stop",
            response_id=f"mock-{self._call_count}",
            latency_ms=self._latency_ms,
            provider=self.provider_name,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[TokenChunk]:
        """Yield pre-configured chunks or split the response into words."""
        self._last_request = request
        self._call_count += 1

        if self._latency_ms > 0:
            import asyncio
            await asyncio.sleep(self._latency_ms / 1000.0)

        if self._error_to_raise:
            err = self._error_to_raise
            self._error_to_raise = None
            raise err

        # Determine chunks to stream
        if self._stream_chunks:
            chunks = self._stream_chunks.pop(0)
        else:
            # Split a scripted response into word-chunks
            content = self._responses.pop(0) if self._responses else self._echo_last_user_message(request)
            chunks = content.split()

        prompt_tokens = sum(len(m.content.split()) for m in request.messages) * 2
        completion_tokens = max(1, sum(len(c.split()) for c in chunks) * 2)

        # Yield content chunks
        for chunk_text in chunks:
            yield TokenChunk(
                content=chunk_text + (" " if not chunk_text.endswith((" ", "\n", ".")) else ""),
                finish_reason=None,
            )

        # Final chunk with usage stats
        yield TokenChunk(
            content="",
            finish_reason="stop",
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def health_check(self) -> bool:
        """Mock is always healthy unless configured otherwise."""
        return self._error_to_raise is None

    # ─── Test helpers (not part of LLMClient interface) ──────────────────

    @property
    def call_count(self) -> int:
        """Number of times complete() or stream() was called."""
        return self._call_count

    @property
    def last_request(self) -> LLMRequest | None:
        """The most recent request received (for assertions)."""
        return self._last_request

    def queue_response(self, response: str) -> None:
        """Add a response to the queue (for sequential test scenarios)."""
        self._responses.append(response)

    def queue_error(self, error: LLMError) -> None:
        """Configure the next call to raise this error."""
        self._error_to_raise = error

    # ─── Private helpers ─────────────────────────────────────────────────

    def _echo_last_user_message(self, request: LLMRequest) -> str:
        """Echo the last user message (default behavior when no scripts)."""
        for msg in reversed(request.messages):
            if msg.role == LLMRole.USER:
                return f"[mock echo] {msg.content}"
        return "[mock] No user message found"
