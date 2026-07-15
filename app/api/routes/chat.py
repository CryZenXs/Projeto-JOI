"""Projeto JOI - Chat endpoints.

Provides two endpoints:
- POST /api/v1/chat        → Non-streaming (returns complete JSON response)
- POST /api/v1/chat/stream → Streaming (Server-Sent Events)

Both endpoints accept the same ChatRequest. The choice between them
depends on the use case:
- Non-streaming: simpler clients, batch processing, testing
- Streaming: chat UIs where perceived latency matters (most user-facing apps)

The streaming endpoint uses Server-Sent Events (SSE) rather than
WebSockets because:
1. SSE is unidirectional (server → client), which matches our use case
2. SSE works over standard HTTP (no upgrade handshake)
3. SSE has automatic reconnection built into browsers
4. SSE is easier to proxy (CDNs, nginx, etc.)
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    DoneEventData,
    ErrorEventData,
    MetaEventData,
    TokenEventData,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.llm.base import LLMClient, LLMRequest
from app.llm.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


# ─── Dependency injection ─────────────────────────────────────────────────
# This function returns the LLMClient instance to use for this request.
# For now, we use a simple factory. In Part 1.3, this will be replaced
# by the LLMRouter that handles Groq→Ollama fallback automatically.


def get_llm_client() -> LLMClient:
    """Dependency: return the LLM client for this request.

    In Part 1.2 (now), this returns a MockLLMClient if no Groq key is
    configured, or a GroqClient if one is. In Part 1.3, this will return
    an LLMRouter that handles fallback automatically.
    """
    if settings.has_groq_key:
        from app.llm.groq_client import GroqClient
        return GroqClient()
    # No Groq key configured — use mock for development
    from app.llm.mock_client import MockLLMClient
    return MockLLMClient()


# ─── Endpoints ────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a complete response (non-streaming)",
    description=(
        "Send a conversation and receive a complete response. "
        "Use this for batch processing or when streaming isn't needed. "
        "For interactive chat UIs, prefer POST /chat/stream for lower perceived latency."
    ),
)
async def chat(
    request: ChatRequest,
    client: LLMClient = Depends(get_llm_client),
) -> ChatResponse:
    """Handle a non-streaming chat request."""
    request_id = str(uuid.uuid4())
    start = time.monotonic()

    logger.info(
        "chat.request.received",
        request_id=request_id,
        stream=False,
        messages_count=len(request.messages),
        user_id=request.user_id,
    )

    # Convert to internal LLMRequest
    llm_request = LLMRequest(
        messages=request.messages,
        stream=False,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        user_id=request.user_id,
        request_id=request_id,
    )

    try:
        response = await client.complete(llm_request)
    except LLMAuthenticationError as exc:
        logger.error("chat.request.auth_failed", request_id=request_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM provider authentication failed: {exc.message}",
        ) from exc
    except LLMRateLimitError as exc:
        logger.warning("chat.request.rate_limited", request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"LLM provider rate limit exceeded. Retry after {exc.retry_after_seconds}s",
            headers={"Retry-After": str(int(exc.retry_after_seconds or 60))},
        ) from exc
    except LLMTimeoutError as exc:
        logger.warning("chat.request.timeout", request_id=request_id, timeout=exc.timeout_seconds)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"LLM provider timed out after {exc.timeout_seconds}s",
        ) from exc
    except LLMError as exc:
        logger.error("chat.request.llm_error", request_id=request_id, error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM error: {exc.message}",
        ) from exc

    latency_ms = (time.monotonic() - start) * 1000

    logger.info(
        "chat.request.complete",
        request_id=request_id,
        latency_ms=round(latency_ms, 2),
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        provider=response.provider,
    )

    return ChatResponse(
        content=response.content,
        model=response.model,
        provider=response.provider,
        finish_reason=response.finish_reason,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
        latency_ms=round(latency_ms, 2),
        ttft_ms=response.latency_ms,
        request_id=request_id,
    )


@router.post(
    "/stream",
    summary="Stream a response via Server-Sent Events",
    description=(
        "Send a conversation and receive a streaming response via SSE. "
        "Events: `meta` (start), `token` (each chunk), `done` (end), `error` (failure). "
        "Use this for interactive chat UIs to minimize perceived latency."
    ),
)
async def chat_stream(
    request: ChatRequest,
    client: LLMClient = Depends(get_llm_client),
) -> EventSourceResponse:
    """Handle a streaming chat request via SSE."""
    # Force stream=True for the internal request
    request.stream = True
    request_id = str(uuid.uuid4())

    logger.info(
        "chat.stream.received",
        request_id=request_id,
        messages_count=len(request.messages),
        user_id=request.user_id,
    )

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        """Generate SSE events from the LLM stream."""
        start = time.monotonic()
        ttft_ms: float | None = None
        chunk_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        finish_reason = "stop"

        # 1. Send meta event first
        meta = MetaEventData(
            request_id=request_id,
            model=client.default_model or "unknown",
            provider=client.provider_name,
            timestamp=time.time(),
        )
        yield {
            "event": "meta",
            "data": meta.model_dump_json(),
        }

        # 2. Stream tokens
        llm_request = LLMRequest(
            messages=request.messages,
            stream=True,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            user_id=request.user_id,
            request_id=request_id,
        )

        try:
            async for chunk in client.stream(llm_request):
                chunk_count += 1
                if ttft_ms is None:
                    ttft_ms = (time.monotonic() - start) * 1000

                # Capture usage from final chunk
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens

                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason

                # Emit token event (skip empty content in final chunks)
                if chunk.content:
                    token_data = TokenEventData(
                        content=chunk.content,
                        timestamp=time.time(),
                    )
                    yield {
                        "event": "token",
                        "data": token_data.model_dump_json(),
                    }

            # 3. Send done event
            total_ms = (time.monotonic() - start) * 1000
            done = DoneEventData(
                finish_reason=finish_reason,  # type: ignore[arg-type]
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=round(total_ms, 2),
                ttft_ms=round(ttft_ms, 2) if ttft_ms else None,
            )
            yield {
                "event": "done",
                "data": done.model_dump_json(),
            }

            logger.info(
                "chat.stream.complete",
                request_id=request_id,
                ttft_ms=round(ttft_ms, 2) if ttft_ms else None,
                total_ms=round(total_ms, 2),
                chunk_count=chunk_count,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        except LLMAuthenticationError as exc:
            logger.error("chat.stream.auth_failed", request_id=request_id, error=str(exc))
            err = ErrorEventData(
                error_type="authentication_error",
                message=f"LLM provider authentication failed: {exc.message}",
                request_id=request_id,
                recoverable=False,
            )
            yield {"event": "error", "data": err.model_dump_json()}

        except LLMRateLimitError as exc:
            logger.warning("chat.stream.rate_limited", request_id=request_id)
            err = ErrorEventData(
                error_type="rate_limit_error",
                message=f"LLM rate limit exceeded. Retry after {exc.retry_after_seconds}s.",
                request_id=request_id,
                recoverable=True,
            )
            yield {"event": "error", "data": err.model_dump_json()}

        except LLMTimeoutError as exc:
            logger.warning("chat.stream.timeout", request_id=request_id)
            err = ErrorEventData(
                error_type="timeout_error",
                message=f"LLM provider timed out after {exc.timeout_seconds}s",
                request_id=request_id,
                recoverable=True,
            )
            yield {"event": "error", "data": err.model_dump_json()}

        except LLMError as exc:
            logger.error("chat.stream.llm_error", request_id=request_id, error=str(exc), exc_info=True)
            err = ErrorEventData(
                error_type="llm_error",
                message=f"LLM error: {exc.message}",
                request_id=request_id,
                recoverable=False,
            )
            yield {"event": "error", "data": err.model_dump_json()}

        except Exception as exc:
            logger.error("chat.stream.unexpected_error", request_id=request_id, error=str(exc), exc_info=True)
            err = ErrorEventData(
                error_type="internal_error",
                message=f"Unexpected error: {type(exc).__name__}",
                request_id=request_id,
                recoverable=False,
            )
            yield {"event": "error", "data": err.model_dump_json()}

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
        ping=15,  # send keepalive ping every 15s
    )
