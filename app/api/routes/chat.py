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

Both endpoints use the LLMRouter (Part 1.3), which automatically handles
Groq → Ollama fallback with circuit breakers and retry logic.
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
from app.core.retry import MaxRetriesExceeded
from app.llm.base import LLMClient, LLMRequest
from app.llm.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.llm.router import LLMRouter, RoutingMode

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


# ─── Router singleton (lazy initialization) ───────────────────────────────
# We create the LLMRouter once and reuse it across requests. This is
# important because circuit breakers maintain state between calls —
# if we created a new router per request, the breaker would never trip.

_router_instance: LLMRouter | None = None


def get_llm_router() -> LLMRouter | None:
    """Get or create the singleton LLMRouter instance.

    Returns None if no providers can be initialized (e.g., no Groq key
    and Ollama not running). In that case, the chat endpoints will
    return an error indicating LLM is not available.
    """
    global _router_instance
    if _router_instance is not None:
        return _router_instance

    # Build primary client
    primary: LLMClient
    if settings.has_groq_key:
        from app.llm.groq_client import GroqClient
        try:
            primary = GroqClient()
            logger.info("chat.router.primary_configured", provider="groq")
        except LLMError as exc:
            logger.error("chat.router.primary_init_failed", error=str(exc))
            primary = _get_mock_client()
    else:
        logger.warning("chat.router.no_groq_key", message="Using MockLLMClient as primary")
        primary = _get_mock_client()

    # Build fallback client (Ollama, if enabled)
    fallback: LLMClient | None = None
    if settings.feature_fallback_local and settings.ollama_enabled:
        from app.llm.ollama_client import OllamaClient
        try:
            fallback = OllamaClient()
            logger.info("chat.router.fallback_configured", provider="ollama")
        except Exception as exc:
            logger.warning("chat.router.fallback_init_failed", error=str(exc))

    # If primary is mock and no fallback, we still create the router
    # (mock works for development/testing)
    _router_instance = LLMRouter(
        primary=primary,
        fallback=fallback,
        max_retries=2,
        retry_base_delay_seconds=1.0,
    )

    logger.info(
        "chat.router.initialized",
        primary=primary.provider_name,
        fallback=fallback.provider_name if fallback else None,
    )

    return _router_instance


def _get_mock_client() -> LLMClient:
    """Create a MockLLMClient (used when no real provider is available)."""
    from app.llm.mock_client import MockLLMClient
    return MockLLMClient()


def reset_router() -> None:
    """Reset the router singleton (for testing)."""
    global _router_instance
    _router_instance = None


# ─── Dependency for backward compatibility ────────────────────────────────


def get_llm_client() -> LLMClient:
    """Legacy dependency: return a client (not router).

    Kept for backward compatibility with tests. New code should use
    get_llm_router() instead.
    """
    r = get_llm_router()
    if r is None:
        return _get_mock_client()
    # Return the primary client directly (bypasses router logic)
    return r._primary  # type: ignore[union-attr]


# ─── Endpoints ────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a complete response (non-streaming)",
    description=(
        "Send a conversation and receive a complete response. "
        "Automatically handles Groq → Ollama fallback if the primary provider fails. "
        "Records conversation turns to memory and retrieves relevant context. "
        "Use this for batch processing or when streaming isn't needed. "
        "For interactive chat UIs, prefer POST /chat/stream for lower perceived latency."
    ),
)
async def chat(
    request: ChatRequest,
    llm_router: LLMRouter | None = Depends(get_llm_router),
) -> ChatResponse:
    """Handle a non-streaming chat request via the LLMRouter."""
    request_id = str(uuid.uuid4())
    start = time.monotonic()

    if llm_router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM router is not configured. Set GROQ_API_KEY or start Ollama.",
        )

    logger.info(
        "chat.request.received",
        request_id=request_id,
        stream=False,
        messages_count=len(request.messages),
        user_id=request.user_id,
    )

    # Determine routing mode (privacy mode could be triggered by user flag in future)
    mode = RoutingMode.NORMAL

    # Get memory manager (may be None if memory is unavailable)
    memory_manager = None
    user_id = request.user_id or "anonymous"
    try:
        from app.memory.manager import get_memory_manager
        memory_manager = get_memory_manager()
    except Exception as exc:
        logger.warning("chat.memory_unavailable", error=str(exc))

    # Record the user's message to working memory
    if memory_manager and request.messages:
        last_msg = request.messages[-1]
        if last_msg.role.value == "user":
            try:
                await memory_manager.add_turn(
                    user_id=user_id,
                    role="user",
                    content=last_msg.content,
                )
            except Exception as exc:
                logger.warning("chat.memory.record_failed", error=str(exc))

    # Retrieve any relevant facts to inject as context
    memory_context: list[str] = []
    if memory_manager:
        try:
            facts = await memory_manager.get_facts(user_id)
            if facts:
                # Build a brief context summary from known facts
                facts_summary = "; ".join(f"{f.key}: {f.value}" for f in facts[:5])
                memory_context.append(f"[Known facts about user: {facts_summary}]")
        except Exception as exc:
            logger.warning("chat.memory.facts_failed", error=str(exc))

    # Convert to internal LLMRequest, optionally augmenting with memory context
    messages = list(request.messages)
    if memory_context:
        # Insert memory context as a system message at the start
        from app.llm.base import LLMMessage, LLMRole
        messages.insert(0, LLMMessage(
            role=LLMRole.SYSTEM,
            content=" | ".join(memory_context),
        ))

    llm_request = LLMRequest(
        messages=messages,
        stream=False,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        user_id=request.user_id,
        request_id=request_id,
    )

    try:
        response = await llm_router.complete(llm_request, mode=mode)
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
    except MaxRetriesExceeded as exc:
        # Router exhausted retries — check the underlying error for proper status
        underlying = exc.last_exception
        logger.warning(
            "chat.request.max_retries",
            request_id=request_id,
            attempts=exc.attempts,
            underlying_error=type(underlying).__name__,
        )
        if isinstance(underlying, LLMRateLimitError):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"LLM rate limit exceeded after {exc.attempts} retries. Retry after {underlying.retry_after_seconds}s",
                headers={"Retry-After": str(int(underlying.retry_after_seconds or 60))},
            ) from exc
        if isinstance(underlying, LLMTimeoutError):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"LLM timed out after {exc.attempts} retries",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM unavailable after {exc.attempts} retries: {underlying}",
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

    # Record the assistant's response to working memory
    if memory_manager:
        try:
            await memory_manager.add_turn(
                user_id=user_id,
                role="assistant",
                content=response.content,
            )
        except Exception as exc:
            logger.warning("chat.memory.record_response_failed", error=str(exc))

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
        "Automatically handles Groq → Ollama fallback if the primary provider fails "
        "before streaming starts. "
        "Events: `meta` (start), `token` (each chunk), `done` (end), `error` (failure). "
        "Use this for interactive chat UIs to minimize perceived latency."
    ),
)
async def chat_stream(
    request: ChatRequest,
    llm_router: LLMRouter | None = Depends(get_llm_router),
) -> EventSourceResponse:
    """Handle a streaming chat request via SSE through the LLMRouter."""
    # Force stream=True for the internal request
    request.stream = True
    request_id = str(uuid.uuid4())

    if llm_router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM router is not configured. Set GROQ_API_KEY or start Ollama.",
        )

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
            model=llm_router._primary.default_model or "unknown",  # type: ignore[union-attr]
            provider=llm_router._primary.provider_name,  # type: ignore[union-attr]
            timestamp=time.time(),
        )
        yield {
            "event": "meta",
            "data": meta.model_dump_json(),
        }

        # 2. Stream tokens via the router (handles fallback)
        llm_request = LLMRequest(
            messages=request.messages,
            stream=True,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            user_id=request.user_id,
            request_id=request_id,
        )

        try:
            async for chunk in llm_router.stream(llm_request):  # type: ignore[union-attr]
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
