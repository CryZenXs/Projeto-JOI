"""Projeto JOI - Chat API request/response schemas.

These schemas are the public contract of our API. External clients
(web frontend, mobile app, CLI) depend on these shapes — breaking
changes here require versioning.

Design choices:
- Pydantic v2 for runtime validation + OpenAPI generation
- Strict types (no `Any` in public API)
- Descriptive field names (no abbreviations)
- Examples in Field descriptions for /docs usability
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.llm.base import LLMMessage, LLMRole


class ChatRequest(BaseModel):
    """Request body for POST /api/v1/chat and /api/v1/chat/stream.

    The caller provides a list of messages (conversation history) and
    optional generation parameters. The server applies persona and memory
    layers on top before calling the LLM.
    """

    messages: list[LLMMessage] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Conversation history. Last message should be from the user.",
        examples=[
            [
                {"role": "user", "content": "Olá, quem é você?"},
            ]
        ],
    )
    stream: bool = Field(
        default=True,
        description="If true, response is streamed as SSE. If false, returns complete JSON.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower = more focused, higher = more creative.",
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description="Maximum tokens to generate in the response.",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional end-user identifier for tracing and personalization.",
        max_length=128,
    )

    @field_validator("messages")
    @classmethod
    def last_message_must_be_from_user(cls, v: list[LLMMessage]) -> list[LLMMessage]:
        """Ensure the conversation ends with a user message (so the LLM has something to respond to)."""
        if v and v[-1].role != LLMRole.USER:
            raise ValueError(
                f"The last message must be from the user (got '{v[-1].role.value}'). "
                f"The LLM needs a user message to respond to."
            )
        return v


class ChatResponse(BaseModel):
    """Response body for POST /api/v1/chat (non-streaming).

    When stream=false, the complete response is returned as JSON.
    When stream=true, the response is a stream of SSE events (see ChatStreamEvent).
    """

    content: str = Field(..., description="The complete generated text")
    model: str = Field(..., description="Model that produced this response")
    provider: str = Field(..., description="LLM provider (groq/ollama/mock)")
    finish_reason: Literal["stop", "length", "tool_call", "content_filter"] = Field(
        default="stop",
        description="Why generation stopped",
    )
    prompt_tokens: int = Field(ge=0, description="Tokens consumed from the input")
    completion_tokens: int = Field(ge=0, description="Tokens generated in the response")
    total_tokens: int = Field(ge=0, description="Sum of prompt + completion tokens")
    latency_ms: float = Field(ge=0, description="Total request latency in milliseconds")
    ttft_ms: float | None = Field(
        default=None,
        description="Time To First Token (only meaningful for streaming requests)",
    )
    request_id: str = Field(..., description="Correlation ID for tracing")


# ─── SSE event schemas (for streaming) ────────────────────────────────────


class ChatStreamEvent(BaseModel):
    """A single SSE event in the streaming response.

    Event types:
    - "meta":     Sent first, contains request_id and model info
    - "token":    Sent for each generated token chunk
    - "done":     Sent last, contains usage stats and finish_reason
    - "error":    Sent if an error occurs mid-stream
    """

    event: Literal["meta", "token", "done", "error"]
    data: dict = Field(default_factory=dict)


class MetaEventData(BaseModel):
    """Data payload for the 'meta' event (sent at stream start)."""

    request_id: str
    model: str
    provider: str
    timestamp: float


class TokenEventData(BaseModel):
    """Data payload for the 'token' event (sent for each chunk)."""

    content: str
    timestamp: float


class DoneEventData(BaseModel):
    """Data payload for the 'done' event (sent at stream end)."""

    finish_reason: Literal["stop", "length", "tool_call", "content_filter"]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    ttft_ms: float | None = None


class ErrorEventData(BaseModel):
    """Data payload for the 'error' event (sent on failure)."""

    error_type: str
    message: str
    request_id: str
    recoverable: bool = Field(
        default=False,
        description="If true, client may retry the request",
    )
