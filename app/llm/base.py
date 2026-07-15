"""Projeto JOI - LLM data models and base interface.

Defines the contracts that every LLM provider implementation must follow.
These models are deliberately framework-agnostic — they don't depend on
LangChain, Groq SDK, or any specific provider's API.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ─── Enums ────────────────────────────────────────────────────────────────


class LLMRole(str, Enum):
    """Standard chat message roles (OpenAI-compatible)."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ─── Request models ───────────────────────────────────────────────────────


class LLMMessage(BaseModel):
    """A single message in a conversation.

    Uses OpenAI's chat format because it's the de facto standard that
    Groq, Ollama, Anthropic, and most providers support natively.
    """

    role: LLMRole
    content: str
    name: str | None = Field(
        default=None,
        description="Optional speaker name (for multi-user contexts)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific metadata (timestamps, tool_call_id, etc.)",
    )

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, v: str) -> str:
        """Empty content is invalid for any role except system placeholder."""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty or whitespace-only")
        return v


class LLMRequest(BaseModel):
    """A request to an LLM provider.

    Encapsulates everything needed to make a single completion call,
    whether streaming or not. Designed to be provider-agnostic so the
    caller doesn't need to know which provider will handle it.
    """

    messages: list[LLMMessage] = Field(
        ...,
        min_length=1,
        description="Conversation history (must include at least one message)",
    )
    model: str | None = Field(
        default=None,
        description="Model ID. If None, provider uses its default.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0 = deterministic, 2 = very random",
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=32768,
        description="Maximum tokens to generate in the response",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling: only consider tokens with top_p cumulative probability",
    )
    stream: bool = Field(
        default=True,
        description="If True, return an async iterator of chunks; if False, return complete response",
    )
    stop: list[str] | None = Field(
        default=None,
        description="Sequences that will stop generation when encountered",
    )
    user_id: str | None = Field(
        default=None,
        description="End-user identifier (for abuse monitoring on provider side)",
    )
    request_id: str | None = Field(
        default=None,
        description="Correlation ID for tracing. Provider should include in logs.",
    )

    @field_validator("messages")
    @classmethod
    def ensure_system_or_user_first(cls, v: list[LLMMessage]) -> list[LLMMessage]:
        """Best practice: conversation should start with system or user message."""
        if v:
            first_role = v[0].role
            if first_role not in (LLMRole.SYSTEM, LLMRole.USER):
                raise ValueError(
                    f"First message must be 'system' or 'user', got '{first_role.value}'"
                )
        return v


# ─── Response models ──────────────────────────────────────────────────────


class TokenUsage(BaseModel):
    """Token usage statistics for billing and budget tracking."""

    prompt_tokens: int = Field(default=0, ge=0, description="Tokens in the input (prompt)")
    completion_tokens: int = Field(default=0, ge=0, description="Tokens generated in the response")
    total_tokens: int = Field(default=0, ge=0, description="Sum of prompt + completion tokens")

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Sum two usage records (useful when accumulating streamed chunks)."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class LLMResponse(BaseModel):
    """A complete (non-streaming) LLM response."""

    content: str = Field(..., description="The generated text")
    model: str = Field(..., description="The model that actually produced this response")
    usage: TokenUsage
    finish_reason: Literal["stop", "length", "tool_call", "content_filter"] = Field(
        default="stop",
        description="Why generation stopped",
    )
    response_id: str | None = Field(
        default=None,
        description="Provider's response ID (for debugging/tracing)",
    )
    created_at: float = Field(
        default_factory=time.time,
        description="Unix timestamp when response was received",
    )
    latency_ms: float | None = Field(
        default=None,
        description="Time to first token (streaming) or full response (non-streaming)",
    )
    provider: str = Field(
        default="unknown",
        description="Name of the provider that handled this request (groq/ollama/mock)",
    )


class TokenChunk(BaseModel):
    """A single chunk in a streaming response.

    Represents one token (or a small group of tokens) as they arrive.
    The final chunk will have `finish_reason` set to a non-None value.
    """

    content: str = Field(
        default="",
        description="The token text (may be empty for the final chunk)",
    )
    finish_reason: Literal["stop", "length", "tool_call", "content_filter"] | None = Field(
        default=None,
        description="Set on the final chunk; None for intermediate chunks",
    )
    usage: TokenUsage | None = Field(
        default=None,
        description="Only present on the final chunk (some providers)",
    )
    delta_role: LLMRole | None = Field(
        default=None,
        description="Role change (rarely used; mainly for tool calls)",
    )
    received_at: float = Field(
        default_factory=time.time,
        description="Unix timestamp when this chunk was received",
    )


# ─── Base client interface ────────────────────────────────────────────────


class LLMClient:
    """Abstract interface that all LLM providers must implement.

    This is an abstract base class (ABC) rather than a Protocol because:
    - We want runtime isinstance() checks (useful for router logic)
    - We want to enforce the same method signatures across all implementations
    - We may add shared helper methods in the future

    Concrete implementations:
    - app.llm.groq_client.GroqClient (primary, cloud-based)
    - app.llm.ollama_client.OllamaClient (fallback, local)  [Part 1.3]
    - app.llm.mock_client.MockLLMClient (for testing)
    """

    provider_name: str = "abstract"

    def __init__(self, default_model: str | None = None) -> None:
        """Initialize the client.

        Args:
            default_model: Model to use when request.model is None.
        """
        self.default_model = default_model

    # ─── Public API ───────────────────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a complete (non-streaming) response.

        Args:
            request: The completion request (request.stream is ignored).

        Returns:
            LLMResponse with the full generated text and usage stats.

        Raises:
            LLMError: On any provider error (network, auth, rate limit, etc.)
        """
        raise NotImplementedError

    async def stream(self, request: LLMRequest) -> AsyncIterator[TokenChunk]:
        """Generate a streaming response.

        Args:
            request: The completion request. request.stream should be True.

        Yields:
            TokenChunk objects as they arrive from the provider.

        Raises:
            LLMError: On any provider error before/during streaming.
        """
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Check if the provider is reachable and credentials are valid.

        Returns:
            True if healthy, False otherwise. Should not raise.
        """
        raise NotImplementedError

    # ─── Shared helpers (used by all implementations) ─────────────────────

    def _resolve_model(self, request: LLMRequest) -> str:
        """Determine which model to use for this request.

        Priority: request.model > self.default_model > raise
        """
        model = request.model or self.default_model
        if not model:
            raise ValueError(
                f"No model specified: request.model is None and "
                f"{self.provider_name} client has no default_model"
            )
        return model

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.provider_name!r} model={self.default_model!r}>"
