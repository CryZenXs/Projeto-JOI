"""Tests for app.llm.base (LLMRequest, LLMResponse, TokenChunk, LLMClient).

Verifies:
- Data model validation (required fields, type coercion, ranges)
- LLMMessage role enforcement
- LLMRequest conversation structure rules
- TokenUsage arithmetic
- LLMClient model resolution logic
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.llm.base import (
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMRole,
    TokenChunk,
    TokenUsage,
)


class TestLLMRole:
    """Test the LLMRole enum."""

    def test_role_values(self) -> None:
        assert LLMRole.SYSTEM.value == "system"
        assert LLMRole.USER.value == "user"
        assert LLMRole.ASSISTANT.value == "assistant"
        assert LLMRole.TOOL.value == "tool"

    def test_role_from_string(self) -> None:
        assert LLMRole("user") == LLMRole.USER
        assert LLMRole("system") == LLMRole.SYSTEM


class TestLLMMessage:
    """Test the LLMMessage model."""

    def test_valid_message(self) -> None:
        msg = LLMMessage(role=LLMRole.USER, content="Hello, JOI!")
        assert msg.role == LLMRole.USER
        assert msg.content == "Hello, JOI!"
        assert msg.name is None
        assert msg.metadata == {}

    def test_message_with_string_role(self) -> None:
        """Role should accept string and convert to LLMRole."""
        msg = LLMMessage(role="user", content="Hi")  # type: ignore[arg-type]
        assert msg.role == LLMRole.USER

    def test_message_with_metadata(self) -> None:
        msg = LLMMessage(
            role=LLMRole.ASSISTANT,
            content="I'm JOI",
            metadata={"timestamp": 1234567890, "tool_call_id": "abc"},
        )
        assert msg.metadata["timestamp"] == 1234567890
        assert msg.metadata["tool_call_id"] == "abc"

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot be empty"):
            LLMMessage(role=LLMRole.USER, content="")

    def test_whitespace_only_content_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot be empty"):
            LLMMessage(role=LLMRole.USER, content="   \n\t  ")


class TestLLMRequest:
    """Test the LLMRequest model."""

    def test_minimal_valid_request(self) -> None:
        req = LLMRequest(
            messages=[LLMMessage(role=LLMRole.USER, content="Hi")]
        )
        assert len(req.messages) == 1
        assert req.temperature == 0.7  # default
        assert req.max_tokens == 2048  # default
        assert req.stream is True  # default
        assert req.model is None

    def test_first_message_must_be_system_or_user(self) -> None:
        with pytest.raises(ValidationError, match="First message must be"):
            LLMRequest(
                messages=[
                    LLMMessage(role=LLMRole.ASSISTANT, content="Hi"),
                    LLMMessage(role=LLMRole.USER, content="Hello"),
                ]
            )

    def test_temperature_range(self) -> None:
        """Temperature must be between 0.0 and 2.0."""
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=[LLMMessage(role=LLMRole.USER, content="Hi")],
                temperature=-0.1,
            )
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=[LLMMessage(role=LLMRole.USER, content="Hi")],
                temperature=2.5,
            )

    def test_max_tokens_range(self) -> None:
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=[LLMMessage(role=LLMRole.USER, content="Hi")],
                max_tokens=0,
            )

    def test_empty_messages_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMRequest(messages=[])


class TestTokenUsage:
    """Test the TokenUsage model and its arithmetic."""

    def test_default_values(self) -> None:
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_addition(self) -> None:
        u1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        u2 = TokenUsage(prompt_tokens=5, completion_tokens=15, total_tokens=20)
        result = u1 + u2
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 35
        assert result.total_tokens == 50

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=-1)


class TestTokenChunk:
    """Test the TokenChunk model."""

    def test_intermediate_chunk(self) -> None:
        chunk = TokenChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.finish_reason is None
        assert chunk.usage is None

    def test_final_chunk(self) -> None:
        chunk = TokenChunk(
            content="",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert chunk.finish_reason == "stop"
        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 15


class TestLLMResponse:
    """Test the LLMResponse model."""

    def test_full_response(self) -> None:
        resp = LLMResponse(
            content="Hi there!",
            model="llama-3.1-70b-versatile",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            finish_reason="stop",
            response_id="resp_123",
            latency_ms=234.5,
            provider="groq",
        )
        assert resp.content == "Hi there!"
        assert resp.provider == "groq"
        assert resp.latency_ms == 234.5


class TestLLMClientBase:
    """Test the abstract LLMClient class."""

    def test_cannot_instantiate_abstract_directly(self) -> None:
        """LLMClient itself can be instantiated (it's not a true ABC) but its
        methods raise NotImplementedError."""
        client = LLMClient(default_model="test")
        assert client.default_model == "test"
        assert client.provider_name == "abstract"

    def test_resolve_model_uses_request_model_first(self) -> None:
        client = LLMClient(default_model="default-model")
        req = LLMRequest(
            messages=[LLMMessage(role=LLMRole.USER, content="Hi")],
            model="custom-model",
        )
        assert client._resolve_model(req) == "custom-model"

    def test_resolve_model_falls_back_to_default(self) -> None:
        client = LLMClient(default_model="default-model")
        req = LLMRequest(
            messages=[LLMMessage(role=LLMRole.USER, content="Hi")],
        )
        assert client._resolve_model(req) == "default-model"

    def test_resolve_model_raises_when_no_model(self) -> None:
        client = LLMClient(default_model=None)
        req = LLMRequest(
            messages=[LLMMessage(role=LLMRole.USER, content="Hi")],
        )
        with pytest.raises(ValueError, match="No model specified"):
            client._resolve_model(req)

    def test_repr(self) -> None:
        client = LLMClient(default_model="llama-3.1-70b")
        repr_str = repr(client)
        assert "LLMClient" in repr_str
        assert "llama-3.1-70b" in repr_str
