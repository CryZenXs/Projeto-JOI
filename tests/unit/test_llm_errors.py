"""Tests for app.llm.errors (exception hierarchy + classify_http_error).

Verifies:
- Exception inheritance chain
- Exception attributes (provider, retry_after, etc.)
- classify_http_error correctly maps status codes
"""

from __future__ import annotations

import pytest

from app.llm.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    classify_http_error,
)


class TestExceptionHierarchy:
    """Verify the exception inheritance structure."""

    def test_all_inherit_from_llm_error(self) -> None:
        """All LLM exceptions should inherit from LLMError.

        This allows `except LLMError` to catch everything.
        """
        assert issubclass(LLMAuthenticationError, LLMError)
        assert issubclass(LLMRateLimitError, LLMError)
        assert issubclass(LLMTimeoutError, LLMError)
        assert issubclass(LLMConnectionError, LLMError)
        assert issubclass(LLMProviderError, LLMError)

    def test_base_error_str_includes_provider(self) -> None:
        err = LLMError("something broke", provider="groq")
        assert "[groq]" in str(err)
        assert "something broke" in str(err)

    def test_auth_error_default_message(self) -> None:
        err = LLMAuthenticationError(provider="groq")
        assert "Invalid API key" in str(err)
        assert err.provider == "groq"

    def test_rate_limit_carries_retry_after(self) -> None:
        err = LLMRateLimitError(provider="groq", retry_after_seconds=30.0)
        assert err.retry_after_seconds == 30.0

    def test_timeout_carries_timeout_seconds(self) -> None:
        err = LLMTimeoutError(provider="groq", timeout_seconds=30.0)
        assert err.timeout_seconds == 30.0

    def test_connection_error_preserves_cause(self) -> None:
        original = ConnectionError("network down")
        err = LLMConnectionError(provider="groq", cause=original)
        assert err.__cause__ is original

    def test_provider_error_carries_status_and_body(self) -> None:
        err = LLMProviderError(
            "server error",
            provider="groq",
            status_code=503,
            response_body='{"error": "unavailable"}',
        )
        assert err.status_code == 503
        assert err.response_body == '{"error": "unavailable"}'


class TestClassifyHttpError:
    """Verify the HTTP status code to exception mapper."""

    @pytest.mark.parametrize(
        "status_code,expected_type",
        [
            (401, LLMAuthenticationError),
            (403, LLMAuthenticationError),
            (429, LLMRateLimitError),
            (500, LLMProviderError),
            (502, LLMProviderError),
            (503, LLMProviderError),
            (504, LLMProviderError),
            (400, LLMProviderError),
            (404, LLMProviderError),
        ],
    )
    def test_status_code_mapping(self, status_code: int, expected_type: type) -> None:
        err = classify_http_error(status_code, provider="groq")
        assert isinstance(err, expected_type)
        assert err.provider == "groq"

    def test_429_includes_retry_after(self) -> None:
        err = classify_http_error(429, provider="groq", retry_after=45.0)
        assert isinstance(err, LLMRateLimitError)
        assert err.retry_after_seconds == 45.0

    def test_5xx_includes_response_body(self) -> None:
        err = classify_http_error(
            500,
            provider="groq",
            response_body='{"error": "internal"}',
        )
        assert err.status_code == 500
        assert err.response_body == '{"error": "internal"}'

    def test_unknown_status_returns_provider_error(self) -> None:
        err = classify_http_error(999, provider="groq")
        assert isinstance(err, LLMProviderError)
        assert "999" in str(err)
