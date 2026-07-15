"""Projeto JOI - LLM-specific exception hierarchy.

These exceptions follow a deliberate hierarchy that allows the router
and circuit breaker to make intelligent decisions:

    LLMError (base)
    ├── LLMAuthenticationError  → token invalid, don't retry
    ├── LLMRateLimitError       → wait + retry, or fallback
    ├── LLMTimeoutError         → retry with backoff
    ├── LLMConnectionError      → retry once, then fallback
    └── LLMProviderError        → log and re-raise (provider bug)

The key insight: the router needs to distinguish "this will never work"
(auth errors) from "this might work if we wait" (rate limit, timeout).
"""

from __future__ import annotations


class LLMError(Exception):
    """Base exception for all LLM-related errors.

    All other LLM exceptions inherit from this, so `except LLMError`
    catches everything LLM-related.
    """

    def __init__(self, message: str, *, provider: str = "unknown") -> None:
        super().__init__(message)
        self.provider = provider
        self.message = message

    def __str__(self) -> str:
        return f"[{self.provider}] {self.message}"


class LLMAuthenticationError(LLMError):
    """API key is invalid, expired, or lacks required scopes.

    Retrying won't help — the user needs to fix the key.
    The router should NOT retry; should NOT fallback (same key would fail).
    """

    def __init__(self, message: str = "Invalid API key", *, provider: str = "unknown") -> None:
        super().__init__(message, provider=provider)


class LLMRateLimitError(LLMError):
    """Provider returned 429 Too Many Requests.

    Retrying with exponential backoff might work, or fallback to another
    provider. The router should respect `retry_after_seconds` if set.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        provider: str = "unknown",
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.retry_after_seconds = retry_after_seconds


class LLMTimeoutError(LLMError):
    """Request took longer than the configured timeout.

    Often transient. Retry with same or longer timeout.
    """

    def __init__(
        self,
        message: str = "Request timed out",
        *,
        provider: str = "unknown",
        timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.timeout_seconds = timeout_seconds


class LLMConnectionError(LLMError):
    """Could not connect to the provider (network down, DNS, etc.).

    Often transient. Retry once, then fallback if still failing.
    """

    def __init__(
        self,
        message: str = "Could not connect to provider",
        *,
        provider: str = "unknown",
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.__cause__ = cause


class LLMProviderError(LLMError):
    """Provider returned an unexpected error (5xx, malformed response, etc.).

    Usually indicates a bug in our integration or the provider's API.
    Log with full context and re-raise (don't retry blindly).
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "unknown",
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.status_code = status_code
        self.response_body = response_body


# ─── Helper: classify HTTP exceptions ─────────────────────────────────────


def classify_http_error(
    status_code: int,
    *,
    provider: str,
    response_body: str | None = None,
    retry_after: float | None = None,
) -> LLMError:
    """Map an HTTP status code to the appropriate LLM exception.

    Centralizing this logic avoids duplicating it across providers.
    """
    if status_code == 401:
        return LLMAuthenticationError(
            "Invalid or expired API key",
            provider=provider,
        )
    if status_code == 403:
        return LLMAuthenticationError(
            "API key lacks required permissions",
            provider=provider,
        )
    if status_code == 429:
        return LLMRateLimitError(
            "Rate limit exceeded",
            provider=provider,
            retry_after_seconds=retry_after,
        )
    if status_code >= 500:
        return LLMProviderError(
            f"Provider server error (HTTP {status_code})",
            provider=provider,
            status_code=status_code,
            response_body=response_body,
        )
    if status_code >= 400:
        return LLMProviderError(
            f"Provider client error (HTTP {status_code})",
            provider=provider,
            status_code=status_code,
            response_body=response_body,
        )
    return LLMProviderError(
        f"Unexpected HTTP status {status_code}",
        provider=provider,
        status_code=status_code,
        response_body=response_body,
    )
