"""Projeto JOI - LLM abstraction layer.

This module defines the contracts that ALL LLM providers must implement.
The architecture follows the "depend on abstractions, not concretions"
principle from SOLID — the rest of the app (Persona Engine, Memory, API
routes) depends on `LLMClient` or `LLMRouter`, never on `GroqClient`
or `OllamaClient` directly.
"""

from __future__ import annotations

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
)
from app.llm.mock_client import MockLLMClient
from app.llm.router import LLMRouter, RoutingDecision, RoutingMode

__all__ = [
    # Core interface
    "LLMClient",
    # Data models
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LLMRole",
    "TokenChunk",
    "TokenUsage",
    # Exceptions
    "LLMError",
    "LLMAuthenticationError",
    "LLMConnectionError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    # Mock client (for testing)
    "MockLLMClient",
    # Router
    "LLMRouter",
    "RoutingMode",
    "RoutingDecision",
]
