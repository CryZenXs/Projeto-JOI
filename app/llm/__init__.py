"""Projeto JOI - LLM abstraction layer.

This module defines the contracts that ALL LLM providers must implement.
The architecture follows the "depend on abstractions, not concretions"
principle from SOLID — the rest of the app (Persona Engine, Memory, API
routes) depends on `LLMClient`, never on `GroqClient` or `OllamaClient`
directly.

This separation is critical for:
1. **Fallback**: When Groq fails, the OllamaClient can transparently
   take over without the caller knowing.
2. **Testing**: Unit tests can inject a MockLLMClient that returns
   deterministic responses.
3. **Future providers**: Adding Anthropic, OpenAI, or a new local model
   is a matter of implementing the interface, not refactoring callers.
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
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)

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
    "LLMRateLimitError",
    "LLMTimeoutError",
]
