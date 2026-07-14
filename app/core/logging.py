"""Projeto JOI - Structured logging configuration.

Uses structlog for JSON-formatted structured logs in production and
pretty-printed colored logs in development. All logs are routed through
this module to ensure consistent format and sensitive data redaction.

Usage:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("message", user_id="123", action="login")
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from app.core.config import settings


# Keys whose values should be masked in logs (partial redaction)
_SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "secret", "secret_key", "password", "passwd",
    "token", "access_token", "refresh_token", "authorization",
    "groq_api_key", "ollama_password", "redis_password",
})

# Keys whose values should be completely removed from logs
_FORBIDDEN_KEYS = frozenset({
    "ssn", "cpf", "cnpj", "credit_card", "card_number", "cvv",
})


def _redact_sensitive_values(
    _logger: Any, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Processor that redacts sensitive values from log events.

    - Sensitive keys have their values partially masked (show first 4 chars + ***)
    - Forbidden keys have their values completely removed
    """
    redacted: EventDict = {}
    for key, value in event_dict.items():
        key_lower = key.lower()
        if key_lower in _FORBIDDEN_KEYS:
            redacted[key] = "[REDACTED]"
        elif key_lower in _SENSITIVE_KEYS:
            if isinstance(value, str) and len(value) > 4:
                redacted[key] = f"{value[:4]}***"
            else:
                redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


def _configure_structlog() -> None:
    """Configure structlog with appropriate processors for the environment."""
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redact_sensitive_values,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.is_development:
        # Pretty colored output for dev
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True, exception_formatter=structlog.dev.plain_traceback),
        ]
    else:
        # JSON for production (parseable by ELK, Datadog, etc.)
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        context_class=dict,
        # Use stdlib LoggerFactory so add_logger_name processor works
        # (PrintLogger doesn't have a .name attribute)
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib logging to structlog so uvicorn/sqlalchemy logs are unified
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(message)s",
        stream=sys.stderr,
        force=True,
    )
    for noisy_logger in ("uvicorn.access", "httpx", "chromadb"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A bound structlog logger.
    """
    return structlog.get_logger(name)


# Configure on module import (idempotent - structlog handles re-configuration)
_configure_structlog()
