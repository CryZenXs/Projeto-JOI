"""Unit tests for app.core.logging.

Verifies that structured logging is configured correctly and that
sensitive values are properly redacted.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import pytest
import structlog

from app.core.logging import _redact_sensitive_values, get_logger


class TestSensitiveDataRedaction:
    """Verify sensitive values are never logged in plaintext."""

    def test_api_key_is_redacted(self) -> None:
        """API keys should be masked (first 4 chars + ***)."""
        event_dict: dict[str, Any] = {
            "event": "test",
            "api_key": "gsk_abcdefghijklmnopqrstuvwxyz1234567890",
        }
        result = _redact_sensitive_values(None, "info", event_dict)
        assert result["api_key"] == "gsk_***"
        assert "gsk_abcdefghijklmnopqrstuvwxyz" not in str(result)

    def test_password_is_redacted(self) -> None:
        """Passwords should be masked."""
        event_dict: dict[str, Any] = {
            "event": "test",
            "password": "super_secret_password_123",
        }
        result = _redact_sensitive_values(None, "info", event_dict)
        assert result["password"] == "supe***"
        assert "super_secret" not in str(result)

    def test_short_secret_fully_masked(self) -> None:
        """Short secrets (≤4 chars) should be fully masked."""
        event_dict: dict[str, Any] = {
            "event": "test",
            "token": "abc",
        }
        result = _redact_sensitive_values(None, "info", event_dict)
        assert result["token"] == "***"

    def test_forbidden_keys_completely_redacted(self) -> None:
        """Forbidden keys (CPF, SSN, credit card) should be fully redacted."""
        event_dict: dict[str, Any] = {
            "event": "test",
            "cpf": "123.456.789-00",
            "credit_card": "4532 1234 5678 9012",
            "cvv": "123",
        }
        result = _redact_sensitive_values(None, "info", event_dict)
        assert result["cpf"] == "[REDACTED]"
        assert result["credit_card"] == "[REDACTED]"
        assert result["cvv"] == "[REDACTED]"

    def test_non_sensitive_values_pass_through(self) -> None:
        """Non-sensitive values should be logged unchanged."""
        event_dict: dict[str, Any] = {
            "event": "user.login",
            "user_id": "user_123",
            "action": "login",
            "ip": "192.168.1.1",
        }
        result = _redact_sensitive_values(None, "info", event_dict)
        assert result == event_dict

    def test_case_insensitive_key_matching(self) -> None:
        """Redaction should be case-insensitive for key names."""
        event_dict: dict[str, Any] = {
            "event": "test",
            "API_KEY": "gsk_test123",
            "Password": "secret",
            "TOKEN": "tok_abc",
        }
        result = _redact_sensitive_values(None, "info", event_dict)
        assert "***" in result["API_KEY"]
        assert "***" in result["Password"]
        assert "***" in result["TOKEN"]


class TestLoggerInstantiation:
    """Verify logger creation and basic functionality."""

    def test_get_logger_returns_bound_logger(self) -> None:
        """get_logger should return a structlog BoundLogger."""
        logger = get_logger("test.module")
        assert logger is not None
        # Should have standard log methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_logger_does_not_raise(self) -> None:
        """Logging with various args should not raise."""
        logger = get_logger("test")
        logger.info("test_event", key="value", number=42, flag=True)
        logger.warning("warning_event", detail="something")
        # No assertion needed - just verifying no exception
