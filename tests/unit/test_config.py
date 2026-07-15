"""Unit tests for app.core.config.Settings.

Verifies that settings are loaded correctly from environment variables,
sensitive values are protected, and production validation works.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from pydantic import ValidationError


def test_default_settings_load_without_env(reset_settings_cache: Any) -> None:
    """Settings should load with sensible defaults when no env vars are set."""
    # Clear any env vars that might interfere
    for key in list(os.environ.keys()):
        if key.upper().startswith(("GROQ_", "OLLAMA_", "DATABASE_", "REDIS_", "APP_", "ENVIRONMENT")):
            if key.upper() != "ENVIRONMENT":
                os.environ.pop(key, None)

    from app.core.config import get_settings
    settings = get_settings()

    assert settings.app_name == "Projeto JOI"
    assert settings.app_version == "0.1.0"
    assert settings.environment == "development"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.groq_model_primary == "llama-3.1-70b-versatile"
    assert settings.memory_working_ttl_seconds == 1800


def test_environment_variable_override(reset_settings_cache: Any) -> None:
    """Environment variables should override defaults."""
    os.environ["APP_NAME"] = "Test JOI Instance"
    os.environ["PORT"] = "9999"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["GROQ_MODEL_PRIMARY"] = "custom-model"

    from app.core.config import get_settings
    settings = get_settings()

    assert settings.app_name == "Test JOI Instance"
    assert settings.port == 9999
    assert settings.log_level == "DEBUG"
    assert settings.groq_model_primary == "custom-model"

    # Cleanup
    for key in ["APP_NAME", "PORT", "LOG_LEVEL", "GROQ_MODEL_PRIMARY"]:
        os.environ.pop(key, None)


def test_allowed_origins_parsed_from_comma_string(reset_settings_cache: Any) -> None:
    """Comma-separated ALLOWED_ORIGINS_RAW env var should be parsed into a list."""
    os.environ["ALLOWED_ORIGINS_RAW"] = "https://a.com,https://b.com,https://c.com"

    from app.core.config import get_settings
    settings = get_settings()

    assert settings.allowed_origins == ["https://a.com", "https://b.com", "https://c.com"]

    os.environ.pop("ALLOWED_ORIGINS_RAW", None)


def test_secret_key_default_is_dev_only(reset_settings_cache: Any) -> None:
    """Default secret key should contain the dev-only marker."""
    from app.core.config import get_settings
    settings = get_settings()
    assert "dev-only" in settings.secret_key.get_secret_value()


def test_production_requires_groq_key(reset_settings_cache: Any) -> None:
    """Production environment without GROQ_API_KEY should fail validation."""
    os.environ["ENVIRONMENT"] = "production"
    os.environ["GROQ_API_KEY"] = ""
    os.environ["SECRET_KEY"] = "a-real-production-secret-key-not-dev-only"

    # Clear cache to force reload
    from app.core.config import get_settings
    get_settings.cache_clear()

    with pytest.raises(ValidationError) as exc_info:
        get_settings()

    assert "GROQ_API_KEY" in str(exc_info.value) or "required" in str(exc_info.value).lower()

    # Restore
    os.environ.pop("ENVIRONMENT", None)
    os.environ.pop("GROQ_API_KEY", None)
    os.environ.pop("SECRET_KEY", None)


def test_production_rejects_dev_secret_key(reset_settings_cache: Any) -> None:
    """Production environment with the default dev secret key should fail."""
    os.environ["ENVIRONMENT"] = "production"
    os.environ["SECRET_KEY"] = "dev-only-DO-NOT-USE-IN-PROD-change-me-now"
    os.environ["GROQ_API_KEY"] = "gsk_test_key_for_validation"

    from app.core.config import get_settings
    get_settings.cache_clear()

    with pytest.raises(ValidationError) as exc_info:
        get_settings()

    assert "SECRET_KEY" in str(exc_info.value) or "secret" in str(exc_info.value).lower()

    # Restore
    os.environ.pop("ENVIRONMENT", None)
    os.environ.pop("GROQ_API_KEY", None)
    os.environ.pop("SECRET_KEY", None)


def test_computed_properties(reset_settings_cache: Any) -> None:
    """Computed properties should reflect environment correctly."""
    from app.core.config import get_settings

    # Dev
    settings = get_settings()
    assert settings.is_development is True
    assert settings.is_production is False
    assert settings.has_groq_key is False  # no key set in test env

    # With key
    os.environ["GROQ_API_KEY"] = "gsk_test_key"
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.has_groq_key is True

    os.environ.pop("GROQ_API_KEY", None)


def test_settings_cached(reset_settings_cache: Any) -> None:
    """get_settings() should return the same instance (cached)."""
    from app.core.config import get_settings
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
