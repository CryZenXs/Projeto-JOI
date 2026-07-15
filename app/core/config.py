"""Projeto JOI - Core configuration module.

This module centralizes all application configuration using Pydantic Settings,
which provides type-safe, validated configuration loading from environment
variables and .env files.

Design principles:
- Single source of truth for all configuration
- Type validation at startup (fail fast)
- Sensitive values (API keys) never logged
- Sensible defaults for local development
- Production overrides via environment variables
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All fields can be overridden via environment variables (case-insensitive)
    or a .env file at the project root. Sensible defaults are provided for
    local development; production deployments must override via env vars.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars (e.g. from Docker)
    )

    # ─── Application ──────────────────────────────────────────────────────
    app_name: str = "Projeto JOI"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = Field(default=False, description="Enable debug mode (verbose logs, reload)")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    # ─── Security ─────────────────────────────────────────────────────────
    # In production, this MUST be set to a cryptographically random string
    # Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
    secret_key: SecretStr = Field(
        default=SecretStr("dev-only-DO-NOT-USE-IN-PROD-change-me-now"),
        description="Secret key for signing tokens, encrypting cookies, etc.",
    )
    # Pydantic-settings v2 doesn't auto-split CSV strings into list[str].
    # Solution: declare as str (raw env value), then expose a computed property
    # that splits on access. This is the cleanest documented approach.
    allowed_origins_raw: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="CORS allowed origins, comma-separated",
    )

    # ─── Groq API (primary LLM provider) ──────────────────────────────────
    groq_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Groq Cloud API key. Required for primary LLM. Get one at https://console.groq.com",
    )
    groq_model_primary: str = "llama-3.1-70b-versatile"
    groq_model_fallback: str = "llama-3.1-8b-instant"
    groq_max_tokens: int = 2048
    groq_temperature: float = 0.7
    groq_timeout_seconds: float = 30.0
    groq_max_retries: int = 3

    # ─── Ollama (local LLM fallback) ──────────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_seconds: float = 60.0
    ollama_enabled: bool = True

    # ─── PostgreSQL (relational DB) ───────────────────────────────────────
    # Format: postgresql+asyncpg://user:password@host:port/database
    database_url: str = "postgresql+asyncpg://joi:joi@localhost:5432/joi"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout_seconds: float = 30.0
    db_echo: bool = False  # set True to log all SQL (very verbose)

    # ─── Redis (cache + queue) ────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_password: SecretStr = Field(default=SecretStr(""))
    redis_namespace: str = "joi"

    # ─── ChromaDB (vector store) ──────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_episodes: str = "episodic_memory"
    chroma_collection_facts: str = "semantic_facts"
    chroma_embedding_model: str = "bge-m3"

    # ─── Memory system ────────────────────────────────────────────────────
    memory_working_ttl_seconds: int = 1800  # 30 minutes
    memory_max_context_turns: int = 12
    memory_retrieval_top_k: int = 5
    memory_rerank_enabled: bool = True

    # ─── Persona ──────────────────────────────────────────────────────────
    persona_config_dir: str = "./config/persona"
    persona_reinjection_interval: int = 8  # reinject system prompt every N turns
    persona_consistency_check_enabled: bool = True

    # ─── Rate limiting & budget ───────────────────────────────────────────
    rate_limit_per_minute: int = 60
    monthly_budget_usd: float = 100.0
    budget_alert_threshold: float = 0.8  # alert at 80% of monthly budget

    # ─── Feature flags (for progressive rollout) ──────────────────────────
    feature_streaming: bool = True
    feature_memory_persistence: bool = True
    feature_persona_consistency_check: bool = True
    feature_fallback_local: bool = True

    # ─── Validators ───────────────────────────────────────────────────────

    # No need for custom validators at this stage - all fields are simple types
    # and the CSV parsing for allowed_origins is handled via computed property below.

    @field_validator("groq_api_key", mode="after")
    @classmethod
    def validate_groq_key_in_production(cls, v: SecretStr, info) -> SecretStr:
        """In production, Groq API key is mandatory."""
        environment = info.data.get("environment", "development")
        if environment == "production" and not v.get_secret_value():
            raise ValueError(
                "GROQ_API_KEY is required in production environment. "
                "Set it via environment variable or .env file."
            )
        return v

    @field_validator("secret_key", mode="after")
    @classmethod
    def validate_secret_key_in_production(cls, v: SecretStr, info) -> SecretStr:
        """In production, secret key must not be the dev default."""
        environment = info.data.get("environment", "development")
        if environment == "production" and "dev-only" in v.get_secret_value():
            raise ValueError(
                "SECRET_KEY must be set to a cryptographically random value in production. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        return v

    # ─── Computed properties (not in env vars) ────────────────────────────

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key.get_secret_value())

    @property
    def allowed_origins(self) -> list[str]:
        """Parse the raw allowed_origins CSV string into a list.

        Pydantic-settings v2 doesn't natively split CSV env vars into list[str],
        so we store as str and parse on access.
        """
        return [
            o.strip() for o in self.allowed_origins_raw.split(",") if o.strip()
        ]

    @property
    def redis_db_url(self) -> int:
        """Extract Redis database number from URL."""
        # redis://host:port/N -> N
        url = self.redis_url.rstrip("/")
        if url.endswith("/0") or url.endswith("/1") or url.endswith("/2"):
            return int(url[-1])
        return 0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance.

    The result is cached for the lifetime of the process. To force a reload
    (e.g. in tests), call `get_settings.cache_clear()`.

    Returns:
        Settings: The validated application settings.
    """
    return Settings()


# Convenience alias for imports
settings = get_settings()
