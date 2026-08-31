"""Application configuration.

Single source of truth for every environment-driven value. Nothing else in the
codebase should read ``os.environ`` directly - that rule is what keeps
configuration auditable and makes Phase 6 evaluation sweeps possible without
editing code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/  (app/core/config.py -> app/core -> app -> backend)
BACKEND_DIR = Path(__file__).resolve().parents[2]

Environment = Literal["development", "production", "test"]
ProviderName = Literal["gemini", "groq"]


class Settings(BaseSettings):
    """Environment-backed settings, validated once at startup."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application -------------------------------------------------------
    app_name: str = "AgentFlow-AI"
    app_version: str = "0.1.0"
    app_env: Environment = "development"
    log_level: str = "INFO"

    # Comma-separated. Kept as a string rather than list[str] because
    # pydantic-settings parses complex types as JSON, which breaks the
    # conventional comma-separated .env format.
    cors_origins: str = "http://localhost:5173"

    # --- LLM provider (Phase 2+) ------------------------------------------
    llm_provider: ProviderName = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    groq_api_key: str = ""
    # gpt-oss models are the production Groq models supporting strict JSON
    # schema mode, which every agent in this system depends on.
    # llama-3.3-70b-versatile is available but cannot enforce a schema.
    groq_model: str = "openai/gpt-oss-20b"
    groq_max_completion_tokens: int = Field(default=4096, ge=256, le=32768)
    """Bounded because Groq counts prompt + this cap against the free tier's
    8,000 tokens-per-minute ceiling, and rejects the request outright with a
    413 when the sum exceeds it. Left unset, generation is instead truncated
    mid-JSON and fails schema validation."""

    # --- Research provider (Phase 3) --------------------------------------
    # Tavily: 1,000 free credits/month, no card. https://app.tavily.com
    tavily_api_key: str = ""
    tavily_max_results: int = Field(default=5, ge=1, le=20)
    tavily_search_depth: Literal["basic", "advanced"] = "basic"

    # --- Orchestration guardrails -----------------------------------------
    max_revisions: int = Field(default=3, ge=1, le=10)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return level

    # --- Derived values ----------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a clean list, empty entries dropped."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def active_api_key(self) -> str:
        """API key belonging to the currently selected provider."""
        keys: dict[str, str] = {
            "gemini": self.gemini_api_key,
            "groq": self.groq_api_key,
        }
        return keys[self.llm_provider].strip()

    @property
    def active_model(self) -> str:
        """Model name belonging to the currently selected provider."""
        models: dict[str, str] = {
            "gemini": self.gemini_model,
            "groq": self.groq_model,
        }
        return models[self.llm_provider]

    @property
    def llm_configured(self) -> bool:
        """Whether the selected provider has a usable API key.

        Phase 1 legitimately reports ``False``: no keys are needed until the
        agents land in Phase 2.
        """
        return bool(self.active_api_key)

    @property
    def research_configured(self) -> bool:
        """Whether a real search backend is available.

        When False the Researcher cannot run: without search results it could
        only produce sources from model output, which would be fabricated.
        """
        return bool(self.tavily_api_key.strip())

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that ``.env`` is read once and every caller observes identical
    configuration. Use ``get_settings.cache_clear()`` in tests.
    """
    return Settings()
