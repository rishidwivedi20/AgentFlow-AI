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

from pydantic import Field, field_validator, model_validator
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
    # schema mode, which every agent in this system depends on;
    # llama-3.3-70b-versatile is available but cannot enforce a schema.
    # 120b over 20b: measured against the real pipeline, 20b reliably produced a
    # valid initial draft but failed constrained decoding on the longer revision
    # prompt, while 120b completed every agent with no errors and earned a
    # higher editorial score.
    groq_model: str = "openai/gpt-oss-120b"
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

    # --- Retrieval (Phase 5) ----------------------------------------------
    retrieval_enabled: bool = True
    """When False the pipeline falls back to search snippets alone. Kept as a
    switch so a run can proceed when the embedding quota is spent, and so
    Phase 6 can measure retrieval's contribution by turning it off."""

    # Always a Gemini model: Groq exposes no embeddings API, so retrieval needs
    # GEMINI_API_KEY whatever LLM_PROVIDER is set to.
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = Field(default=768, ge=128, le=3072)
    """Truncated from the model's native 3072. Comparable retrieval quality on
    a corpus this small, at a quarter the memory."""

    retrieval_chunk_size: int = Field(default=1200, ge=200, le=4000)
    retrieval_chunk_overlap: int = Field(default=150, ge=0, le=1000)
    retrieval_top_k: int = Field(default=8, ge=1, le=50)
    """Eight ~300-token chunks is ~2,400 tokens of evidence, which is what fits
    alongside a prompt and its output inside Groq's 8,000 tokens-per-minute
    free-tier ceiling."""

    retrieval_min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    """Below this a chunk is treated as off-topic rather than as evidence.
    Never as evidence that a claim is false."""

    retrieval_max_chunks_per_source: int = Field(default=3, ge=1, le=20)
    """Stops one long page filling the entire evidence budget. Relaxed
    automatically when honouring it would leave the budget unfilled."""

    retrieval_max_chunks_per_run: int = Field(default=90, ge=10, le=2000)
    """Bounds how much corpus is embedded, and with it what a run costs. The
    free tier allows 100 embedded passages per minute, counting items rather
    than HTTP calls - batching saves round trips, not quota. 90 leaves room for
    the query embedding and a second run inside the same minute."""

    # --- Orchestration guardrails -----------------------------------------
    max_revisions: int = Field(default=3, ge=1, le=10)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # --- Cost estimation (Phase 7) ----------------------------------------
    llm_input_price_per_mtok: float | None = Field(default=None, ge=0)
    """USD per million prompt tokens. Overrides the built-in price table in
    ``app.services.pricing``; both this and the output price must be set."""
    llm_output_price_per_mtok: float | None = Field(default=None, ge=0)
    """USD per million output tokens, reasoning included."""

    # --- Run history (Phase 7) --------------------------------------------
    run_store_enabled: bool = True
    """Persist finished runs to SQLite so history and cross-run metrics
    survive a restart and outlive the in-memory registry's retention."""
    run_store_path: Path = BACKEND_DIR / "data" / "agentflow.db"

    @model_validator(mode="after")
    def _check_chunk_geometry(self) -> Settings:
        """Overlap must leave room for new content in every chunk.

        Each chunk carries up to ``retrieval_chunk_overlap`` characters of the
        previous one, so an overlap at or above the chunk size would leave no
        budget for anything new and the chunker could not make progress.
        """
        if self.retrieval_chunk_overlap >= self.retrieval_chunk_size:
            raise ValueError(
                "retrieval_chunk_overlap must be smaller than retrieval_chunk_size "
                f"(got {self.retrieval_chunk_overlap} >= {self.retrieval_chunk_size})"
            )
        return self

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
    def embeddings_configured(self) -> bool:
        """Whether retrieval has a usable embedding key.

        Deliberately reads ``gemini_api_key`` rather than ``active_api_key``:
        Groq has no embeddings API, so retrieval depends on Gemini's key even
        when Gemini is not the generation provider. A Groq run without it can
        still complete - the pipeline falls back to search snippets - but it
        will not retrieve.
        """
        return bool(self.gemini_api_key.strip())

    @property
    def retrieval_available(self) -> bool:
        """Whether retrieval is both switched on and able to run."""
        return self.retrieval_enabled and self.embeddings_configured

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
