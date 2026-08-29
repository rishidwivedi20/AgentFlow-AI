"""Centralised LLM access: retries, timeouts and error translation.

Agents call this service; they never touch a provider adapter, an SDK or an
API key. Two policies live here deliberately:

**Retries are bounded and belong to the service.** The graph never retries. Loops
belong to the graph, retries belong to the service - blurring the two is how a
run silently multiplies its token spend.

**Only transient failures are retried.** Adapters mark a failure retryable when
another attempt could plausibly succeed (a 5xx, or output that failed schema
validation). A bad API key or an exhausted quota is surfaced immediately: a
retry cannot succeed and would spend another free-tier request to learn that.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Final

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ConfigurationError,
    InvalidAgentOutputError,
    LLMProviderError,
    LLMTimeoutError,
    OrchestrAIError,
)
from app.services.llm.base import RETRYABLE_DETAIL_KEY, LLMProvider, LLMResult, SchemaT
from app.services.llm.gemini import GeminiProvider

logger = logging.getLogger(__name__)

# Deterministic default. Agents that benefit from variety (the Writer) raise it
# explicitly; agents that must be reproducible (the Editor) keep it low.
DEFAULT_TEMPERATURE: Final = 0.4


def _is_retryable(exc: OrchestrAIError) -> bool:
    return bool(exc.details.get(RETRYABLE_DETAIL_KEY, False))


class LLMService:
    """Provider-agnostic entry point for structured generation."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        self._provider = provider
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def model(self) -> str:
        return self._provider.model

    @property
    def max_attempts(self) -> int:
        """Total attempts per call: the first plus configured retries."""
        return self._max_retries + 1

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> LLMResult[SchemaT]:
        """Generate a validated ``schema`` instance, retrying transient faults.

        Raises the last error once attempts are exhausted. Never returns a
        partially valid object and never loops unbounded.
        """
        last_error: OrchestrAIError | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return await self._attempt(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    temperature=temperature,
                )
            except (InvalidAgentOutputError, LLMProviderError) as exc:
                last_error = exc
                if not _is_retryable(exc) or attempt == self.max_attempts:
                    raise
                logger.warning(
                    "Retrying LLM call after transient failure",
                    extra={
                        "attempt": attempt,
                        "max_attempts": self.max_attempts,
                        "schema": schema.__name__,
                        "error_type": exc.error_type,
                    },
                )

        # Unreachable: the loop either returns or raises. Guards against a
        # future edit silently falling through with no result.
        raise last_error or InvalidAgentOutputError("LLM call produced no result.")

    async def _attempt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        temperature: float,
    ) -> LLMResult[SchemaT]:
        """One provider call, bounded by the configured timeout.

        The timeout is applied here rather than through provider-specific HTTP
        options so that every provider behaves identically and the unit is
        unambiguously seconds.
        """
        try:
            return await asyncio.wait_for(
                self._provider.generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    temperature=temperature,
                ),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            # Not retried: another attempt would double an already-long wait.
            raise LLMTimeoutError(
                f"{self._provider.name} did not respond within {self._timeout_seconds:g}s.",
                details={"provider": self._provider.name, "timeout_seconds": self._timeout_seconds},
            ) from exc


def build_provider(settings: Settings) -> LLMProvider:
    """Instantiate the configured provider adapter.

    Raises ``ConfigurationError`` rather than constructing a client that would
    fail on first use with a less obvious message.
    """
    if not settings.llm_configured:
        raise ConfigurationError(
            f"No API key configured for provider '{settings.llm_provider}'. "
            f"Set {settings.llm_provider.upper()}_API_KEY in backend/.env.",
            details={"provider": settings.llm_provider},
        )

    if settings.llm_provider == "gemini":
        return GeminiProvider(api_key=settings.active_api_key, model=settings.active_model)

    # Unreachable while ProviderName is a Literal, but explicit so adding a
    # member to that Literal without an adapter fails loudly here.
    raise ConfigurationError(
        f"No adapter implemented for provider '{settings.llm_provider}'.",
        details={"provider": settings.llm_provider},
    )


def build_llm_service(settings: Settings | None = None) -> LLMService:
    """Build the service from configuration.

    Not cached: tests construct :class:`LLMService` directly with a stub
    provider, and the graph receives its service by injection rather than
    reaching for a global.
    """
    resolved = settings or get_settings()
    return LLMService(
        build_provider(resolved),
        timeout_seconds=resolved.llm_timeout_seconds,
        max_retries=resolved.llm_max_retries,
    )
