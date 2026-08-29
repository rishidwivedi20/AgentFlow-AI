"""LLM service: retry bounds, timeout mapping and configuration errors."""

from __future__ import annotations

import asyncio

import pytest

from app.core.config import Settings
from app.core.exceptions import (
    ConfigurationError,
    InvalidAgentOutputError,
    LLMProviderError,
    LLMTimeoutError,
)
from app.models.metrics import TokenUsage
from app.models.review import EditorOutput
from app.services.llm.base import RETRYABLE_DETAIL_KEY, LLMResult
from app.services.llm_service import LLMService, build_provider

VERDICT = EditorOutput(approved=True, quality_score=8.5, issues=[], feedback=[])


class ScriptedProvider:
    """Replays a scripted list of behaviours, counting calls."""

    name = "scripted"
    model = "scripted-model"

    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls = 0

    async def generate_structured(self, **_):
        self.calls += 1
        behaviour = self.behaviours.pop(0) if self.behaviours else "ok"
        if behaviour == "ok":
            return LLMResult(
                value=VERDICT,
                usage=TokenUsage(
                    prompt_tokens=1, output_tokens=2, thought_tokens=3, total_tokens=6
                ),
                model=self.model,
            )
        if behaviour == "hang":
            await asyncio.sleep(30)
        if behaviour == "bad_output":
            raise InvalidAgentOutputError("unparsable", details={RETRYABLE_DETAIL_KEY: True})
        if behaviour == "server":
            raise LLMProviderError("5xx", details={RETRYABLE_DETAIL_KEY: True})
        raise LLMProviderError("bad api key")


def service(provider, *, retries=2, timeout=0.2):
    return LLMService(provider, timeout_seconds=timeout, max_retries=retries)


async def call(svc):
    return await svc.generate_structured(system_prompt="s", user_prompt="u", schema=EditorOutput)


async def test_returns_validated_value_and_usage():
    provider = ScriptedProvider(["ok"])
    result = await call(service(provider))
    assert result.value.quality_score == 8.5
    assert result.usage.total_tokens == 6
    assert provider.calls == 1


async def test_retries_transient_failures_then_succeeds():
    provider = ScriptedProvider(["bad_output", "server", "ok"])
    assert (await call(service(provider))).value.quality_score == 8.5
    assert provider.calls == 3


async def test_retries_are_bounded():
    provider = ScriptedProvider(["bad_output"] * 10)
    with pytest.raises(InvalidAgentOutputError):
        await call(service(provider, retries=2))
    assert provider.calls == 3, "must not exceed max_retries + 1"


async def test_zero_retries_means_one_attempt():
    provider = ScriptedProvider(["bad_output"] * 5)
    with pytest.raises(InvalidAgentOutputError):
        await call(service(provider, retries=0))
    assert provider.calls == 1


async def test_non_retryable_errors_are_not_retried():
    """Retrying a bad key cannot succeed and burns a free-tier request."""
    provider = ScriptedProvider(["client"] * 5)
    with pytest.raises(LLMProviderError):
        await call(service(provider))
    assert provider.calls == 1


async def test_timeout_maps_to_domain_error_and_is_not_retried():
    provider = ScriptedProvider(["hang"] * 5)
    with pytest.raises(LLMTimeoutError) as exc:
        await call(service(provider, timeout=0.05))
    assert exc.value.status_code == 504
    assert provider.calls == 1


async def test_max_attempts_property():
    assert service(ScriptedProvider([]), retries=2).max_attempts == 3
    assert service(ScriptedProvider([]), retries=0).max_attempts == 1


def test_missing_api_key_raises_configuration_error():
    with pytest.raises(ConfigurationError) as exc:
        build_provider(Settings(gemini_api_key="", llm_provider="gemini", app_env="test"))
    assert exc.value.status_code == 503
    assert "GEMINI_API_KEY" in exc.value.message


def test_configured_key_builds_gemini_provider():
    provider = build_provider(
        Settings(gemini_api_key="k", llm_provider="gemini", gemini_model="m", app_env="test")
    )
    assert provider.name == "gemini"
    assert provider.model == "m"
