"""LLM service: retry bounds, timeout mapping and configuration errors."""

from __future__ import annotations

import asyncio

import pytest

from app.core.config import Settings
from app.core.exceptions import (
    ConfigurationError,
    InvalidAgentOutputError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.models.metrics import TokenUsage
from app.models.review import EditorOutput
from app.services.llm.base import RETRY_AFTER_DETAIL_KEY, RETRYABLE_DETAIL_KEY, LLMResult
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


# --- provider-signalled backoff --------------------------------------------


class DelayAwareProvider:
    """Fails with a rate limit carrying a stated delay, then succeeds."""

    name = "delayed"
    model = "delayed-model"

    def __init__(self, *, failures: int, retry_after: float | None):
        self.failures = failures
        self.retry_after = retry_after
        self.calls = 0

    async def generate_structured(self, **_):
        self.calls += 1
        if self.calls <= self.failures:
            details = {"provider": self.name}
            if self.retry_after is not None:
                details[RETRYABLE_DETAIL_KEY] = True
                details[RETRY_AFTER_DETAIL_KEY] = self.retry_after
            raise LLMRateLimitError("rate limited", details=details)
        return LLMResult(value=VERDICT, usage=None, model=self.model)


async def test_service_waits_the_stated_delay_before_retrying(monkeypatch):
    """The wait is the provider's number, not a backoff curve we invented."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("app.services.llm_service.asyncio.sleep", fake_sleep)

    provider = DelayAwareProvider(failures=1, retry_after=6.5)
    result = await call(service(provider, retries=2))

    assert result.value.quality_score == 8.5
    assert provider.calls == 2
    assert slept == [6.5]


async def test_backoff_stays_within_the_attempt_budget(monkeypatch):
    """Waiting must not buy extra attempts."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("app.services.llm_service.asyncio.sleep", fake_sleep)

    provider = DelayAwareProvider(failures=99, retry_after=2.0)
    with pytest.raises(LLMRateLimitError):
        await call(service(provider, retries=2))

    assert provider.calls == 3, "max_retries + 1, unchanged by the delay"
    assert slept == [2.0, 2.0], "no sleep after the final attempt"


async def test_rate_limit_without_a_delay_is_not_retried(monkeypatch):
    """A daily quota does not clear by waiting - Gemini's case, unchanged."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("app.services.llm_service.asyncio.sleep", fake_sleep)

    provider = DelayAwareProvider(failures=99, retry_after=None)
    with pytest.raises(LLMRateLimitError):
        await call(service(provider, retries=2))

    assert provider.calls == 1
    assert slept == []


async def test_retryable_without_a_delay_retries_immediately(monkeypatch):
    """Schema failures and 5xx keep their existing no-wait behaviour."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("app.services.llm_service.asyncio.sleep", fake_sleep)

    provider = ScriptedProvider(["bad_output", "ok"])
    await call(service(provider, retries=2))

    assert provider.calls == 2
    assert slept == [], "no delay was signalled, so none was taken"
