"""Phase 7: list-price cost estimates."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.config import Settings
from app.models.enums import AgentName, AgentStatus
from app.models.metrics import AgentMetric, TokenUsage
from app.schemas.workflow import build_workflow_response
from app.services.pricing import (
    ModelPricing,
    billed_output_tokens,
    estimate_cost,
    estimate_run_cost,
    resolve_pricing,
    usage_cost,
)
from tests.conftest import StubProvider, run_pipeline

RATES = ModelPricing(model="m", input_per_mtok=1.0, output_per_mtok=4.0, source="test")


def _metric(agent: AgentName, usage: TokenUsage | None, status=AgentStatus.COMPLETED):
    now = datetime.now(UTC)
    return AgentMetric(
        sequence=0,
        agent=agent,
        status=status,
        started_at=now,
        ended_at=now,
        latency_seconds=1.0,
        usage=usage,
    )


def test_billed_output_includes_reasoning_for_either_provider():
    # Groq: completion tokens already include reasoning.
    groq = TokenUsage(prompt_tokens=100, output_tokens=300, thought_tokens=120, total_tokens=400)
    # Gemini: candidate tokens exclude thoughts, the total includes them.
    gemini = TokenUsage(prompt_tokens=100, output_tokens=180, thought_tokens=120, total_tokens=400)
    assert billed_output_tokens(groq) == 300
    assert billed_output_tokens(gemini) == 300


def test_billed_output_falls_back_when_total_is_missing():
    usage = TokenUsage(prompt_tokens=100, output_tokens=50, thought_tokens=10, total_tokens=0)
    assert billed_output_tokens(usage) == 60


def test_usage_cost_prices_input_and_output_separately():
    usage = TokenUsage(prompt_tokens=1_000_000, output_tokens=0, thought_tokens=0,
                       total_tokens=1_500_000)
    assert usage_cost(usage, RATES) == pytest.approx(1.0 + 0.5 * 4.0)


def test_unknown_usage_is_unpriced_not_free():
    usage = TokenUsage(prompt_tokens=10, output_tokens=10, thought_tokens=0, total_tokens=20)
    estimate = estimate_cost([usage, None], RATES)
    assert estimate is not None
    assert estimate.priced_calls == 1
    assert estimate.unpriced_calls == 1
    assert estimate_cost([usage], None) is None


def test_run_cost_skips_non_llm_nodes_and_failed_calls():
    usage = TokenUsage(prompt_tokens=10, output_tokens=10, thought_tokens=0, total_tokens=20)
    metrics = [
        _metric(AgentName.RETRIEVER, None),
        _metric(AgentName.WRITER, usage),
        _metric(AgentName.EDITOR, None, status=AgentStatus.FAILED),
        _metric(AgentName.FINALIZE, None),
    ]
    estimate = estimate_run_cost(metrics, RATES)
    assert estimate is not None
    assert estimate.priced_calls == 1
    assert estimate.unpriced_calls == 0


def test_known_models_are_priced_from_the_table():
    pricing = resolve_pricing(Settings(llm_provider="groq", groq_model="openai/gpt-oss-120b"))
    assert pricing is not None
    assert (pricing.input_per_mtok, pricing.output_per_mtok) == (0.15, 0.60)
    assert pricing.source.startswith("list price")


def test_unknown_models_are_not_guessed_at():
    assert resolve_pricing(Settings(llm_provider="gemini", gemini_model="mystery-model")) is None


def test_settings_override_the_table_only_when_both_rates_are_set():
    both = Settings(
        llm_provider="groq",
        groq_model="openai/gpt-oss-120b",
        llm_input_price_per_mtok=2.0,
        llm_output_price_per_mtok=3.0,
    )
    pricing = resolve_pricing(both)
    assert pricing is not None
    assert (pricing.input_per_mtok, pricing.output_per_mtok) == (2.0, 3.0)
    assert pricing.source == "settings"

    one = Settings(
        llm_provider="gemini", gemini_model="mystery-model", llm_input_price_per_mtok=2.0
    )
    assert resolve_pricing(one) is None


async def test_workflow_response_carries_the_estimate(settings, run_kwargs):
    state = await run_pipeline(StubProvider(), settings, run_kwargs)

    priced = build_workflow_response(state, RATES)
    assert priced.cost is not None
    assert priced.cost.total_usd > 0
    assert priced.cost.unpriced_calls == 0

    assert build_workflow_response(state).cost is None
