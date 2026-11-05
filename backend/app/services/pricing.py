"""Token pricing and cost estimation.

Every figure produced here is an *estimate at list price*. On a free tier the
real bill is zero; the estimate answers a different question - what this run
would cost on a paid plan - which is what makes a single-agent and a
multi-agent run comparable in money rather than only in tokens.

Prices change, and nothing here fetches them. The built-in table is a
convenience with a date on it; ``LLM_INPUT_PRICE_PER_MTOK`` and
``LLM_OUTPUT_PRICE_PER_MTOK`` override it, and a model in neither place is
reported as unpriced rather than guessed at.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.models.metrics import AgentMetric, TokenUsage

PRICES_AS_OF: Final = "2026-11"
"""When the built-in table was last checked against published rates."""

_TOKENS_PER_UNIT: Final = 1_000_000

# USD per million tokens: (input, output). Output covers reasoning tokens,
# which both providers bill at the output rate.
_KNOWN_PRICES: Final[dict[str, tuple[float, float]]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    # The standard paid-tier rate. An introductory discount may apply for a
    # while after launch; override through settings to price against it.
    "gemini-3.6-flash": (1.50, 7.50),
}


class ModelPricing(BaseModel):
    """The rates a cost estimate was computed with."""

    model: str
    input_per_mtok: float = Field(ge=0.0, description="USD per million prompt tokens.")
    output_per_mtok: float = Field(
        ge=0.0, description="USD per million output tokens, reasoning included."
    )
    source: str = Field(description="'settings' when overridden, else the table's date.")


class CostEstimate(BaseModel):
    """Estimated list-price cost of one run."""

    pricing: ModelPricing
    total_usd: float = Field(ge=0.0)
    priced_calls: int = Field(ge=0, description="Executions whose token usage was priced.")
    unpriced_calls: int = Field(
        ge=0,
        description=(
            "LLM executions with no reported usage. Their cost is unknown, so the "
            "total is a lower bound whenever this is non-zero."
        ),
    )


def resolve_pricing(settings: Settings, model: str | None = None) -> ModelPricing | None:
    """Rates for ``model`` (default: the active one), or None when unknown.

    Settings win over the table so a user can price against their own
    contract, or against a model the table has never heard of.
    """
    resolved_model = model or settings.active_model
    input_price = settings.llm_input_price_per_mtok
    output_price = settings.llm_output_price_per_mtok
    if input_price is not None and output_price is not None:
        return ModelPricing(
            model=resolved_model,
            input_per_mtok=input_price,
            output_per_mtok=output_price,
            source="settings",
        )
    rates = _KNOWN_PRICES.get(resolved_model)
    if rates is None:
        return None
    return ModelPricing(
        model=resolved_model,
        input_per_mtok=rates[0],
        output_per_mtok=rates[1],
        source=f"list price as of {PRICES_AS_OF}",
    )


def billed_output_tokens(usage: TokenUsage) -> int:
    """Output tokens as billed, reasoning included.

    Derived from the provider's total rather than by adding fields, because
    the providers disagree on what ``output_tokens`` covers: Groq's completion
    count already includes reasoning, Gemini's candidate count excludes it.
    Total minus prompt is right for both.
    """
    from_total = usage.total_tokens - usage.prompt_tokens
    if from_total > 0:
        return from_total
    return usage.output_tokens + usage.thought_tokens


def usage_cost(usage: TokenUsage, pricing: ModelPricing) -> float:
    """USD cost of one call's usage."""
    return (
        usage.prompt_tokens * pricing.input_per_mtok
        + billed_output_tokens(usage) * pricing.output_per_mtok
    ) / _TOKENS_PER_UNIT


def estimate_cost(
    usages: Iterable[TokenUsage | None], pricing: ModelPricing | None
) -> CostEstimate | None:
    """Sum the cost of a set of calls. None when the model is unpriced.

    A ``None`` usage is counted as unpriced, not as free.
    """
    if pricing is None:
        return None
    total = 0.0
    priced = 0
    unpriced = 0
    for usage in usages:
        if usage is None:
            unpriced += 1
            continue
        total += usage_cost(usage, pricing)
        priced += 1
    return CostEstimate(
        pricing=pricing,
        total_usd=round(total, 6),
        priced_calls=priced,
        unpriced_calls=unpriced,
    )


# Nodes that never call an LLM. Their missing usage is expected, not a gap
# in the provider's reporting, so it must not mark the estimate incomplete.
_NON_LLM_AGENTS: Final = frozenset({"retriever", "finalize"})


def estimate_run_cost(
    metrics: Iterable[AgentMetric], pricing: ModelPricing | None
) -> CostEstimate | None:
    """Cost of a workflow run from its per-agent metrics.

    Failed executions are skipped: a call that raised reports no usage, and
    whether the provider billed it is unknowable from here.
    """
    usages = [
        metric.usage
        for metric in metrics
        if metric.agent.value not in _NON_LLM_AGENTS and metric.status.value == "completed"
    ]
    return estimate_cost(usages, pricing)
