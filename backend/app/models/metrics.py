"""Per-agent execution metrics.

Populated exclusively by the metrics wrapper in
``app.services.metrics_service`` so no agent contains timing or token-counting
code of its own.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import AgentName, AgentStatus


class TokenUsage(BaseModel):
    """Token counts as reported by the provider.

    Only ever constructed from real provider usage data. When a provider does
    not report usage, the surrounding :class:`AgentMetric` carries ``None``
    rather than zeros, so "not reported" is never mistaken for "free".
    """

    prompt_tokens: int = Field(ge=0, description="Tokens in the request.")
    output_tokens: int = Field(ge=0, description="Tokens in the visible completion.")
    thought_tokens: int = Field(
        ge=0,
        description=(
            "Internal reasoning tokens. Gemini 2.5+ reports these separately and bills "
            "them, so omitting them would undercount real cost."
        ),
    )
    total_tokens: int = Field(ge=0, description="Provider-reported total for the call.")


class AgentMetric(BaseModel):
    """One agent execution, successful or failed."""

    sequence: int = Field(
        ge=0,
        description=(
            "Zero-based index of this execution within the run. The metrics list "
            "is already ordered, but a live stream delivers events individually - "
            "without this the frontend cannot tell the Writer's first run from its "
            "third when highlighting a node inside the revision loop."
        ),
    )
    agent: AgentName
    status: AgentStatus
    started_at: datetime
    ended_at: datetime
    latency_seconds: float = Field(ge=0.0, description="Wall-clock duration of the execution.")
    usage: TokenUsage | None = Field(
        default=None,
        description=(
            "Provider token usage. None when the agent made no LLM call, or when the "
            "provider reported none."
        ),
    )
    error: str | None = Field(
        default=None,
        description="Human-readable failure reason. Set only when status is 'failed'.",
    )
