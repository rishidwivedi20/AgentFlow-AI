"""Evaluation data: judge output, per-system outcomes and aggregates."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from statistics import fmean

from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.models.content import Post
from app.models.factcheck import FactCheckReport
from app.services.pricing import ModelPricing


class SystemName(StrEnum):
    """The two systems under comparison."""

    MULTI_AGENT = "multi_agent"
    SINGLE_AGENT = "single_agent"


class JudgeOutput(BaseModel):
    """The quality judge's scores for one set of posts.

    LLM-facing: every field required, none nullable, per the structured-output
    rules on :class:`app.models.review.EditorOutput`. Dimensions are scored
    separately before the overall score so the overall cannot be produced
    first and then rationalised.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    clarity: float = Field(ge=0.0, le=10.0, description="Is each post easy to follow?")
    structure: float = Field(
        ge=0.0, le=10.0, description="Hook, body and close; scannable on the platform."
    )
    engagement: float = Field(ge=0.0, le=10.0, description="Would a reader stop and read?")
    relevance: float = Field(ge=0.0, le=10.0, description="Does it stay on the topic?")
    tone_consistency: float = Field(
        ge=0.0, le=10.0, description="Does it hold the requested tone throughout?"
    )
    distinctness: float = Field(
        ge=0.0,
        le=10.0,
        description="Do the posts take genuinely different angles? 10 for a single post.",
    )
    overall: float = Field(ge=0.0, le=10.0, description="Overall publishability.")
    rationale: str = Field(
        min_length=1, description="Two or three sentences justifying the overall score."
    )


class Grounding(BaseModel):
    """Claim-level verification of one output against the shared evidence."""

    report: FactCheckReport

    @computed_field
    @property
    def claims(self) -> int:
        return len(self.report.claims)

    @computed_field
    @property
    def verified(self) -> int:
        return len(self.report.verified)

    @computed_field
    @property
    def uncertain(self) -> int:
        return len(self.report.uncertain)

    @computed_field
    @property
    def unsupported(self) -> int:
        return len(self.report.unsupported)

    @computed_field
    @property
    def supported_rate(self) -> float | None:
        """Share of claims that are verified. None when there were no claims.

        Not 1.0 for a claim-free output: a post that asserts nothing has not
        earned a perfect grounding score, it has simply not been tested.
        """
        if not self.report.claims:
            return None
        return round(len(self.report.verified) / len(self.report.claims), 4)


class SystemOutcome(BaseModel):
    """What one system produced for one topic, and how it scored."""

    system: SystemName
    posts: list[Post] = Field(default_factory=list)
    llm_calls: int = Field(ge=0, description="Provider calls made to produce the posts.")
    total_tokens: int = Field(ge=0)
    latency_seconds: float = Field(
        ge=0.0, description="Generation time, excluding the shared search and retrieval."
    )
    cost_usd: float | None = Field(
        default=None, description="List-price estimate. None when the model is unpriced."
    )
    revisions: int = Field(
        default=0, ge=0, description="Writer revisions; always 0 for the baseline."
    )
    workflow_status: str | None = Field(
        default=None, description="The multi-agent run's terminal status. None for the baseline."
    )
    quality: JudgeOutput | None = None
    grounding: Grounding | None = None
    error: str | None = Field(
        default=None, description="Why this system produced nothing, when it did not."
    )

    @property
    def succeeded(self) -> bool:
        return bool(self.posts) and self.error is None


class TopicResult(BaseModel):
    """Both systems' outcomes for one topic."""

    topic: str
    source_count: int = Field(ge=0)
    evidence_status: str | None = None
    multi_agent: SystemOutcome
    single_agent: SystemOutcome
    evaluator_tokens: int = Field(
        ge=0, description="Tokens the judge and grounding check spent. Charged to neither system."
    )
    evaluator_errors: list[str] = Field(
        default_factory=list,
        description="Scores that could not be produced. A missing score is never a zero.",
    )

    @computed_field
    @property
    def quality_winner(self) -> SystemName | None:
        """Higher overall score, or None on a tie or a missing score."""
        multi = self.multi_agent.quality
        single = self.single_agent.quality
        if multi is None or single is None or multi.overall == single.overall:
            return None
        return SystemName.MULTI_AGENT if multi.overall > single.overall else SystemName.SINGLE_AGENT


def _mean(values: list[float]) -> float | None:
    return round(fmean(values), 4) if values else None


class SystemSummary(BaseModel):
    """One system's results averaged across every topic it completed."""

    system: SystemName
    topics_completed: int
    topics_failed: int
    mean_quality: float | None
    mean_dimensions: dict[str, float]
    mean_supported_rate: float | None
    total_claims: int
    total_unsupported: int
    mean_unsupported_per_topic: float | None
    mean_tokens: float | None
    mean_latency_seconds: float | None
    mean_cost_usd: float | None
    mean_llm_calls: float | None
    quality_wins: int


_DIMENSIONS = (
    "clarity",
    "structure",
    "engagement",
    "relevance",
    "tone_consistency",
    "distinctness",
)


def summarise(system: SystemName, results: list[TopicResult]) -> SystemSummary:
    """Average one system's outcomes over the topics it completed."""
    outcomes = [
        result.multi_agent if system is SystemName.MULTI_AGENT else result.single_agent
        for result in results
    ]
    done = [outcome for outcome in outcomes if outcome.succeeded]
    scored = [outcome.quality for outcome in done if outcome.quality is not None]
    grounded = [outcome.grounding for outcome in done if outcome.grounding is not None]
    rates = [g.supported_rate for g in grounded if g.supported_rate is not None]
    costs = [outcome.cost_usd for outcome in done if outcome.cost_usd is not None]

    return SystemSummary(
        system=system,
        topics_completed=len(done),
        topics_failed=len(outcomes) - len(done),
        mean_quality=_mean([score.overall for score in scored]),
        mean_dimensions={
            name: value
            for name in _DIMENSIONS
            if (value := _mean([getattr(score, name) for score in scored])) is not None
        },
        mean_supported_rate=_mean(rates),
        total_claims=sum(g.claims for g in grounded),
        total_unsupported=sum(g.unsupported for g in grounded),
        mean_unsupported_per_topic=_mean([float(g.unsupported) for g in grounded]),
        mean_tokens=_mean([float(outcome.total_tokens) for outcome in done]),
        mean_latency_seconds=_mean([outcome.latency_seconds for outcome in done]),
        mean_cost_usd=_mean(costs),
        mean_llm_calls=_mean([float(outcome.llm_calls) for outcome in done]),
        quality_wins=sum(1 for result in results if result.quality_winner is system),
    )


class EvaluationReport(BaseModel):
    """A complete evaluation run: configuration, per-topic results, summaries."""

    started_at: datetime
    finished_at: datetime
    provider: str
    model: str
    pricing: ModelPricing | None
    number_of_posts: int
    max_revisions: int
    retrieval_enabled: bool
    results: list[TopicResult]

    @computed_field
    @property
    def summaries(self) -> list[SystemSummary]:
        return [summarise(system, self.results) for system in SystemName]

    @computed_field
    @property
    def ties(self) -> int:
        return sum(
            1
            for result in self.results
            if result.quality_winner is None
            and result.multi_agent.quality is not None
            and result.single_agent.quality is not None
        )
