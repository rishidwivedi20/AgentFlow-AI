"""Runs both systems over a topic set and scores them.

Topics run one at a time. Running them concurrently would finish sooner but
trip the free tiers' per-minute limits, and the retry waits that followed would
land in the latency being measured.

The baseline and the evaluators wait out a rate limit and try again, which the
production service deliberately does not do: a user waiting on a run is better
told to come back, while an evaluation left running is better finishing late
than finishing with holes in it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Final, TypeVar

from app.core.config import Settings
from app.core.exceptions import AgentFlowError, LLMRateLimitError
from app.evaluation.baseline import run_single_agent
from app.evaluation.judge import judge_grounding, judge_quality
from app.evaluation.models import EvaluationReport, SystemName, SystemOutcome, TopicResult
from app.graph.state import WorkflowState
from app.graph.workflow import run_workflow
from app.models.content import Post, WriterOutput
from app.models.enums import AgentName, Platform, Tone
from app.models.evidence import EvidenceSet
from app.models.metrics import TokenUsage
from app.models.research import Source
from app.services.llm.base import RETRYABLE_DETAIL_KEY, LLMResult
from app.services.llm_service import LLMService
from app.services.pricing import ModelPricing, estimate_cost, estimate_run_cost
from app.services.research_service import ResearchService
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

# Shared by both systems, so excluded from both systems' generation cost.
_SHARED_AGENTS: Final = frozenset({AgentName.RETRIEVER, AgentName.FINALIZE})

ProgressCallback = Callable[[int, int, str], None]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """How long to wait after a rate limit, and how many times to try again."""

    wait_seconds: float = 60.0
    """A free tier's per-minute window clears within a minute."""
    retries: int = 2


def _worth_waiting_for(exc: AgentFlowError) -> bool:
    """A rate limit, or an outage the provider marked transient (a 503 under load).

    The service has already retried the latter immediately; by the time it
    reaches here, only a wait is left to try.
    """
    return isinstance(exc, LLMRateLimitError) or bool(exc.details.get(RETRYABLE_DETAIL_KEY))


async def _patiently(call: Callable[[], Awaitable[T]], policy: RateLimitPolicy) -> T:
    """Run ``call``, waiting and retrying on a rate limit or a transient outage.

    Every other failure is raised at once: no amount of waiting fixes a bad
    key or a malformed response.
    """
    for attempt in range(policy.retries + 1):
        try:
            return await call()
        except AgentFlowError as exc:
            if not _worth_waiting_for(exc) or attempt == policy.retries:
                raise
            logger.warning(
                "Rate limited; waiting before retrying",
                extra={"wait_seconds": policy.wait_seconds, "attempt": attempt + 1},
            )
            await asyncio.sleep(policy.wait_seconds)
    raise AssertionError("unreachable")  # the loop returns or raises


def _multi_agent_outcome(state: WorkflowState, pricing: ModelPricing | None) -> SystemOutcome:
    """Project a finished graph run onto the comparison's terms."""
    generating = [m for m in state["agent_metrics"] if m.agent not in _SHARED_AGENTS]
    cost = estimate_run_cost(generating, pricing)
    posts = state["final_output"]
    return SystemOutcome(
        system=SystemName.MULTI_AGENT,
        posts=posts,
        llm_calls=len(generating),
        total_tokens=sum(m.usage.total_tokens for m in generating if m.usage),
        latency_seconds=round(sum(m.latency_seconds for m in generating), 4),
        cost_usd=cost.total_usd if cost else None,
        revisions=state["revision_count"],
        workflow_status=state["workflow_status"].value,
        error=None if posts else _no_posts_reason(state),
    )


def _no_posts_reason(state: WorkflowState) -> str:
    """Why the graph ended empty-handed, from the errors it recorded."""
    causes = "; ".join(f"{error.agent.value}: {error.message}" for error in state["errors"])
    return f"The workflow produced no posts ({causes})." if causes else (
        "The workflow produced no posts."
    )


def _failed(system: SystemName, message: str) -> SystemOutcome:
    return SystemOutcome(
        system=system, llm_calls=0, total_tokens=0, latency_seconds=0.0, error=message
    )


def _describe(exc: BaseException) -> str:
    """A domain error's own message, or the exception type for anything else."""
    if isinstance(exc, AgentFlowError):
        return exc.message
    return f"unexpected {type(exc).__name__}"


def _tokens(usage: TokenUsage | None) -> int:
    return usage.total_tokens if usage else 0


async def evaluate_topic(
    llm: LLMService,
    research: ResearchService,
    *,
    topic: str,
    platform: Platform,
    tone: Tone,
    number_of_posts: int,
    settings: Settings,
    retrieval: RetrievalService | None,
    pricing: ModelPricing | None,
    rate_limits: RateLimitPolicy | None = None,
) -> TopicResult:
    """Run both systems on one topic, then score each output the same way."""
    policy = rate_limits or RateLimitPolicy()
    # --- multi-agent: the production graph, untouched ----------------------
    try:
        state = await run_workflow(
            llm,
            research,
            topic=topic,
            platform=platform,
            tone=tone,
            number_of_posts=number_of_posts,
            settings=settings,
            retrieval=retrieval,
        )
    except AgentFlowError as exc:
        message = f"The workflow failed to execute: {exc.message}"
        return TopicResult(
            topic=topic,
            source_count=0,
            multi_agent=_failed(SystemName.MULTI_AGENT, message),
            single_agent=_failed(
                SystemName.SINGLE_AGENT,
                "Skipped: the shared search runs inside the workflow, which failed.",
            ),
            evaluator_tokens=0,
        )

    multi = _multi_agent_outcome(state, pricing)
    sources = state["sources"]
    evidence = state["evidence"]

    # --- single agent: same sources, same evidence, one call ---------------
    if not sources:
        single = _failed(
            SystemName.SINGLE_AGENT, "Skipped: the shared search returned no sources."
        )
    else:
        async def baseline() -> tuple[LLMResult[WriterOutput], float]:
            # Timed per attempt, so a rate-limit wait is not billed to the
            # baseline as generation latency.
            started = perf_counter()
            result = await run_single_agent(
                llm,
                topic=topic,
                platform=platform,
                tone=tone,
                number_of_posts=number_of_posts,
                sources=sources,
                evidence=evidence,
            )
            return result, perf_counter() - started

        try:
            result, elapsed = await _patiently(baseline, policy)
        except Exception as exc:  # noqa: BLE001 - one system failing must not end the evaluation
            # A domain error is already explained; only a surprise needs its traceback.
            logger.warning(
                "Baseline failed",
                extra={"topic": topic[:80]},
                exc_info=None if isinstance(exc, AgentFlowError) else exc,
            )
            single = _failed(SystemName.SINGLE_AGENT, _describe(exc))
        else:
            cost = estimate_cost([result.usage], pricing)
            single = SystemOutcome(
                system=SystemName.SINGLE_AGENT,
                posts=result.value.posts,
                llm_calls=1,
                total_tokens=_tokens(result.usage),
                latency_seconds=round(elapsed, 4),
                cost_usd=cost.total_usd if cost else None,
            )

    # --- scoring: identical treatment, blind to the system -----------------
    evaluator_tokens = 0
    evaluator_errors: list[str] = []
    for outcome in (multi, single):
        if not outcome.succeeded:
            continue
        evaluator_tokens += await _score(
            llm,
            outcome,
            topic=topic,
            platform=platform,
            tone=tone,
            number_of_posts=number_of_posts,
            sources=sources,
            evidence=evidence,
            errors=evaluator_errors,
            policy=policy,
        )

    return TopicResult(
        topic=topic,
        source_count=len(sources),
        evidence_status=evidence.status.value if evidence else None,
        multi_agent=multi,
        single_agent=single,
        evaluator_tokens=evaluator_tokens,
        evaluator_errors=evaluator_errors,
    )


async def _score(
    llm: LLMService,
    outcome: SystemOutcome,
    *,
    topic: str,
    platform: Platform,
    tone: Tone,
    number_of_posts: int,
    sources: list[Source],
    evidence: EvidenceSet | None,
    errors: list[str],
    policy: RateLimitPolicy,
) -> int:
    """Attach quality and grounding to ``outcome``. Returns evaluator tokens.

    A failed evaluator leaves its score as None and records why. Substituting
    a zero would count an evaluator's outage against the system being scored.
    """
    posts: list[Post] = outcome.posts
    spent = 0
    try:
        quality = await _patiently(
            lambda: judge_quality(
                llm,
                topic=topic,
                platform=platform,
                tone=tone,
                number_of_posts=number_of_posts,
                posts=posts,
            ),
            policy,
        )
    except Exception as exc:  # noqa: BLE001 - see docstring
        errors.append(f"{outcome.system.value} quality: {_describe(exc)}")
    else:
        outcome.quality = quality.value
        spent += _tokens(quality.usage)

    try:
        grounding, raw = await _patiently(
            lambda: judge_grounding(llm, sources=sources, evidence=evidence, posts=posts),
            policy,
        )
    except Exception as exc:  # noqa: BLE001 - see docstring
        errors.append(f"{outcome.system.value} grounding: {_describe(exc)}")
    else:
        outcome.grounding = grounding
        spent += _tokens(raw.usage)
    return spent


async def evaluate(
    llm: LLMService,
    research: ResearchService,
    *,
    topics: Sequence[str],
    platform: Platform,
    tone: Tone,
    number_of_posts: int,
    settings: Settings,
    retrieval: RetrievalService | None,
    pricing: ModelPricing | None,
    pause_seconds: float = 0.0,
    rate_limits: RateLimitPolicy | None = None,
    on_progress: ProgressCallback | None = None,
) -> EvaluationReport:
    """Evaluate every topic in order and assemble the report.

    ``pause_seconds`` waits between topics, letting a per-minute quota refill
    before the next topic starts rather than mid-way through it.
    """
    started_at = datetime.now(UTC)
    results: list[TopicResult] = []

    for index, topic in enumerate(topics):
        if index and pause_seconds:
            await asyncio.sleep(pause_seconds)
        if on_progress:
            on_progress(index + 1, len(topics), topic)
        logger.info("Evaluating topic", extra={"index": index + 1, "topic": topic[:80]})
        results.append(
            await evaluate_topic(
                llm,
                research,
                topic=topic,
                platform=platform,
                tone=tone,
                number_of_posts=number_of_posts,
                settings=settings,
                retrieval=retrieval,
                pricing=pricing,
                rate_limits=rate_limits,
            )
        )

    return EvaluationReport(
        started_at=started_at,
        finished_at=datetime.now(UTC),
        provider=llm.provider_name,
        model=llm.model,
        pricing=pricing,
        number_of_posts=number_of_posts,
        max_revisions=settings.max_revisions,
        retrieval_enabled=retrieval is not None,
        results=results,
    )
