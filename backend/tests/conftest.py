"""Shared fixtures and stubs.

``StubProvider`` implements the ``LLMProvider`` protocol by dispatching on the
requested schema and replaying scripted results. ``StubSearchProvider`` does
the same for ``ResearchProvider``. Together they make the whole pipeline
deterministic: a test states which drafts get approved and which claims are
unsupported, and the graph does the rest.

No test performs a network call, and none requires an API key.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest

from app.core.config import Settings
from app.models.content import Post, WriterOutput
from app.models.enums import Platform, Tone
from app.models.factcheck import ClaimVerdict, FactCheckClaim, FactCheckReport
from app.models.metrics import TokenUsage
from app.models.plan import ContentAngle, ContentPlan, ContentType
from app.models.research import (
    ResearchFact,
    ResearchOutput,
    ResearchResult,
    Source,
    StatementKind,
)
from app.models.review import EditorIssue, EditorOutput, IssueSeverity, ReviewDimension
from app.services.llm.base import LLMResult
from app.services.llm_service import LLMService
from app.services.research_service import ResearchService

STUB_MODEL = "stub-model"


# --- builders --------------------------------------------------------------


def make_sources(count: int = 3) -> list[Source]:
    return [
        Source(
            title=f"Source {i}",
            url=f"https://example.com/article-{i}",
            snippet=f"Snippet from source {i}.",
        )
        for i in range(count)
    ]


def make_research_output(fact_count: int = 2) -> ResearchOutput:
    return ResearchOutput(
        summary="Sources broadly agree the topic is significant.",
        facts=[
            ResearchFact(
                statement=f"Finding number {i}.",
                kind=StatementKind.FACT if i % 2 == 0 else StatementKind.OPINION,
                source_index=i,
            )
            for i in range(fact_count)
        ],
        limitations=["No figures on adoption rates in smaller teams."],
    )


def make_research_result(*, sources: int = 3, facts: int = 2) -> ResearchResult:
    output = make_research_output(facts)
    return ResearchResult(
        summary=output.summary,
        facts=output.facts,
        limitations=output.limitations,
        sources=make_sources(sources),
    )


def make_plan(count: int = 2) -> ContentPlan:
    types = list(ContentType)
    return ContentPlan(
        angles=[
            ContentAngle(
                post_number=i + 1,
                angle=f"Angle {i}",
                content_type=types[i % len(types)],
                hook_strategy=f"Open with a question about {i}.",
                key_points=[f"Point {i}a", f"Point {i}b"],
                target_audience="Mid-level software engineers",
                intended_takeaway=f"Takeaway {i}",
            )
            for i in range(count)
        ]
    )


def make_posts(count: int = 2, *, tag: str = "d") -> list[Post]:
    return [
        Post(angle=f"angle-{tag}-{i}", content=f"content-{tag}-{i}", hashtags=["ai"])
        for i in range(count)
    ]


def make_verdict(*, approved: bool, score: float) -> EditorOutput:
    issues = (
        []
        if approved
        else [
            EditorIssue(
                dimension=ReviewDimension.CLARITY,
                severity=IssueSeverity.MEDIUM,
                description="The opening buries the point.",
            )
        ]
    )
    return EditorOutput(
        approved=approved,
        quality_score=score,
        issues=issues,
        feedback=[] if approved else ["Lead with the conclusion."],
    )


def make_fact_check(*, unsupported: int = 0, verified: int = 1) -> FactCheckReport:
    claims = [
        FactCheckClaim(
            claim=f"Verified claim {i}",
            verdict=ClaimVerdict.VERIFIED,
            reasoning="Directly supported by a research fact.",
            post_index=1,
            supporting_source_index=0,
        )
        for i in range(verified)
    ]
    claims += [
        FactCheckClaim(
            claim=f"Invented statistic {i}",
            verdict=ClaimVerdict.UNSUPPORTED,
            reasoning="No research fact establishes this number.",
            post_index=1,
            supporting_source_index=-1,
        )
        for i in range(unsupported)
    ]
    return FactCheckReport(claims=claims)


def make_usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=100, output_tokens=200, thought_tokens=50, total_tokens=350)


# --- stubs -----------------------------------------------------------------


class StubSearchProvider:
    """Returns fixed sources, or raises to simulate a search failure."""

    name = "stub-search"

    def __init__(self, *, sources: list[Source] | None = None, fail: bool = False) -> None:
        self._sources = make_sources() if sources is None else sources
        self._fail = fail
        self.calls = 0

    async def search(self, query: str, *, max_results: int) -> list[Source]:
        self.calls += 1
        if self._fail:
            raise RuntimeError("stub search failure")
        return self._sources[:max_results]


class StubProvider:
    """Replays scripted results, dispatching on the requested schema.

    ``verdicts`` drives Editor responses in order; once exhausted the last
    verdict repeats, so a test need only script what it cares about.
    ``fact_checks`` behaves the same way for the Fact Checker.
    """

    name = "stub"
    model = STUB_MODEL

    def __init__(
        self,
        *,
        verdicts: Sequence[EditorOutput] | None = None,
        fact_checks: Sequence[FactCheckReport] | None = None,
        post_count: int = 2,
        angle_count: int = 2,
        fail_on: Sequence[str] = (),
    ) -> None:
        self._verdicts = list(verdicts or [make_verdict(approved=True, score=9.0)])
        self._fact_checks = list(fact_checks or [make_fact_check()])
        self._post_count = post_count
        self._angle_count = angle_count
        self._fail_on = set(fail_on)
        self.researcher_calls = 0
        self.planner_calls = 0
        self.writer_calls = 0
        self.editor_calls = 0
        self.fact_check_calls = 0

    @staticmethod
    def _pick(items: list, index: int):
        return items[min(index, len(items) - 1)]

    def _result(self, value) -> LLMResult:
        return LLMResult(value=value, usage=make_usage(), model=STUB_MODEL)

    async def generate_structured(self, *, schema, **_: object) -> LLMResult:
        if schema is ResearchOutput:
            self.researcher_calls += 1
            self._maybe_fail("researcher")
            return self._result(make_research_output())

        if schema is ContentPlan:
            self.planner_calls += 1
            self._maybe_fail("planner")
            return self._result(make_plan(self._angle_count))

        if schema is WriterOutput:
            self.writer_calls += 1
            self._maybe_fail("writer")
            return self._result(
                WriterOutput(posts=make_posts(self._post_count, tag=f"r{self.writer_calls}"))
            )

        if schema is FactCheckReport:
            self.fact_check_calls += 1
            self._maybe_fail("fact_checker")
            return self._result(self._pick(self._fact_checks, self.fact_check_calls - 1))

        self.editor_calls += 1
        self._maybe_fail("editor")
        return self._result(self._pick(self._verdicts, self.editor_calls - 1))

    def _maybe_fail(self, agent: str) -> None:
        if agent in self._fail_on:
            raise RuntimeError(f"stub {agent} failure")


# --- service builders ------------------------------------------------------


def make_service(provider: StubProvider, *, timeout: float = 5.0, retries: int = 0) -> LLMService:
    return LLMService(provider, timeout_seconds=timeout, max_retries=retries)


def make_research(
    provider: StubSearchProvider | None = None, *, max_results: int = 5
) -> ResearchService:
    return ResearchService(provider or StubSearchProvider(), max_results=max_results)


async def run_pipeline(provider, settings, run_kwargs, *, search=None):
    """Execute a full workflow run with stubbed services."""
    from app.graph.workflow import run_workflow

    return await run_workflow(
        make_service(provider), make_research(search), settings=settings, **run_kwargs
    )


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    """Settings for a test run: keys present so config checks pass."""
    return Settings(
        gemini_api_key="test-key",
        tavily_api_key="test-search-key",
        llm_provider="gemini",
        max_revisions=3,
        app_env="test",
    )


@pytest.fixture
def run_kwargs() -> dict[str, object]:
    return {
        "topic": "How AI is changing software engineering",
        "platform": Platform.LINKEDIN,
        "tone": Tone.PROFESSIONAL,
        "number_of_posts": 2,
    }


@pytest.fixture
def approving_provider() -> StubProvider:
    return StubProvider(verdicts=[make_verdict(approved=True, score=9.0)])


@pytest.fixture
def rejecting_provider() -> StubProvider:
    """Never approves, forcing the loop to the revision cap."""
    return StubProvider(verdicts=[make_verdict(approved=False, score=5.0)])


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Keep the cached settings singleton from leaking between tests."""
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
