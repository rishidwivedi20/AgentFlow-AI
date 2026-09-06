"""The Retriever node: what it writes to state, and how it degrades.

The node's whole reason for existing is that retrieval can fail on its own
terms - a spent embedding quota, a page that will not extract - without that
being a reason to abandon a search already paid for. Most of what is tested
here is that degradation, because it is the behaviour a passing pipeline would
never exercise.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.exceptions import EmbeddingRateLimitError
from app.graph.nodes import build_retriever_node
from app.graph.routing import RetrievalRoute, route_after_retriever
from app.graph.state import initial_state
from app.models.enums import AgentName, AgentStatus, Platform, Tone, WorkflowStatus
from app.models.evidence import RetrievalStatus
from app.models.metrics import AgentMetric
from tests.conftest import (
    StubEmbeddingProvider,
    StubProvider,
    StubSearchProvider,
    make_research,
    make_retrieval,
    make_sources,
    run_pipeline,
)

PAGE = " ".join(f"alpha beta paragraph number {index} of the page." for index in range(30))
OFF_TOPIC_PAGE = " ".join(f"zeta epsilon paragraph number {index}." for index in range(30))


def make_state(**overrides):
    state = initial_state(
        topic="alpha beta",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
        max_revisions=3,
    )
    state.update(overrides)
    return state


async def run_node(*, page_text=None, retrieval=None, fail_search=False, sources=3):
    """Execute the node once and return its state update."""
    search = StubSearchProvider(
        sources=make_sources(sources), fail=fail_search, page_text=page_text
    )
    node = build_retriever_node(
        make_research(search, include_page_text=retrieval is not None), retrieval
    )
    return await node(make_state())


# --- the happy path --------------------------------------------------------


async def test_sources_and_evidence_both_reach_state():
    update = await run_node(page_text={0: PAGE, 1: OFF_TOPIC_PAGE}, retrieval=make_retrieval())

    assert len(update["sources"]) == 3
    evidence = update["evidence"]
    assert evidence.status is RetrievalStatus.RETRIEVED
    assert evidence.chunks
    assert all(chunk.source_index in {0, 1} for chunk in evidence.chunks)


async def test_the_topic_is_what_chunks_are_ranked_against():
    update = await run_node(page_text={0: PAGE}, retrieval=make_retrieval())
    assert update["evidence"].query == "alpha beta"


async def test_the_node_reports_no_token_usage():
    """It makes no LLM call. Zero would read as "this was free"."""
    search = StubSearchProvider(sources=make_sources(2), page_text={0: PAGE})
    node = build_retriever_node(make_research(search, include_page_text=True), make_retrieval())
    update = await node(make_state())
    assert update["agent_metrics"][0].usage is None
    assert update["agent_metrics"][0].agent is AgentName.RETRIEVER


# --- degradation, not failure ----------------------------------------------


async def test_no_retrieval_service_still_yields_sources():
    """Unconfigured embeddings must not cost the run its search."""
    update = await run_node(retrieval=None)

    assert len(update["sources"]) == 3
    assert update["evidence"].status is RetrievalStatus.NOT_ATTEMPTED
    assert "disabled" in update["evidence"].warnings[0]


async def test_pages_that_cannot_be_extracted_are_not_an_error():
    """A PDF or a paywall returns no text. The sources stay citable."""
    update = await run_node(page_text={}, retrieval=make_retrieval())

    assert len(update["sources"]) == 3
    assert update["evidence"].status is RetrievalStatus.NOT_ATTEMPTED
    assert "No page text" in update["evidence"].warnings[0]


async def test_an_embedding_failure_keeps_the_sources_the_search_paid_for():
    """The metrics wrapper discards a handler's update when it raises, so the
    node must catch this itself or a retrieval outage becomes a dead run."""

    class ExhaustedRetrieval:
        async def retrieve(self, *, query, documents):
            raise EmbeddingRateLimitError("Embedding quota exhausted.")

    update = await run_node(page_text={0: PAGE}, retrieval=ExhaustedRetrieval())

    assert len(update["sources"]) == 3
    assert update["evidence"].status is RetrievalStatus.NOT_ATTEMPTED
    assert "Retrieval failed" in update["evidence"].warnings[0]


async def test_an_embedding_failure_is_still_attributed_to_the_retriever():
    """Degrading must not mean the failure disappears from the run's record."""

    class ExhaustedRetrieval:
        async def retrieve(self, *, query, documents):
            raise EmbeddingRateLimitError("Embedding quota exhausted.")

    update = await run_node(page_text={0: PAGE}, retrieval=ExhaustedRetrieval())

    assert len(update["errors"]) == 1
    assert update["errors"][0].agent is AgentName.RETRIEVER
    assert update["errors"][0].error_type == "embedding_rate_limited"
    # The node itself completed: it produced usable output.
    assert update["agent_metrics"][0].status is AgentStatus.COMPLETED


async def test_an_unmapped_retrieval_error_also_keeps_the_sources():
    """The invariant holds whatever the cause - a network error from the
    embedding SDK that no adapter mapped included."""

    class BrokenRetrieval:
        async def retrieve(self, *, query, documents):
            raise ConnectionResetError("socket died")

    update = await run_node(page_text={0: PAGE}, retrieval=BrokenRetrieval())

    assert len(update["sources"]) == 3
    assert update["errors"][0].error_type == "internal_error"
    # No internals in the client-facing message.
    assert "socket" not in update["errors"][0].message
    assert update["agent_metrics"][0].status is AgentStatus.COMPLETED


async def test_a_search_failure_is_a_real_node_failure():
    """Nothing usable came back, so this is not degradation."""
    update = await run_node(fail_search=True, retrieval=make_retrieval())

    assert update["agent_metrics"][0].status is AgentStatus.FAILED
    assert update["errors"][0].agent is AgentName.RETRIEVER
    assert "sources" not in update


# --- page text is only fetched when it will be used ------------------------


async def test_page_text_is_requested_only_when_retrieval_can_use_it():
    """It is ~15x the size of a snippet and may cost more provider credits."""
    with_retrieval = StubSearchProvider(sources=make_sources(2))
    node = build_retriever_node(
        make_research(with_retrieval, include_page_text=True), make_retrieval()
    )
    await node(make_state())
    assert with_retrieval.include_page_text is True

    without = StubSearchProvider(sources=make_sources(2))
    node = build_retriever_node(make_research(without, include_page_text=False), None)
    await node(make_state())
    assert without.include_page_text is False


# --- routing ---------------------------------------------------------------


def test_routing_continues_when_retrieval_was_never_attempted():
    """Snippets alone are how the pipeline worked before Phase 5."""
    state = make_state(sources=make_sources(3), evidence=None)
    assert route_after_retriever(state) is RetrievalRoute.RESEARCH


def test_routing_continues_when_a_search_legitimately_matched_nothing():
    """Since Phase 3 this degrades to ungrounded content with a warning."""
    state = make_state(sources=[])
    assert route_after_retriever(state) is RetrievalRoute.RESEARCH


def test_routing_stops_only_when_the_search_failed():
    now = datetime.now(UTC)
    state = make_state(
        agent_metrics=[
            AgentMetric(
                sequence=0,
                agent=AgentName.RETRIEVER,
                status=AgentStatus.FAILED,
                started_at=now,
                ended_at=now,
                latency_seconds=0.1,
                error="boom",
            )
        ]
    )
    assert route_after_retriever(state) is RetrievalRoute.FINALIZE


# --- end to end through the graph ------------------------------------------


@pytest.mark.parametrize("with_retrieval", [True, False])
async def test_the_pipeline_completes_with_and_without_retrieval(
    settings, run_kwargs, with_retrieval
):
    """The graph's shape does not change with configuration - only the
    evidence does - so a visualisation of it never misrepresents what ran."""
    retrieval = make_retrieval(StubEmbeddingProvider()) if with_retrieval else None
    search = StubSearchProvider(sources=make_sources(3), page_text={0: PAGE})
    state = await run_pipeline(
        StubProvider(), settings, run_kwargs, search=search, retrieval=retrieval
    )

    assert state["workflow_status"] is WorkflowStatus.COMPLETED
    assert len(state["sources"]) == 3
    agents = [metric.agent for metric in state["agent_metrics"]]
    assert agents[0] is AgentName.RETRIEVER
    assert agents[1] is AgentName.RESEARCHER

    # The grade depends on how well the page matches the topic, which this
    # test does not control. What it does assert is whether retrieval ran.
    if with_retrieval:
        assert state["evidence"].status is not RetrievalStatus.NOT_ATTEMPTED
        assert state["evidence"].is_usable
    else:
        assert state["evidence"].status is RetrievalStatus.NOT_ATTEMPTED


async def test_sources_survive_a_researcher_that_fails_after_them(settings, run_kwargs):
    """They were really found. Losing them would also lose what a fact check
    would have to cite against."""
    provider = StubProvider(fail_on=["researcher"])
    search = StubSearchProvider(sources=make_sources(3))
    state = await run_pipeline(provider, settings, run_kwargs, search=search)

    assert state["research"] is None
    assert len(state["sources"]) == 3
