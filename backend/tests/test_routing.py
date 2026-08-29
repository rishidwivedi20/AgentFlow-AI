"""Routing is a pure function, so every branch is testable without a graph."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.graph.routing import (
    EditorRoute,
    WriterRoute,
    route_after_editor,
    route_after_writer,
)
from app.graph.state import WorkflowState, initial_state
from app.models.content import WriterOutput
from app.models.enums import AgentName, AgentStatus, Platform, Tone
from app.models.metrics import AgentMetric
from tests.conftest import make_posts


def _state(*, approved: bool, revision_count: int, max_revisions: int = 3) -> WorkflowState:
    state = initial_state(
        topic="topic",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
        max_revisions=max_revisions,
    )
    state["approved"] = approved
    state["revision_count"] = revision_count
    return state


def _metric(agent: AgentName, status: AgentStatus, sequence: int = 0) -> AgentMetric:
    now = datetime.now(UTC)
    return AgentMetric(
        sequence=sequence,
        agent=agent,
        status=status,
        started_at=now,
        ended_at=now,
        latency_seconds=0.0,
    )


def test_approved_draft_finalizes():
    assert route_after_editor(_state(approved=True, revision_count=0)) is EditorRoute.FINALIZE


@pytest.mark.parametrize("revision_count", [0, 1, 2])
def test_rejected_under_cap_revises(revision_count):
    state = _state(approved=False, revision_count=revision_count)
    assert route_after_editor(state) is EditorRoute.REVISE


def test_rejected_at_cap_finalizes():
    assert route_after_editor(_state(approved=False, revision_count=3)) is EditorRoute.FINALIZE


def test_rejected_above_cap_finalizes():
    """Defensive: a counter past the cap must not resume looping."""
    assert route_after_editor(_state(approved=False, revision_count=9)) is EditorRoute.FINALIZE


def test_approved_at_cap_still_finalizes():
    assert route_after_editor(_state(approved=True, revision_count=3)) is EditorRoute.FINALIZE


def test_agent_failure_short_circuits_to_finalize():
    """A failed agent must not loop back to the writer."""
    state = _state(approved=False, revision_count=0)
    state["agent_metrics"] = [_metric(AgentName.EDITOR, AgentStatus.FAILED)]
    assert route_after_editor(state) is EditorRoute.FINALIZE


def test_recovered_failure_does_not_pin_to_finalize():
    """Only the latest writer/editor execution matters."""
    state = _state(approved=False, revision_count=0)
    state["agent_metrics"] = [
        _metric(AgentName.EDITOR, AgentStatus.FAILED),
        _metric(AgentName.WRITER, AgentStatus.COMPLETED),
        _metric(AgentName.EDITOR, AgentStatus.COMPLETED),
    ]
    assert route_after_editor(state) is EditorRoute.REVISE


def test_finalize_metric_is_ignored_by_failure_check():
    state = _state(approved=False, revision_count=0)
    state["agent_metrics"] = [
        _metric(AgentName.EDITOR, AgentStatus.COMPLETED),
        _metric(AgentName.FINALIZE, AgentStatus.FAILED),
    ]
    assert route_after_editor(state) is EditorRoute.REVISE


def test_routing_does_not_mutate_state():
    state = _state(approved=False, revision_count=1)
    before = dict(state)
    route_after_editor(state)
    assert dict(state) == before


# --- route_after_writer ----------------------------------------------------

def test_writer_route_reviews_when_draft_exists():

    state = _state(approved=False, revision_count=0)
    state["current_draft"] = WriterOutput(posts=make_posts(2))
    state["agent_metrics"] = [_metric(AgentName.WRITER, AgentStatus.COMPLETED)]
    assert route_after_writer(state) is WriterRoute.REVIEW


def test_writer_route_skips_review_when_writer_failed():

    state = _state(approved=False, revision_count=1)
    # A stale draft from a previous revision is present, so the failure - not
    # the draft's presence - must drive the decision.
    state["current_draft"] = WriterOutput(posts=make_posts(2))
    state["agent_metrics"] = [_metric(AgentName.WRITER, AgentStatus.FAILED)]
    assert route_after_writer(state) is WriterRoute.FINALIZE


def test_writer_route_skips_review_when_no_draft():

    assert route_after_writer(_state(approved=False, revision_count=0)) is WriterRoute.FINALIZE
