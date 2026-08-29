"""Conditional routing for the workflow graph.

Routing functions are pure: they read state and return a destination label.
They never mutate state, call an LLM or increment a counter. That is what makes
the loop's termination argument checkable by reading one module, and what lets
every branch be unit-tested without a graph or an event loop.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from app.graph.state import WorkflowState
from app.models.enums import AgentName, AgentStatus

logger = logging.getLogger(__name__)


class WriterRoute(StrEnum):
    """Destinations available after the Writer node."""

    REVIEW = "review"
    """A draft exists: send it to the Editor."""

    FINALIZE = "finalize"
    """The Writer failed: skip review entirely."""


class EditorRoute(StrEnum):
    """Destinations available after the Editor node."""

    REVISE = "revise"
    """Rejected with revisions remaining: back to the Writer."""

    FINALIZE = "finalize"
    """Approved, capped, or failed: proceed to finalisation."""


def latest_status(state: WorkflowState, agent: AgentName) -> AgentStatus | None:
    """Status of the most recent execution of ``agent``, or None if it has not run.

    Only the latest attempt matters: an earlier transient failure that a later
    revision recovered from must not permanently pin the run to finalisation.
    """
    for metric in reversed(state["agent_metrics"]):
        if metric.agent is agent:
            return metric.status
    return None


def _agent_failed(state: WorkflowState, agent: AgentName) -> bool:
    return latest_status(state, agent) is AgentStatus.FAILED


def route_after_writer(state: WorkflowState) -> WriterRoute:
    """Decide whether the draft can be reviewed.

    Reviewing after a Writer failure would either crash on a missing draft or,
    worse on a revision, re-review the stale draft the Editor already rejected.
    """
    if _agent_failed(state, AgentName.WRITER):
        logger.warning(
            "Writer failed; skipping review",
            extra={"revision": state["revision_count"]},
        )
        return WriterRoute.FINALIZE

    if state["current_draft"] is None:
        logger.warning("Writer produced no draft; skipping review")
        return WriterRoute.FINALIZE

    return WriterRoute.REVIEW


def route_after_editor(state: WorkflowState) -> EditorRoute:
    """Decide what happens after an Editor run.

    Returns ``REVISE`` under exactly one condition: the draft was not approved,
    the Editor did not fail, and the revision budget is not exhausted.

    Termination: every ``REVISE`` leads to the Writer, which strictly
    increments ``revision_count``. A strictly increasing integer bounded above
    by ``max_revisions`` cannot cycle indefinitely, so the loop runs at most
    ``max_revisions + 1`` times. Every other path terminates immediately.
    """
    # A failed Editor short-circuits: looping back would spend the revision
    # budget re-running a broken step, and there is no verdict to act on.
    if _agent_failed(state, AgentName.EDITOR):
        logger.warning(
            "Routing to finalize after editor failure",
            extra={"revision_count": state["revision_count"]},
        )
        return EditorRoute.FINALIZE

    if state["approved"]:
        return EditorRoute.FINALIZE

    if state["revision_count"] < state["max_revisions"]:
        return EditorRoute.REVISE

    logger.info(
        "Revision cap reached without approval",
        extra={"revision_count": state["revision_count"], "max_revisions": state["max_revisions"]},
    )
    return EditorRoute.FINALIZE
