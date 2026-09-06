"""Conditional routing for the workflow graph.

Routing functions are pure: they read state and return a destination label.
They never mutate state, call a service or increment a counter. That is what
makes the loop's termination argument checkable by reading one module, and what
lets every branch be unit-tested without a graph or an event loop.

Termination
-----------
Three routes can send work back to the Writer: an Editor rejection, a Fact
Checker rejection, and nothing else. Both are gated on the same
``revision_count`` budget, and the Writer strictly increments that counter on
every revision. A strictly increasing integer bounded above by
``max_revisions`` cannot cycle indefinitely, so the Writer runs at most
``max_revisions + 1`` times no matter which agent sent it back.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from app.graph.state import WorkflowState
from app.models.enums import AgentName, AgentStatus

logger = logging.getLogger(__name__)


class RetrievalRoute(StrEnum):
    """Destinations after the Retriever node."""

    RESEARCH = "research"
    """Sources were found: continue, with or without retrieved evidence."""

    FINALIZE = "finalize"
    """No sources at all: there is nothing for any later agent to work from."""


class ResearchRoute(StrEnum):
    """Destinations after the Researcher node."""

    PLAN = "plan"
    """Research succeeded: continue to planning."""

    FINALIZE = "finalize"
    """The Researcher failed: nothing downstream can proceed safely."""


class PlanRoute(StrEnum):
    """Destinations after the Planner node."""

    WRITE = "write"
    FINALIZE = "finalize"


class WriterRoute(StrEnum):
    """Destinations after the Writer node."""

    REVIEW = "review"
    """A draft exists: send it to the Editor."""

    FINALIZE = "finalize"
    """The Writer failed: skip review entirely."""


class EditorRoute(StrEnum):
    """Destinations after the Editor node."""

    REVISE = "revise"
    """Rejected with revisions remaining: back to the Writer."""

    FACT_CHECK = "fact_check"
    """Approved: verify the draft's claims before finalising."""

    FINALIZE = "finalize"
    """Capped or failed: finalise without further work."""


class FactCheckRoute(StrEnum):
    """Destinations after the Fact Checker node."""

    REVISE = "revise"
    """Unsupported claims found, with revisions remaining."""

    FINALIZE = "finalize"
    """Claims verified, cap reached, or the checker failed."""


def latest_status(state: WorkflowState, agent: AgentName) -> AgentStatus | None:
    """Status of the most recent execution of ``agent``, or None if not run.

    Only the latest attempt matters: an earlier transient failure that a later
    revision recovered from must not permanently pin the run to finalisation.
    """
    for metric in reversed(state["agent_metrics"]):
        if metric.agent is agent:
            return metric.status
    return None


def _agent_failed(state: WorkflowState, agent: AgentName) -> bool:
    return latest_status(state, agent) is AgentStatus.FAILED


def _revisions_remain(state: WorkflowState) -> bool:
    return state["revision_count"] < state["max_revisions"]


def route_after_retriever(state: WorkflowState) -> RetrievalRoute:
    """Stop the run only if the search itself failed.

    Gated on the node failing, not on what it found. Two non-failures are
    deliberately allowed through:

    * **Retrieval unavailable.** An exhausted embedding quota, or pages that
      could not be extracted, leaves the run with search snippets - which is
      exactly how it worked before Phase 5, and far better than abandoning a
      search already paid for.
    * **A search that legitimately matched nothing.** Since Phase 3 this
      degrades to ungrounded content carrying an explicit warning, rather than
      failing. That judgement is unchanged here.
    """
    if _agent_failed(state, AgentName.RETRIEVER):
        logger.warning("Search failed; finalising without content")
        return RetrievalRoute.FINALIZE
    return RetrievalRoute.RESEARCH


def route_after_researcher(state: WorkflowState) -> ResearchRoute:
    """Stop the run if research failed.

    Continuing would hand the Writer no grounding and the Fact Checker no
    evidence, so every downstream claim would be unverifiable by construction.
    A search that legitimately found nothing is not a failure and proceeds.
    """
    if _agent_failed(state, AgentName.RESEARCHER) or state["research"] is None:
        logger.warning("Research unavailable; finalising without content")
        return ResearchRoute.FINALIZE
    return ResearchRoute.PLAN


def route_after_planner(state: WorkflowState) -> PlanRoute:
    """Stop the run if planning failed."""
    if _agent_failed(state, AgentName.PLANNER) or state["content_plan"] is None:
        logger.warning("Content plan unavailable; finalising without content")
        return PlanRoute.FINALIZE
    return PlanRoute.WRITE


def route_after_writer(state: WorkflowState) -> WriterRoute:
    """Decide whether the draft can be reviewed.

    Reviewing after a Writer failure would either crash on a missing draft or,
    worse on a revision, re-review the stale draft the Editor already rejected.
    """
    if _agent_failed(state, AgentName.WRITER) or state["current_draft"] is None:
        logger.warning("Writer produced no draft; skipping review")
        return WriterRoute.FINALIZE
    return WriterRoute.REVIEW


def route_after_editor(state: WorkflowState) -> EditorRoute:
    """Decide what happens after an Editor run.

    An approved draft goes to fact checking rather than straight to output:
    prose quality and factual support are separate standards, and the Editor
    judges only the former.
    """
    if _agent_failed(state, AgentName.EDITOR):
        logger.warning(
            "Routing to finalize after editor failure",
            extra={"revision_count": state["revision_count"]},
        )
        return EditorRoute.FINALIZE

    if state["approved"]:
        return EditorRoute.FACT_CHECK

    if _revisions_remain(state):
        return EditorRoute.REVISE

    logger.info(
        "Revision cap reached without approval",
        extra={"revision_count": state["revision_count"], "max_revisions": state["max_revisions"]},
    )
    return EditorRoute.FINALIZE


def route_after_fact_check(state: WorkflowState) -> FactCheckRoute:
    """Decide whether unsupported claims warrant another revision.

    Shares the Editor's revision budget deliberately. A separate allowance
    would let the two loops alternate and multiply the worst-case token spend.
    """
    if _agent_failed(state, AgentName.FACT_CHECKER):
        logger.warning("Routing to finalize after fact checker failure")
        return FactCheckRoute.FINALIZE

    report = state["fact_check"]
    if report is None or not report.has_critical_issues:
        return FactCheckRoute.FINALIZE

    if _revisions_remain(state):
        logger.info(
            "Unsupported claims found; returning to writer",
            extra={
                "unsupported": len(report.unsupported),
                "revision_count": state["revision_count"],
            },
        )
        return FactCheckRoute.REVISE

    logger.warning(
        "Unsupported claims remain but the revision cap is reached",
        extra={"unsupported": len(report.unsupported)},
    )
    return FactCheckRoute.FINALIZE
