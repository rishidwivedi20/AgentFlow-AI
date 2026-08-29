"""Graph node implementations.

Each node adapts an agent to the graph: it reads what the agent needs from
state and turns the agent's result into a state update. Nodes own the loop
bookkeeping that agents must not know about - most importantly the revision
counter.
"""

from __future__ import annotations

import logging

from app.agents.editor import run_editor
from app.agents.writer import run_writer
from app.graph.state import WorkflowState, select_best_draft
from app.models.content import DraftRecord
from app.models.enums import AgentName, WorkflowStatus
from app.services.llm_service import LLMService
from app.services.metrics_service import AgentRunResult, GraphNode, StateUpdate, as_node

logger = logging.getLogger(__name__)


def build_writer_node(llm: LLMService) -> GraphNode:
    """Writer node: produce a draft, advancing the revision counter."""

    async def handler(state: WorkflowState) -> AgentRunResult:
        feedback = state["editor_feedback"]
        # The counter advances here and nowhere else. Presence of feedback is
        # what makes a run a revision; the initial draft leaves it at 0.
        revision = state["revision_count"] + 1 if feedback is not None else 0

        result = await run_writer(
            llm,
            topic=state["topic"],
            platform=state["platform"],
            tone=state["tone"],
            number_of_posts=state["number_of_posts"],
            research_context=state["research_context"],
            feedback=feedback,
        )
        return AgentRunResult(
            update={"current_draft": result.value, "revision_count": revision},
            usage=result.usage,
        )

    return as_node(AgentName.WRITER, handler)


def build_editor_node(llm: LLMService) -> GraphNode:
    """Editor node: evaluate the current draft and record it with its verdict."""

    async def handler(state: WorkflowState) -> AgentRunResult:
        draft = state["current_draft"]
        if draft is None:
            # Only reachable if the Writer failed. Routing sends failures to
            # finalize, so this is a guard against a future graph edit.
            raise RuntimeError("Editor invoked with no draft to review.")

        result = await run_editor(
            llm,
            topic=state["topic"],
            platform=state["platform"],
            tone=state["tone"],
            posts=draft.posts,
            revision=state["revision_count"],
        )
        verdict = result.value
        # The Editor appends the draft record because it is the only
        # participant that knows the score. Draft and evaluation are stored as
        # one already-paired unit.
        record = DraftRecord(
            revision=state["revision_count"], posts=draft.posts, evaluation=verdict
        )
        return AgentRunResult(
            update={
                "editor_feedback": verdict,
                "approved": verdict.approved,
                "drafts": [record],
            },
            usage=result.usage,
        )

    return as_node(AgentName.EDITOR, handler)


def build_finalize_node() -> GraphNode:
    """Finalize node: choose the output and assign a terminal status.

    Makes no LLM call, so it reports no token usage.
    """

    async def handler(state: WorkflowState) -> AgentRunResult:
        return AgentRunResult(update=_finalize(state))

    return as_node(AgentName.FINALIZE, handler)


def _finalize(state: WorkflowState) -> StateUpdate:
    """Select final posts and derive the terminal workflow status."""
    best = select_best_draft(state)
    had_errors = bool(state["errors"])

    if best is not None:
        posts = best.posts
        source = f"revision {best.revision} (score {best.quality_score:.1f})"
    elif state["current_draft"] is not None:
        # The Editor never produced a verdict, but the Writer produced content.
        # Returning it beats returning nothing, flagged by the status.
        posts = state["current_draft"].posts
        source = "unevaluated draft"
    else:
        posts = []
        source = "none"

    status = _derive_status(state, has_posts=bool(posts), had_errors=had_errors)
    logger.info(
        "Workflow finalised",
        extra={
            "status": status.value,
            "selected": source,
            "revision_count": state["revision_count"],
            "post_count": len(posts),
        },
    )
    return {"final_output": posts, "workflow_status": status}


def _derive_status(
    state: WorkflowState, *, has_posts: bool, had_errors: bool
) -> WorkflowStatus:
    """Map the finished run onto a terminal status.

    Precedence matters: a run with no usable output is FAILED regardless of why,
    and errors downgrade an otherwise clean completion rather than hiding.
    """
    if not has_posts:
        return WorkflowStatus.FAILED
    if had_errors:
        return WorkflowStatus.COMPLETED_WITH_ERRORS
    if state["approved"]:
        return WorkflowStatus.COMPLETED
    return WorkflowStatus.MAX_REVISIONS_REACHED
