"""Graph node implementations.

Each node adapts an agent to the graph: it reads what the agent needs from
state and turns the agent's result into a state update. Nodes own the loop
bookkeeping that agents must not know about - most importantly the revision
counter.
"""

from __future__ import annotations

import logging

from app.agents.editor import run_editor
from app.agents.fact_checker import run_fact_checker
from app.agents.planner import run_planner
from app.agents.researcher import run_researcher
from app.agents.writer import run_writer
from app.graph.state import WorkflowState, select_best_draft
from app.models.content import DraftRecord
from app.models.enums import AgentName, WorkflowStatus
from app.services.llm_service import LLMService
from app.services.metrics_service import AgentRunResult, GraphNode, StateUpdate, as_node
from app.services.research_service import ResearchService

logger = logging.getLogger(__name__)


def build_researcher_node(llm: LLMService, research: ResearchService) -> GraphNode:
    """Researcher node: search the topic and extract cited facts."""

    async def handler(state: WorkflowState) -> AgentRunResult:
        result, llm_result = await run_researcher(llm, research, topic=state["topic"])
        return AgentRunResult(
            update={
                "research": result,
                # Rendered here so the Writer never depends on the research
                # model's shape - the same seam it has used since Phase 2.
                "research_context": result.as_writer_context(),
            },
            # None when the search found nothing and no LLM call was made.
            usage=llm_result.usage if llm_result else None,
        )

    return as_node(AgentName.RESEARCHER, handler)


def build_planner_node(llm: LLMService) -> GraphNode:
    """Planner node: assign each post a distinct angle."""

    async def handler(state: WorkflowState) -> AgentRunResult:
        research = state["research"]
        if research is None:
            # Unreachable: routing sends a failed Researcher to finalize.
            raise RuntimeError("Planner invoked with no research.")

        result = await run_planner(
            llm,
            topic=state["topic"],
            platform=state["platform"],
            tone=state["tone"],
            research=research,
            number_of_posts=state["number_of_posts"],
        )
        return AgentRunResult(update={"content_plan": result.value}, usage=result.usage)

    return as_node(AgentName.PLANNER, handler)


def build_writer_node(llm: LLMService) -> GraphNode:
    """Writer node: produce a draft, advancing the revision counter."""

    async def handler(state: WorkflowState) -> AgentRunResult:
        feedback = state["editor_feedback"]
        report = state["fact_check"]

        # Unsupported claims are only worth resending when the Fact Checker
        # actually rejected this draft.
        fact_feedback = (
            report.as_writer_feedback()
            if report is not None and report.has_critical_issues
            else None
        )

        # Either critic can make a run a revision. The counter advances here
        # and nowhere else, which is what bounds both loops together.
        is_revision = feedback is not None or fact_feedback is not None
        revision = state["revision_count"] + 1 if is_revision else 0

        result = await run_writer(
            llm,
            topic=state["topic"],
            platform=state["platform"],
            tone=state["tone"],
            number_of_posts=state["number_of_posts"],
            research_context=state["research_context"],
            content_plan=state["content_plan"],
            feedback=feedback,
            fact_check_feedback=fact_feedback,
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
            # Unreachable: routing sends a failed Writer to finalize.
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
        # participant that knows the score.
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


def build_fact_checker_node(llm: LLMService) -> GraphNode:
    """Fact Checker node: judge the approved draft's claims against research."""

    async def handler(state: WorkflowState) -> AgentRunResult:
        draft = state["current_draft"]
        research = state["research"]
        if draft is None or research is None:
            # Unreachable: routing guarantees both exist on this path.
            raise RuntimeError("Fact checker invoked without a draft and research.")

        result = await run_fact_checker(llm, research=research, posts=draft.posts)
        return AgentRunResult(update={"fact_check": result.value}, usage=result.usage)

    return as_node(AgentName.FACT_CHECKER, handler)


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
        # No verdict was produced, but content exists. Returning it beats
        # returning nothing, flagged by the status.
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

    Precedence matters: a run with no usable output is FAILED regardless of
    why, and errors downgrade an otherwise clean completion rather than hiding.
    Unsupported claims that survived the revision budget are reported the same
    way - the posts are usable, but the run did not fully succeed.
    """
    if not has_posts:
        return WorkflowStatus.FAILED
    if had_errors:
        return WorkflowStatus.COMPLETED_WITH_ERRORS

    report = state["fact_check"]
    if report is not None and report.has_critical_issues:
        return WorkflowStatus.COMPLETED_WITH_ERRORS

    if state["approved"]:
        return WorkflowStatus.COMPLETED
    return WorkflowStatus.MAX_REVISIONS_REACHED
