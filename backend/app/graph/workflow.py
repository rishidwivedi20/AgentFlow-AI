"""Assembly and execution of the multi-agent workflow.

    START -> researcher -> (plan     -> planner  | finalize) -> END
             planner    -> (write    -> writer   | finalize)
             writer     -> (review   -> editor   | finalize)
             editor     -> (revise   -> writer   |
                            fact_check -> fact_checker | finalize)
             fact_checker -> (revise -> writer   | finalize)
             finalize   -> END

The graph is built with injected services, so tests supply stubs and the
production path supplies configured ones. Nothing here reaches for a global.
"""

from __future__ import annotations

import logging
from typing import Final

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.config import Settings, get_settings
from app.core.exceptions import WorkflowError
from app.graph.nodes import (
    build_editor_node,
    build_fact_checker_node,
    build_finalize_node,
    build_planner_node,
    build_researcher_node,
    build_writer_node,
)
from app.graph.routing import (
    EditorRoute,
    FactCheckRoute,
    PlanRoute,
    ResearchRoute,
    WriterRoute,
    route_after_editor,
    route_after_fact_check,
    route_after_planner,
    route_after_researcher,
    route_after_writer,
)
from app.graph.state import WorkflowState, initial_state
from app.models.enums import AgentName, Platform, Tone
from app.services.llm_service import LLMService
from app.services.research_service import ResearchService

logger = logging.getLogger(__name__)

# Nodes executed per revision cycle: writer, editor, fact_checker.
_STEPS_PER_ITERATION: Final = 3
# Researcher, planner and finalize run once each, plus headroom for LangGraph's
# own bookkeeping. Deliberately looser than the routing predicates need: this
# is a backstop that should never trip, not the mechanism enforcing the cap.
_FIXED_STEPS: Final = 3
_RECURSION_HEADROOM: Final = 6


def recursion_limit_for(max_revisions: int) -> int:
    """Backstop step budget for a run with the given revision cap."""
    return (max_revisions + 1) * _STEPS_PER_ITERATION + _FIXED_STEPS + _RECURSION_HEADROOM


def build_workflow(llm: LLMService, research: ResearchService) -> CompiledStateGraph:
    """Construct and compile the graph."""
    builder = StateGraph(WorkflowState)

    # Node ids are the AgentName values, so a client can map a metric's
    # `agent` field onto a graph node without a translation table.
    researcher = AgentName.RESEARCHER.value
    planner = AgentName.PLANNER.value
    writer = AgentName.WRITER.value
    editor = AgentName.EDITOR.value
    fact_checker = AgentName.FACT_CHECKER.value
    finalize = AgentName.FINALIZE.value

    builder.add_node(researcher, build_researcher_node(llm, research))
    builder.add_node(planner, build_planner_node(llm))
    builder.add_node(writer, build_writer_node(llm))
    builder.add_node(editor, build_editor_node(llm))
    builder.add_node(fact_checker, build_fact_checker_node(llm))
    builder.add_node(finalize, build_finalize_node())

    builder.add_edge(START, researcher)

    # Research and planning are prerequisites: without either, downstream
    # agents would work from nothing.
    builder.add_conditional_edges(
        researcher,
        route_after_researcher,
        {ResearchRoute.PLAN: planner, ResearchRoute.FINALIZE: finalize},
    )
    builder.add_conditional_edges(
        planner,
        route_after_planner,
        {PlanRoute.WRITE: writer, PlanRoute.FINALIZE: finalize},
    )
    # A failed Writer must skip review: there is no draft to evaluate, and on a
    # revision the stale draft would be re-reviewed.
    builder.add_conditional_edges(
        writer,
        route_after_writer,
        {WriterRoute.REVIEW: editor, WriterRoute.FINALIZE: finalize},
    )
    builder.add_conditional_edges(
        editor,
        route_after_editor,
        {
            EditorRoute.REVISE: writer,
            EditorRoute.FACT_CHECK: fact_checker,
            EditorRoute.FINALIZE: finalize,
        },
    )
    builder.add_conditional_edges(
        fact_checker,
        route_after_fact_check,
        {FactCheckRoute.REVISE: writer, FactCheckRoute.FINALIZE: finalize},
    )
    builder.add_edge(finalize, END)

    return builder.compile()


async def run_workflow(
    llm: LLMService,
    research: ResearchService,
    *,
    topic: str,
    platform: Platform,
    tone: Tone,
    number_of_posts: int,
    settings: Settings | None = None,
) -> WorkflowState:
    """Execute one workflow run and return its final state.

    Raises ``WorkflowError`` only if the graph itself fails to execute. Agent
    failures do not raise: they are captured into state by the metrics wrapper
    so the caller still receives metrics, errors and any usable output.
    """
    resolved = settings or get_settings()
    graph = build_workflow(llm, research)
    state = initial_state(
        topic=topic,
        platform=platform,
        tone=tone,
        number_of_posts=number_of_posts,
        max_revisions=resolved.max_revisions,
    )

    logger.info(
        "Workflow starting",
        extra={
            "topic": topic,
            "platform": platform.value,
            "tone": tone.value,
            "number_of_posts": number_of_posts,
            "max_revisions": resolved.max_revisions,
        },
    )

    try:
        final = await graph.ainvoke(
            state,
            config={"recursion_limit": recursion_limit_for(resolved.max_revisions)},
        )
    except Exception as exc:
        # The recursion backstop tripping means routing failed to terminate -
        # a bug worth surfacing loudly rather than degrading quietly.
        raise WorkflowError(
            "The workflow graph failed to execute.",
            details={"reason": type(exc).__name__},
        ) from exc

    return WorkflowState(**final)
