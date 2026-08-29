"""Assembly and execution of the Writer/Editor workflow.

    START -> writer -> (review -> editor | finalize) -> END
                          editor -> (revise -> writer | finalize) -> END

The graph is built with an injected :class:`LLMService`, so tests supply a stub
provider and the production path supplies the configured one. Nothing here
reaches for a global.
"""

from __future__ import annotations

import logging
from typing import Final

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.config import Settings, get_settings
from app.core.exceptions import WorkflowError
from app.graph.nodes import build_editor_node, build_finalize_node, build_writer_node
from app.graph.routing import (
    EditorRoute,
    WriterRoute,
    route_after_editor,
    route_after_writer,
)
from app.graph.state import WorkflowState, initial_state
from app.models.enums import AgentName, Platform, Tone
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

# Nodes executed per loop iteration (writer + editor), plus finalize, plus
# headroom for LangGraph's own bookkeeping. Deliberately larger than the
# routing predicate needs: this is a backstop that should never trip, not the
# mechanism that enforces the cap.
_STEPS_PER_ITERATION: Final = 2
_RECURSION_HEADROOM: Final = 5


def recursion_limit_for(max_revisions: int) -> int:
    """Backstop step budget for a run with the given revision cap."""
    return (max_revisions + 1) * _STEPS_PER_ITERATION + _RECURSION_HEADROOM


def build_workflow(llm: LLMService) -> CompiledStateGraph:
    """Construct and compile the graph."""
    builder = StateGraph(WorkflowState)

    # Node ids are the AgentName values, so a client can map a metric's
    # `agent` field onto a graph node without a translation table.
    writer, editor, finalize = (
        AgentName.WRITER.value,
        AgentName.EDITOR.value,
        AgentName.FINALIZE.value,
    )

    builder.add_node(writer, build_writer_node(llm))
    builder.add_node(editor, build_editor_node(llm))
    builder.add_node(finalize, build_finalize_node())

    builder.add_edge(START, writer)
    # A failed Writer must skip review: there is no draft to evaluate, and on a
    # revision the stale draft would be re-reviewed.
    builder.add_conditional_edges(
        writer,
        route_after_writer,
        {
            WriterRoute.REVIEW: editor,
            WriterRoute.FINALIZE: finalize,
        },
    )
    builder.add_conditional_edges(
        editor,
        route_after_editor,
        {
            EditorRoute.REVISE: writer,
            EditorRoute.FINALIZE: finalize,
        },
    )
    builder.add_edge(finalize, END)

    return builder.compile()


async def run_workflow(
    llm: LLMService,
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
    graph = build_workflow(llm)
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
