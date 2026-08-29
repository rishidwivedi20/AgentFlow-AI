"""Reusable agent execution tracking.

Every graph node is registered through :func:`as_node`, so timing, failure
capture and token accounting exist in exactly one place. An agent cannot forget
to record a metric because it never records one itself.

Also the workflow's failure boundary: an agent that raises does not propagate
out of the graph. The failure is converted into state - a failed metric plus an
error record - so the run degrades to a meaningful result instead of a 500.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from app.core.exceptions import OrchestrAIError
from app.graph.state import WorkflowState
from app.models.enums import AgentName, AgentStatus
from app.models.errors import WorkflowErrorRecord
from app.models.events import EventType
from app.models.metrics import AgentMetric, TokenUsage
from app.services.event_bus import EventEmitter, NullEmitter

logger = logging.getLogger(__name__)

StateUpdate = dict[str, Any]


@dataclass(slots=True)
class AgentRunResult:
    """What a node handler returns.

    Separating the state update from the token usage is what lets the wrapper
    build metrics without any agent knowing metrics exist.
    """

    update: StateUpdate = field(default_factory=dict)
    """Partial state update, merged by LangGraph's reducers."""

    usage: TokenUsage | None = None
    """Provider-reported usage for this execution, if any LLM call was made."""


AgentHandler = Callable[[WorkflowState], Awaitable[AgentRunResult]]
GraphNode = Callable[[WorkflowState], Awaitable[StateUpdate]]


def _build_metric(
    agent: AgentName,
    *,
    sequence: int,
    started_at: datetime,
    elapsed: float,
    status: AgentStatus,
    usage: TokenUsage | None = None,
    error: str | None = None,
) -> AgentMetric:
    return AgentMetric(
        sequence=sequence,
        agent=agent,
        status=status,
        started_at=started_at,
        ended_at=datetime.now(UTC),
        latency_seconds=round(elapsed, 4),
        usage=usage,
        error=error,
    )


async def run_agent_with_metrics(
    agent: AgentName,
    handler: AgentHandler,
    state: WorkflowState,
    emitter: EventEmitter | None = None,
) -> StateUpdate:
    """Execute ``handler``, returning its update plus a metric for the run.

    On failure, returns a state update carrying the failed metric and an error
    record rather than raising. Nothing is swallowed: the full traceback is
    logged, and the client-safe message reaches the caller through state.
    """
    events = emitter or NullEmitter()
    started_at = datetime.now(UTC)
    started = perf_counter()
    # Ordered append reducer means the current length is this execution's index.
    sequence = len(state["agent_metrics"])

    # The execution index is what lets a client tell the Writer's first pass
    # from its third, which is what makes the feedback loop drawable.
    await events.emit(
        EventType.AGENT_STARTED,
        agent=agent,
        status=AgentStatus.RUNNING,
        agent_execution=sequence,
    )

    try:
        result = await handler(state)
    except OrchestrAIError as exc:
        # Expected domain failures: LLM provider errors, timeouts, invalid
        # agent output. Message is already client-safe.
        return await _failure_update(
            agent,
            sequence,
            started_at,
            perf_counter() - started,
            exc.error_type,
            exc.message,
            exc,
            events,
        )
    except Exception as exc:  # noqa: BLE001 - deliberate boundary
        # Unexpected failures must not escape into FastAPI as a 500. Log the
        # traceback; return a generic message so no internals reach the client.
        return await _failure_update(
            agent,
            sequence,
            started_at,
            perf_counter() - started,
            "internal_error",
            f"The {agent.value} agent failed unexpectedly.",
            exc,
            events,
        )

    metric = _build_metric(
        agent,
        sequence=sequence,
        started_at=started_at,
        elapsed=perf_counter() - started,
        status=AgentStatus.COMPLETED,
        usage=result.usage,
    )
    logger.info(
        "Agent completed",
        extra={
            "agent": agent.value,
            "latency_seconds": metric.latency_seconds,
            "total_tokens": result.usage.total_tokens if result.usage else None,
        },
    )
    await events.emit(
        EventType.AGENT_COMPLETED,
        agent=agent,
        status=AgentStatus.COMPLETED,
        agent_execution=sequence,
        latency_seconds=metric.latency_seconds,
        usage=metric.usage,
    )
    return {**result.update, "agent_metrics": [metric]}


async def _failure_update(
    agent: AgentName,
    sequence: int,
    started_at: datetime,
    elapsed: float,
    error_type: str,
    message: str,
    exc: BaseException,
    events: EventEmitter,
) -> StateUpdate:
    """Convert an exception into state: a failed metric and an error record."""
    logger.exception(
        "Agent failed",
        extra={"agent": agent.value, "error_type": error_type},
        exc_info=exc,
    )
    # The failure reaches the client as a typed event, not a stack trace. The
    # traceback stays in the log above.
    await events.emit(
        EventType.AGENT_FAILED,
        agent=agent,
        status=AgentStatus.FAILED,
        agent_execution=sequence,
        latency_seconds=round(elapsed, 4),
        error_type=error_type,
        message=message,
    )
    return {
        "agent_metrics": [
            _build_metric(
                agent,
                sequence=sequence,
                started_at=started_at,
                elapsed=elapsed,
                status=AgentStatus.FAILED,
                error=message,
            )
        ],
        "errors": [
            WorkflowErrorRecord(
                agent=agent,
                error_type=error_type,
                message=message,
                occurred_at=datetime.now(UTC),
            )
        ],
    }


def as_node(
    agent: AgentName, handler: AgentHandler, emitter: EventEmitter | None = None
) -> GraphNode:
    """Wrap a handler into a LangGraph node with metrics and failure capture."""

    async def node(state: WorkflowState) -> StateUpdate:
        return await run_agent_with_metrics(agent, handler, state, emitter)

    node.__name__ = f"{agent.value}_node"
    return node
