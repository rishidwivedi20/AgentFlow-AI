"""Emitting workflow events.

Two implementations. ``NullEmitter`` is the default everywhere, so the
synchronous endpoint, the graph's own tests and any caller that does not want
streaming are completely unaffected by this module. ``RunEventEmitter``
publishes into the run registry for streaming clients.

The emitter owns the sequence counter. Assigning it here rather than in each
call site is what guarantees a single, gap-free ordering per run - including
workflow-level events, which no agent produces.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from app.models.enums import AgentName, AgentStatus, WorkflowStatus
from app.models.events import EventType, WorkflowEvent
from app.models.metrics import TokenUsage

if TYPE_CHECKING:
    from app.services.run_registry import RunRegistry


class EventEmitter(Protocol):
    """Where workflow events go."""

    async def emit(
        self,
        event_type: EventType,
        *,
        agent: AgentName | None = None,
        status: AgentStatus | None = None,
        agent_execution: int | None = None,
        latency_seconds: float | None = None,
        usage: TokenUsage | None = None,
        error_type: str | None = None,
        message: str | None = None,
        workflow_status: WorkflowStatus | None = None,
    ) -> None:
        """Record one event. Must never raise into the caller.

        An emitter failure must not fail the workflow: streaming is an
        observability concern, and losing an event is not worth losing a run.
        """
        ...


class NullEmitter:
    """Discards every event.

    The default, so nothing outside the streaming endpoints pays for this
    feature or changes behaviour because of it.
    """

    async def emit(self, event_type: EventType, **_: object) -> None:
        return None


class RunEventEmitter:
    """Publishes events for one run into the registry."""

    def __init__(self, run_id: str, registry: RunRegistry) -> None:
        self._run_id = run_id
        self._registry = registry
        self._sequence = 0

    @property
    def next_sequence(self) -> int:
        """The sequence the next emitted event will carry. For tests."""
        return self._sequence

    async def emit(
        self,
        event_type: EventType,
        *,
        agent: AgentName | None = None,
        status: AgentStatus | None = None,
        agent_execution: int | None = None,
        latency_seconds: float | None = None,
        usage: TokenUsage | None = None,
        error_type: str | None = None,
        message: str | None = None,
        workflow_status: WorkflowStatus | None = None,
    ) -> None:
        event = WorkflowEvent(
            sequence=self._sequence,
            type=event_type,
            timestamp=datetime.now(UTC),
            agent=agent,
            status=status,
            agent_execution=agent_execution,
            latency_seconds=latency_seconds,
            usage=usage,
            error_type=error_type,
            message=message,
            workflow_status=workflow_status,
        )
        self._sequence += 1
        await self._registry.publish(self._run_id, event)
