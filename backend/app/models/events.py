"""Events streamed to clients while a workflow runs.

Safety contract
---------------
An event carries execution *status* only: which agent, what state, how long,
how many tokens. It never carries prompts, model reasoning, provider payloads,
API keys or stack traces. Messages come only from the client-safe strings the
domain-error layer already produces.

Ordering
--------
``sequence`` is assigned by the emitter, monotonically from zero within a run.
It is the replay cursor: a client that has seen up to N asks for everything
after N. Because the graph executes one node at a time, the sequence is a total
order over what happened.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from app.models.enums import AgentName, AgentStatus, WorkflowStatus
from app.models.metrics import TokenUsage


class EventType(StrEnum):
    """Kinds of event a run can emit."""

    WORKFLOW_STARTED = "workflow_started"
    """The graph is about to execute. Always the first event."""

    AGENT_STARTED = "agent_started"
    """A node began executing. Its ``agent_execution`` identifies the pass."""

    AGENT_COMPLETED = "agent_completed"
    """A node finished successfully, with latency and any token usage."""

    AGENT_FAILED = "agent_failed"
    """A node failed. The run continues; the graph decides what happens next."""

    WORKFLOW_COMPLETED = "workflow_completed"
    """The graph finished. Carries the terminal workflow status."""

    WORKFLOW_FAILED = "workflow_failed"
    """The graph itself could not execute. Always the last event."""


class WorkflowEvent(BaseModel):
    """One thing that happened during a run.

    Every field beyond ``sequence``, ``type`` and ``timestamp`` is optional
    because event kinds carry different payloads. A client switches on
    ``type`` and reads what that kind defines.
    """

    sequence: int = Field(ge=0, description="Monotonic index within the run; the replay cursor.")
    type: EventType
    timestamp: datetime

    agent: AgentName | None = Field(
        default=None, description="Node this event concerns. Absent on workflow-level events."
    )
    status: AgentStatus | None = Field(default=None, description="Node status after this event.")
    agent_execution: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Zero-based index of this execution within the run, matching "
            "AgentMetric.sequence so events join to metrics. Unique per "
            "execution, so a second 'writer' event with a higher index is what "
            "marks a pass through the feedback loop. Deliberately not the "
            "revision counter: the Writer increments that during execution, so "
            "a value read before the handler runs would lag by one and report "
            "0 for every pass."
        ),
    )
    latency_seconds: float | None = Field(
        default=None, ge=0.0, description="Wall-clock duration. Set on completion or failure."
    )
    usage: TokenUsage | None = Field(
        default=None, description="Provider-reported tokens, when the agent made an LLM call."
    )
    error_type: str | None = Field(
        default=None, description="Machine-readable category, matching AgentFlowError."
    )
    message: str | None = Field(
        default=None, description="Client-safe explanation. Never a stack trace or a prompt."
    )
    workflow_status: WorkflowStatus | None = Field(
        default=None, description="Terminal status. Set on workflow_completed."
    )
