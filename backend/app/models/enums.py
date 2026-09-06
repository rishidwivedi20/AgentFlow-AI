"""Cross-cutting enumerations shared by the graph, the agents and the API.

``StrEnum`` members serialise as their plain string value, so these types are
usable directly in Pydantic models, in JSON responses and as LangGraph routing
literals without conversion helpers.

Agent-specific vocabularies (for example the Editor's review dimensions) live
next to the model that owns them rather than here.
"""

from __future__ import annotations

from enum import StrEnum


class Platform(StrEnum):
    """Target publishing platform.

    Only LinkedIn is supported. Adding a platform means adding a member here
    and a prompt variant - the graph and state are platform-agnostic.
    """

    LINKEDIN = "linkedin"


class Tone(StrEnum):
    """Requested voice for the generated content."""

    PROFESSIONAL = "professional"
    CONVERSATIONAL = "conversational"
    INSPIRATIONAL = "inspirational"
    ANALYTICAL = "analytical"


class AgentName(StrEnum):
    """Nodes that appear in the workflow graph.

    Used as the key for metrics and error records so that observability data
    cannot drift from the node names the graph actually registers.
    """

    RETRIEVER = "retriever"
    """Searches, fetches page text, and selects the passages most relevant to
    the topic. Makes no LLM call, but has its own latency and quota, so it is
    measured separately rather than folded into the Researcher's timing."""

    RESEARCHER = "researcher"
    PLANNER = "planner"
    WRITER = "writer"
    EDITOR = "editor"
    FACT_CHECKER = "fact_checker"
    FINALIZE = "finalize"


class AgentStatus(StrEnum):
    """Lifecycle state of a single agent execution.

    ``AgentMetric`` only ever records a terminal state - a metric is written
    once execution finishes. ``PENDING`` and ``RUNNING`` exist so that Phase 4
    live status events and this module share one vocabulary instead of the
    frontend inventing a parallel set of strings.
    """

    PENDING = "pending"
    """Not yet started. Never stored on a metric; the initial UI node state."""

    RUNNING = "running"
    """Executing. Never stored on a metric; streamed when execution begins."""

    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Whether this state can appear on a recorded AgentMetric."""
        return self in {AgentStatus.COMPLETED, AgentStatus.FAILED}


class WorkflowStatus(StrEnum):
    """Terminal (or in-flight) state of a whole workflow run.

    ``COMPLETED``               Editor approved a draft within the revision cap.
    ``MAX_REVISIONS_REACHED``   Cap hit without approval; best-scoring draft returned.
    ``COMPLETED_UNVERIFIED``    Publishable prose, but claims are unverified or
                                unsupported. Deliberately distinct from
                                COMPLETED_WITH_ERRORS: nothing crashed.
    ``COMPLETED_WITH_ERRORS``   An agent failed but a usable draft still exists.
    ``FAILED``                  No usable draft was produced.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    MAX_REVISIONS_REACHED = "max_revisions_reached"
    COMPLETED_UNVERIFIED = "completed_unverified"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Whether no further graph execution is expected from this state."""
        return self not in {WorkflowStatus.PENDING, WorkflowStatus.RUNNING}
