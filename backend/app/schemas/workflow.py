"""HTTP contract for workflow execution.

Separate from ``app.models`` so the graph's internals can change without
altering the API. Response models are built from final state by
:func:`build_workflow_response`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.graph.state import WorkflowState, select_best_draft
from app.models.content import Post
from app.models.enums import Platform, Tone, WorkflowStatus
from app.models.errors import WorkflowErrorRecord
from app.models.metrics import AgentMetric
from app.models.review import EditorOutput

MIN_POSTS = 1
MAX_POSTS = 5
"""Upper bound is a cost guard, not a technical limit: each post multiplies
output tokens, and free-tier budgets are what this project is designed around."""


class WorkflowRunRequest(BaseModel):
    """Input for ``POST /api/workflows/run``."""

    topic: str = Field(
        min_length=3,
        max_length=200,
        description="Subject the posts should cover.",
    )
    platform: Platform = Field(description="Target platform.")
    tone: Tone = Field(description="Requested voice.")
    number_of_posts: int = Field(
        default=2,
        ge=MIN_POSTS,
        le=MAX_POSTS,
        description=f"How many posts to generate ({MIN_POSTS}-{MAX_POSTS}).",
    )

    @field_validator("topic")
    @classmethod
    def _reject_blank_topic(cls, value: str) -> str:
        """Reject whitespace-only topics, which pass min_length but are empty."""
        stripped = value.strip()
        if len(stripped) < 3:
            raise ValueError("topic must contain at least 3 non-whitespace characters")
        return stripped


class WorkflowRunResponse(BaseModel):
    """Result of one workflow run.

    Returned with HTTP 200 even when ``workflow_status`` is a failure: the run
    executed, and the client needs its metrics and errors. Non-2xx is reserved
    for requests that could not be processed at all.
    """

    workflow_status: WorkflowStatus
    topic: str
    platform: Platform
    tone: Tone
    final_posts: list[Post] = Field(description="Selected posts. Empty when the run failed.")
    revision_count: int = Field(description="Writer revisions performed; the initial draft is 0.")
    max_revisions: int
    approved: bool = Field(description="Whether the Editor approved the final draft.")
    quality_score: float | None = Field(
        default=None, description="Score of the selected draft, if one was evaluated."
    )
    editor_feedback: EditorOutput | None = Field(
        default=None, description="The Editor's most recent verdict."
    )
    agent_metrics: list[AgentMetric] = Field(description="One entry per agent execution.")
    errors: list[WorkflowErrorRecord] = Field(
        description="Client-safe failure records. Non-empty does not always mean failure."
    )


def build_workflow_response(state: WorkflowState) -> WorkflowRunResponse:
    """Project final graph state onto the API contract."""
    best = select_best_draft(state)
    return WorkflowRunResponse(
        workflow_status=state["workflow_status"],
        topic=state["topic"],
        platform=state["platform"],
        tone=state["tone"],
        final_posts=state["final_output"],
        revision_count=state["revision_count"],
        max_revisions=state["max_revisions"],
        approved=state["approved"],
        quality_score=best.quality_score if best else None,
        editor_feedback=state["editor_feedback"],
        agent_metrics=state["agent_metrics"],
        errors=state["errors"],
    )
