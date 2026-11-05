"""HTTP contracts for asynchronous runs.

Separate from ``schemas.workflow``, which describes the synchronous endpoint
and the workflow result itself. A *run* is the execution; a *workflow result*
is what it produced.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.workflow import WorkflowRunResponse
from app.services.run_registry import RunState
from app.services.run_store import RunSummary


class RunAcceptedResponse(BaseModel):
    """Returned by ``POST /api/workflows`` once a run has been queued.

    The work has not happened yet: this is an acknowledgement plus the two URLs
    needed to follow it.
    """

    run_id: str
    state: RunState
    topic: str
    created_at: datetime
    events_url: str = Field(description="SSE stream of execution events for this run.")
    status_url: str = Field(description="Poll or refresh target for the final result.")


class RunStatusResponse(BaseModel):
    """Current state of a run, with its result once one exists.

    Serves both polling and page refresh: a client that lost its event stream
    reads this to recover, without needing the stream to have been observed.
    """

    run_id: str
    state: RunState
    topic: str
    created_at: datetime
    finished_at: datetime | None = None
    event_count: int = Field(
        description="Events buffered so far; also the cursor a fresh stream would replay from."
    )
    result: WorkflowRunResponse | None = Field(
        default=None, description="The workflow result. Present once the run has finished."
    )
    error_type: str | None = Field(
        default=None, description="Set when the run itself failed to execute."
    )
    error_message: str | None = Field(
        default=None, description="Client-safe explanation. Never a stack trace."
    )


class RunHistoryResponse(BaseModel):
    """A page of finished runs from the run store, newest first."""

    history_enabled: bool = Field(
        description="False when run history is switched off; the list is then always empty."
    )
    total: int = Field(ge=0, description="Finished runs stored in all.")
    limit: int
    offset: int
    runs: list[RunSummary] = Field(default_factory=list)
