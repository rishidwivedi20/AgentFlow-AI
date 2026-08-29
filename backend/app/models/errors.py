"""Structured, client-safe records of failures that occurred during a run.

Distinct from ``app.core.exceptions``: those are raised and become HTTP
responses, whereas these are *recorded* into workflow state so a run can
degrade gracefully and still report what went wrong. Messages here are safe to
return to the frontend - no stack traces, no internals.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import AgentName


class WorkflowErrorRecord(BaseModel):
    """A failure captured during graph execution."""

    agent: AgentName = Field(description="Node the failure occurred in.")
    error_type: str = Field(
        description="Machine-readable category, matching OrchestrAIError.error_type."
    )
    message: str = Field(description="Client-safe explanation. Never a stack trace.")
    occurred_at: datetime
