"""Internal domain models used inside the orchestration graph state.

Deliberately distinct from ``app.schemas``, which describes the HTTP boundary:
the graph's internals can change without altering the API contract.
"""

from app.models.content import DraftRecord, Post, WriterOutput
from app.models.enums import AgentName, AgentStatus, Platform, Tone, WorkflowStatus
from app.models.errors import WorkflowErrorRecord
from app.models.metrics import AgentMetric, TokenUsage
from app.models.review import EditorIssue, EditorOutput, IssueSeverity, ReviewDimension

__all__ = [
    "AgentMetric",
    "AgentName",
    "AgentStatus",
    "DraftRecord",
    "EditorIssue",
    "EditorOutput",
    "IssueSeverity",
    "Platform",
    "Post",
    "ReviewDimension",
    "TokenUsage",
    "Tone",
    "WorkflowErrorRecord",
    "WorkflowStatus",
    "WriterOutput",
]
