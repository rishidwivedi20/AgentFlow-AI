"""Editor agent output: structured critique, never rewritten content.

Every model here is sent to the LLM as a response schema, so they follow the
structured-output rules documented on :class:`EditorOutput`.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class ReviewDimension(StrEnum):
    """The axes the Editor is required to assess.

    Constraining issues to a fixed vocabulary keeps critiques comparable across
    revisions and lets the Writer prompt group feedback by dimension.
    """

    CLARITY = "clarity"
    STRUCTURE = "structure"
    ENGAGEMENT = "engagement"
    REPETITION = "repetition"
    RELEVANCE = "relevance"
    TONE_CONSISTENCY = "tone_consistency"


class IssueSeverity(StrEnum):
    """How much an issue should weigh against approval."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EditorIssue(BaseModel):
    """One concrete problem found in a draft.

    Deliberately has no ``post_index`` field. Optional/nullable fields render as
    ``anyOf`` with ``null`` in JSON Schema, which Gemini's structured output
    handles poorly; the affected post is named in ``description`` instead.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dimension: ReviewDimension = Field(description="Which review axis this issue belongs to.")
    severity: IssueSeverity = Field(description="How seriously this issue harms the draft.")
    description: str = Field(
        min_length=1,
        description="What is wrong, naming the affected post where relevant.",
    )


class EditorOutput(BaseModel):
    """The Editor's verdict on one draft.

    Structured-output constraints, applying to every LLM-facing model:

    * No optional or nullable fields - ``anyOf``/``null`` is poorly supported.
    * No field defaults - the model must supply every value explicitly.
    * ``extra="forbid"`` so a response that drifts from the schema fails
      validation loudly rather than being silently truncated.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    approved: bool = Field(
        description="True only when the draft is publishable with no further changes."
    )
    quality_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Overall quality from 0 (unusable) to 10 (excellent).",
    )
    issues: list[EditorIssue] = Field(
        description="Problems found. Empty when the draft is approved without reservation."
    )
    feedback: list[str] = Field(
        description=(
            "Actionable instructions addressed to the Writer. Each item states a change "
            "to make, never rewritten content."
        )
    )

    @property
    def has_blocking_issues(self) -> bool:
        """Whether any issue is severe enough to justify a revision."""
        return any(issue.severity is IssueSeverity.HIGH for issue in self.issues)

    @model_validator(mode="after")
    def _enforce_approval_consistency(self) -> EditorOutput:
        """Reject any verdict that approves a draft carrying a high-severity issue.

        Models do contradict themselves - reporting a serious flaw and then
        setting ``approved=true``. Left alone, that flaw would finalise the run
        without a single revision, which is exactly the failure the
        Writer/Editor loop exists to prevent.

        The rule is deterministic and enforced here, at the schema boundary, so
        it cannot be bypassed: routing, the draft history and the API response
        all observe the same corrected verdict. Approval is downgraded rather
        than raising, because a contradiction is a usable verdict (the issues
        are still actionable feedback) and failing would spend a revision on a
        model quirk.

        No flag records the override. ``EditorOutput`` is sent to the provider
        as a response schema, so any field added here would become something
        the model must produce. The contradiction stays visible in the data -
        ``approved=false`` alongside a high-severity issue - and is logged.
        """
        if self.approved and self.has_blocking_issues:
            logger.warning(
                "Editor approved a draft with high-severity issues; overriding to rejected",
                extra={
                    "quality_score": self.quality_score,
                    "high_severity_count": sum(
                        1 for issue in self.issues if issue.severity is IssueSeverity.HIGH
                    ),
                },
            )
            self.approved = False
        return self
