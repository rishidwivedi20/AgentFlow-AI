"""Fact check: claims extracted from a draft, judged against the research.

The Fact Checker verifies only against the research gathered for this run. It
does not search, and it does not fall back on the model's own knowledge - a
claim the sources do not support is unsupported, however true it may be.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ClaimVerdict(StrEnum):
    """How a claim stands against the collected research."""

    VERIFIED = "verified"
    """Directly supported by a research fact."""

    UNCERTAIN = "uncertain"
    """Plausible and related to the research, but not directly supported."""

    UNSUPPORTED = "unsupported"
    """Contradicted by the research, or a specific claim with nothing behind it."""


class FactCheckClaim(BaseModel):
    """One factual assertion found in the draft, with its verdict.

    LLM-facing: all fields required, none nullable.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim: str = Field(
        min_length=1, description="The assertion quoted or closely paraphrased from the draft."
    )
    verdict: ClaimVerdict = Field(description="How the claim stands against the research.")
    reasoning: str = Field(
        min_length=1,
        description="Which research fact supports or fails to support this claim, and why.",
    )
    post_index: int = Field(
        ge=1, description="Which post the claim appears in, numbered from 1."
    )


class FactCheckReport(BaseModel):
    """The Fact Checker's verdict on a whole draft."""

    model_config = ConfigDict(extra="forbid")

    claims: list[FactCheckClaim] = Field(
        description="Every factual assertion found. Empty when the draft makes none."
    )

    @property
    def unsupported(self) -> list[FactCheckClaim]:
        return [c for c in self.claims if c.verdict is ClaimVerdict.UNSUPPORTED]

    @property
    def uncertain(self) -> list[FactCheckClaim]:
        return [c for c in self.claims if c.verdict is ClaimVerdict.UNCERTAIN]

    @property
    def verified(self) -> list[FactCheckClaim]:
        return [c for c in self.claims if c.verdict is ClaimVerdict.VERIFIED]

    @property
    def has_critical_issues(self) -> bool:
        """Whether the draft must go back to the Writer.

        Only ``UNSUPPORTED`` is critical. ``UNCERTAIN`` is reported but does not
        trigger a rewrite: treating "not directly evidenced" as a blocker would
        spend the revision budget on hedged, unfalsifiable prose.
        """
        return bool(self.unsupported)

    def as_writer_feedback(self) -> str:
        """Render unsupported claims as revision instructions."""
        lines = [
            "A fact check found claims the research does not support. Remove them, "
            "soften them into qualitative statements, or replace them with something "
            "the research does support."
        ]
        for claim in self.unsupported:
            lines.append(f'- POST {claim.post_index}: "{claim.claim}" - {claim.reasoning}')
        return "\n".join(lines)
