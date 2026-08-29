"""Research output: verified sources and the facts drawn from them.

The central guarantee of this module is that a ``Source`` can only originate
from a real search result. URLs never appear in an LLM response schema, so the
Researcher is structurally incapable of inventing one - it can only reference
sources by index into a list the search provider produced.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class StatementKind(StrEnum):
    """Whether a research statement is checkable or someone's view.

    The Writer is told to treat these differently, and the Fact Checker only
    holds ``FACT`` statements to a verification standard.
    """

    FACT = "fact"
    """A verifiable claim: a number, event, date or documented behaviour."""

    OPINION = "opinion"
    """A prediction, judgement or recommendation attributed to a source."""


class Source(BaseModel):
    """One real search result.

    Never constructed from LLM output. Built only by a research provider
    adapter from an actual HTTP response.
    """

    title: str = Field(description="Page title as reported by the search provider.")
    url: HttpUrl = Field(description="Canonical URL of the source.")
    snippet: str = Field(description="Extract the provider returned for this result.")

    @property
    def domain(self) -> str:
        """Host portion of the URL, for display and de-duplication."""
        return self.url.host or ""


class ResearchFact(BaseModel):
    """A single statement the Researcher drew from the supplied sources.

    LLM-facing: every field is required and none are nullable, per the
    structured-output rules in ``app.models.review``.

    ``source_index`` is the mechanism that prevents fabricated citations. The
    model chooses a position in a list it was given; it never emits a URL.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    statement: str = Field(
        min_length=1,
        description="The fact or opinion, stated plainly in one or two sentences.",
    )
    kind: StatementKind = Field(description="Whether this is a verifiable fact or an opinion.")
    source_index: int = Field(
        ge=0,
        description=(
            "Zero-based position of the supporting source in the provided list. "
            "Must refer to a source that was actually supplied."
        ),
    )


class ResearchOutput(BaseModel):
    """What the Researcher agent returns from the LLM.

    Deliberately holds no URLs - only indexes into the sources it was handed.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str = Field(
        min_length=1,
        description="Neutral overview of what the sources collectively say about the topic.",
    )
    facts: list[ResearchFact] = Field(
        description="Statements extracted from the sources, each tied to one of them."
    )


class ResearchResult(BaseModel):
    """The Researcher's output joined back to the real sources it cited.

    Assembled in code, not by the model: this is where indexes become URLs.
    Facts referencing an out-of-range index are dropped rather than trusted.
    """

    summary: str
    facts: list[ResearchFact]
    sources: list[Source] = Field(description="Sources actually returned by the search provider.")

    @property
    def verifiable_facts(self) -> list[ResearchFact]:
        """Only the statements held to a verification standard."""
        return [fact for fact in self.facts if fact.kind is StatementKind.FACT]

    def source_for(self, fact: ResearchFact) -> Source | None:
        """Resolve a fact's citation, or None if the index is out of range."""
        if 0 <= fact.source_index < len(self.sources):
            return self.sources[fact.source_index]
        return None

    def as_writer_context(self) -> str:
        """Render the research as the grounding text handed to the Writer.

        This is the string that fills ``research_context`` in workflow state -
        the seam the Writer prompt has accepted since Phase 2.
        """
        lines = [f"SUMMARY: {self.summary}", ""]
        for fact in self.facts:
            source = self.source_for(fact)
            attribution = f" [source: {source.domain}]" if source else ""
            lines.append(f"- ({fact.kind.value}) {fact.statement}{attribution}")
        return "\n".join(lines)
