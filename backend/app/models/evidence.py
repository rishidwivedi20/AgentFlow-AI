"""Retrieved evidence: chunks of real source documents, ranked by relevance.

Extends the guarantee ``app.models.research`` makes about sources. A chunk is
a verbatim slice of text a search provider actually returned, cut in code and
never written by a model. Chunks reference their document by index into the
run's source list, exactly as facts do, so nothing here can carry a fabricated
URL.

Retrieved text is *evidence*, not instruction. Nothing in this module treats it
as trusted; the prompt layer is responsible for delimiting it as untrusted data
before it reaches a model.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RetrievalStatus(StrEnum):
    """How much the retrieval step actually established.

    Kept distinct from success/failure: retrieval can complete perfectly and
    still find nothing convincing, and the downstream agents need to tell those
    apart. A low similarity score means "these sources do not discuss this",
    never "this claim is false".
    """

    RETRIEVED = "retrieved"
    """Chunks were found above the similarity floor."""

    WEAK_EVIDENCE = "weak_evidence"
    """Chunks exist, but none cleared the floor. Usable with caution."""

    NO_EVIDENCE = "no_evidence"
    """Nothing to retrieve from: no documents, or no usable text in them."""

    NOT_ATTEMPTED = "not_attempted"
    """Retrieval was disabled or skipped. Not a statement about the sources."""


class SourceDocument(BaseModel):
    """Full text of one source, ready to be chunked.

    Built from an HTTP response body, never from model output. ``source_index``
    is the position of the corresponding :class:`~app.models.research.Source`
    in the run's source list.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    source_index: int = Field(ge=0, description="Position in the run's source list.")
    text: str = Field(description="Extracted page text as the provider returned it.")


class DocumentChunk(BaseModel):
    """One slice of a source document, before it has been scored."""

    source_index: int = Field(ge=0, description="Which source this text came from.")
    position: int = Field(ge=0, description="Zero-based position within that document.")
    text: str = Field(min_length=1, description="Verbatim slice of the document.")


class EvidenceChunk(DocumentChunk):
    """A chunk that survived retrieval, with the score that selected it."""

    similarity: float = Field(
        ge=-1.0,
        le=1.0,
        description="Cosine similarity between this chunk and the retrieval query.",
    )


class EvidenceSet(BaseModel):
    """Everything the retrieval step produced for one run.

    Carries the counts and warnings needed to explain *why* the evidence looks
    the way it does, so a thin result reads as a measured outcome rather than a
    silent failure.
    """

    query: str = Field(description="The text chunks were ranked against.")
    status: RetrievalStatus = Field(description="How much this retrieval established.")
    chunks: list[EvidenceChunk] = Field(
        default_factory=list,
        description="Selected chunks, most relevant first.",
    )
    documents_indexed: int = Field(
        default=0, ge=0, description="Source documents that yielded at least one chunk."
    )
    chunks_indexed: int = Field(
        default=0, ge=0, description="Total chunks embedded before selection."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal observations about the quality or breadth of the evidence.",
    )

    @property
    def is_usable(self) -> bool:
        """Whether any chunk was selected, at any confidence."""
        return bool(self.chunks)

    @property
    def source_indexes(self) -> list[int]:
        """Distinct sources represented in the selection, in rank order."""
        seen: list[int] = []
        for chunk in self.chunks:
            if chunk.source_index not in seen:
                seen.append(chunk.source_index)
        return seen

    @classmethod
    def not_attempted(cls, query: str, reason: str) -> EvidenceSet:
        """An empty set that says retrieval did not run, and why."""
        return cls(query=query, status=RetrievalStatus.NOT_ATTEMPTED, warnings=[reason])
