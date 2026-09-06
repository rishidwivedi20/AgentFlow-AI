"""Per-run retrieval over the text of the sources a search returned.

Deliberately not a vector database. The corpus is rebuilt on every run and is
roughly 100-150 chunks, so an exact cosine scan is both simpler and more
accurate than an approximate index - and it has nothing to keep in sync, no
persistence, and no staleness to reason about. The scan is measured in the
tests; if a future phase indexes a standing corpus, that is when a real store
earns its dependency.

Four guardrails shape what comes back, and each answers a different failure:

* a **similarity floor**, so unrelated text is not presented as evidence;
* a **per-source cap**, so one long page cannot crowd out every other source;
* **exact-duplicate removal**, because navigation furniture repeats across
  pages of the same site and would otherwise look like corroboration;
* a **per-run corpus cap**, because embedding cost is charged in requests, and
  an unbounded corpus means an unbounded number of them.

A low score means these sources do not discuss the query. It is never evidence
that a claim is false, and nothing here reports it as such.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError, EmbeddingProviderError
from app.models.evidence import (
    DocumentChunk,
    EvidenceChunk,
    EvidenceSet,
    RetrievalStatus,
    SourceDocument,
)
from app.services.chunking import DEFAULT_MIN_CHUNK_CHARS, chunk_document
from app.services.embedding.base import EmbeddingProvider, Vector
from app.services.embedding.gemini import GeminiEmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_TOP_K: Final = 8
"""Eight chunks of ~300 tokens is ~2,400 tokens of evidence, which leaves room
for a prompt and its output inside Groq's free-tier 8,000 tokens-per-minute
ceiling - the tightest budget of any configured provider."""

DEFAULT_MIN_SIMILARITY: Final = 0.5
DEFAULT_MAX_CHUNKS_PER_SOURCE: Final = 3

DEFAULT_MAX_CHUNKS_PER_RUN: Final = 90
"""Bounds how much corpus is embedded, and with it what a run costs.

The free tier allows 100 embedded passages per minute
(``EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier``), counting
items rather than HTTP calls - every text in a batch is charged separately,
which is also why the API refuses batches above 100.

That leaves 100 per minute, minus one for the query, minus headroom for a
second run in the same minute. Five uncapped pages produce 400-500 chunks."""


@dataclass(frozen=True, slots=True)
class _ScoredChunk:
    """A chunk paired with its score, before selection rules are applied."""

    chunk: DocumentChunk
    similarity: float


def cosine_similarity(left: Vector, right: Vector) -> float:
    """True cosine similarity, normalising both operands.

    Full division rather than a dot product because provider vectors are not
    reliably unit length: Gemini returns unit vectors at its native 3072
    dimensions but not at the truncated widths this project uses (measured 0.58
    at 768). A dot product would scale every score by a constant and quietly
    push everything below the floor.
    """
    if len(left) != len(right):
        raise ValueError(f"vector width mismatch: {len(left)} vs {len(right)}")
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_sq += a * a
        right_sq += b * b
    if left_sq <= 0.0 or right_sq <= 0.0:
        # A zero vector has no direction, so it has no similarity to anything.
        return 0.0
    return dot / math.sqrt(left_sq * right_sq)


class RetrievalService:
    """Chunks source documents, embeds them, and selects the best evidence."""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        *,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        max_chunks_per_source: int = DEFAULT_MAX_CHUNKS_PER_SOURCE,
        max_chunks_per_run: int = DEFAULT_MAX_CHUNKS_PER_RUN,
        min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    ) -> None:
        self._embeddings = embeddings
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._max_chunks_per_source = max_chunks_per_source
        self._max_chunks_per_run = max_chunks_per_run
        self._min_chunk_chars = min_chunk_chars

    @property
    def provider_name(self) -> str:
        return self._embeddings.name

    @property
    def model(self) -> str:
        return self._embeddings.model

    async def retrieve(self, *, query: str, documents: Sequence[SourceDocument]) -> EvidenceSet:
        """Return the chunks of ``documents`` most relevant to ``query``.

        Raises ``EmbeddingProviderError`` or ``ConfigurationError`` when the
        embedding provider fails: the caller records that as a real failure
        rather than presenting an empty result as an absence of evidence.
        """
        chunks, duplicates = self._index(documents)
        chunks, dropped = self._cap(chunks)
        if not chunks:
            return EvidenceSet(
                query=query,
                status=RetrievalStatus.NO_EVIDENCE,
                warnings=["No usable text was extracted from the sources."],
            )

        chunk_vectors = await self._embeddings.embed_documents([c.text for c in chunks])
        if len(chunk_vectors) != len(chunks):
            raise EmbeddingProviderError(
                f"Embedded {len(chunk_vectors)} vectors for {len(chunks)} chunks.",
                details={"provider": self._embeddings.name},
            )
        query_vector = await self._embeddings.embed_query(query)

        ranked = sorted(
            (
                _ScoredChunk(chunk=chunk, similarity=cosine_similarity(vector, query_vector))
                for chunk, vector in zip(chunks, chunk_vectors, strict=True)
            ),
            key=lambda scored: scored.similarity,
            reverse=True,
        )

        warnings: list[str] = []
        if duplicates:
            warnings.append(
                f"Removed {duplicates} chunk(s) repeated verbatim across the sources."
            )
        if dropped:
            warnings.append(
                f"Only the first {len(chunks)} passages of the sources were searched; "
                f"{dropped} more were left out to stay inside the embedding budget."
            )

        above_floor = [s for s in ranked if s.similarity >= self._min_similarity]
        if above_floor:
            pool = above_floor
            status = RetrievalStatus.RETRIEVED
            if len(above_floor) < self._top_k:
                warnings.append(
                    f"Only {len(above_floor)} of {len(ranked)} chunks scored at or above "
                    f"the {self._min_similarity:.2f} similarity floor."
                )
        else:
            # Return the best available anyway, flagged. Withholding it would
            # tell the agents nothing, and thin evidence they can see is more
            # useful than an empty set they cannot interpret.
            pool = ranked
            status = RetrievalStatus.WEAK_EVIDENCE
            warnings.append(
                f"No chunk reached the {self._min_similarity:.2f} similarity floor; the "
                f"best scored {ranked[0].similarity:.2f}. These sources may not discuss "
                "the topic - which is not evidence that anything about it is false."
            )

        selected = self._select(pool)
        evidence = EvidenceSet(
            query=query,
            status=status,
            chunks=[
                EvidenceChunk(
                    source_index=scored.chunk.source_index,
                    position=scored.chunk.position,
                    text=scored.chunk.text,
                    similarity=scored.similarity,
                )
                for scored in selected
            ],
            documents_indexed=len({chunk.source_index for chunk in chunks}),
            chunks_indexed=len(chunks),
            warnings=warnings,
        )
        logger.info(
            "Retrieval completed",
            extra={
                "provider": self._embeddings.name,
                "status": status.value,
                "chunks_indexed": evidence.chunks_indexed,
                "chunks_selected": len(evidence.chunks),
                "sources_represented": len(evidence.source_indexes),
            },
        )
        return evidence

    def _index(self, documents: Sequence[SourceDocument]) -> tuple[list[DocumentChunk], int]:
        """Chunk every document, dropping text that repeats verbatim.

        Boilerplate - cookie banners, nav menus, footers - recurs across pages
        of one site. Left in, it occupies retrieval slots and, worse, makes the
        same sentence appear to be corroborated by several sources.
        """
        chunks: list[DocumentChunk] = []
        seen: set[str] = set()
        duplicates = 0
        for document in documents:
            for chunk in chunk_document(
                document.text,
                source_index=document.source_index,
                chunk_size=self._chunk_size,
                overlap=self._chunk_overlap,
                min_chunk_chars=self._min_chunk_chars,
            ):
                key = chunk.text.strip().casefold()
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                chunks.append(chunk)
        return chunks, duplicates

    def _cap(self, chunks: list[DocumentChunk]) -> tuple[list[DocumentChunk], int]:
        """Bound how much corpus is embedded, taking from each source in turn.

        The cap exists because embedding cost is measured in requests, not
        tokens, and an unbounded corpus means an unbounded number of them.

        Round-robin rather than truncating the tail: taking the first N chunks
        outright would embed one long page in full and none of the last source
        at all, and a passage that is never embedded can never be selected, so
        that source could never appear in the evidence at all.
        """
        if len(chunks) <= self._max_chunks_per_run:
            return chunks, 0

        by_source: dict[int, list[DocumentChunk]] = {}
        for chunk in chunks:
            by_source.setdefault(chunk.source_index, []).append(chunk)

        kept: list[DocumentChunk] = []
        for position in range(max(len(group) for group in by_source.values())):
            for group in by_source.values():
                if position < len(group):
                    kept.append(group[position])
                    if len(kept) >= self._max_chunks_per_run:
                        # Restore reading order so positions stay meaningful.
                        kept.sort(key=lambda c: (c.source_index, c.position))
                        return kept, len(chunks) - len(kept)

        kept.sort(key=lambda c: (c.source_index, c.position))
        return kept, len(chunks) - len(kept)

    def _select(self, ranked: Sequence[_ScoredChunk]) -> list[_ScoredChunk]:
        """Take the top ``top_k``, preferring breadth across sources.

        The per-source cap is a preference, not a quota: if honouring it would
        leave the budget unfilled - one source, or four - the deferred chunks
        are backfilled in score order. Diversity is worth having when it is
        available, not worth spending unused budget on.
        """
        selected: list[_ScoredChunk] = []
        deferred: list[_ScoredChunk] = []
        per_source: Counter[int] = Counter()

        for scored in ranked:
            if len(selected) >= self._top_k:
                break
            if per_source[scored.chunk.source_index] >= self._max_chunks_per_source:
                deferred.append(scored)
                continue
            selected.append(scored)
            per_source[scored.chunk.source_index] += 1

        for scored in deferred:
            if len(selected) >= self._top_k:
                break
            selected.append(scored)

        selected.sort(key=lambda scored: scored.similarity, reverse=True)
        return selected


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Instantiate the embedding adapter.

    Always Gemini, whatever ``LLM_PROVIDER`` says: Groq exposes no embeddings
    API, so a Groq run still needs ``GEMINI_API_KEY`` for retrieval.
    """
    if not settings.embeddings_configured:
        raise ConfigurationError(
            "No API key configured for embeddings. Set GEMINI_API_KEY in "
            "backend/.env. It is required for retrieval even when LLM_PROVIDER "
            "is groq, because Groq has no embeddings API.",
            details={"provider": "gemini"},
        )
    return GeminiEmbeddingProvider(
        api_key=settings.gemini_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )


def build_optional_retrieval_service(settings: Settings) -> RetrievalService | None:
    """The retrieval service, or None when retrieval cannot or should not run.

    Never raises. Retrieval is an enhancement, not a prerequisite: a run with
    no embedding key still searches, still grounds the Writer in snippets, and
    still fact checks. Refusing the request over a missing optional key would
    be a worse outcome than the degraded run.
    """
    if not settings.retrieval_available:
        logger.info(
            "Retrieval unavailable; the run will use search snippets alone",
            extra={
                "enabled": settings.retrieval_enabled,
                "embeddings_configured": settings.embeddings_configured,
            },
        )
        return None
    return build_retrieval_service(settings)


def build_retrieval_service(settings: Settings | None = None) -> RetrievalService:
    """Build the service from configuration.

    Not cached, matching ``build_research_service``: tests construct one with a
    stub provider, and the graph receives its services by injection.
    """
    resolved = settings or get_settings()
    return RetrievalService(
        build_embedding_provider(resolved),
        chunk_size=resolved.retrieval_chunk_size,
        chunk_overlap=resolved.retrieval_chunk_overlap,
        top_k=resolved.retrieval_top_k,
        min_similarity=resolved.retrieval_min_similarity,
        max_chunks_per_source=resolved.retrieval_max_chunks_per_source,
        max_chunks_per_run=resolved.retrieval_max_chunks_per_run,
    )
