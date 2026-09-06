"""Retrieval: ranking, the three guardrails, and the scan's cost.

No network and no API key. ``StubEmbeddingProvider`` turns text into a count
vector over a six-word vocabulary, so relevance is exactly predictable: a chunk
sharing words with the query outranks one that does not.
"""

from __future__ import annotations

import math
import time

import pytest

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, EmbeddingProviderError
from app.models.evidence import EvidenceSet, RetrievalStatus, SourceDocument
from app.services.retrieval_service import (
    RetrievalService,
    build_embedding_provider,
    build_retrieval_service,
    cosine_similarity,
)
from tests.conftest import StubEmbeddingProvider, make_retrieval


def document(index: int, *terms: str, repeat: int = 4) -> SourceDocument:
    """A document built from vocabulary terms, so its score is predictable.

    The leading label keeps documents textually distinct even when they carry
    the same terms - otherwise the duplicate guardrail would collapse them and
    the test would be measuring deduplication rather than ranking. Words
    outside the stub vocabulary do not affect the vector.
    """
    body = " ".join(" ".join(terms) for _ in range(repeat))
    return SourceDocument(source_index=index, text=f"Document number {index}. {body}")


# --- cosine similarity -----------------------------------------------------


def test_identical_vectors_score_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_magnitude_does_not_change_the_score():
    """The finding that drove this: Gemini's truncated vectors are not unit
    length (measured 0.58 at 768 dims), so a raw dot product would understate
    every similarity by a constant and push everything below the floor."""
    plain = cosine_similarity([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    scaled = cosine_similarity([0.1, 0.2, 0.3], [30.0, 20.0, 10.0])
    assert plain == pytest.approx(scaled)


def test_a_zero_vector_has_no_similarity_to_anything():
    """It has no direction, so any answer but zero would be invented."""
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_mismatched_widths_are_rejected_rather_than_truncated():
    with pytest.raises(ValueError, match="width mismatch"):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


# --- ranking ---------------------------------------------------------------


async def test_chunks_are_returned_most_relevant_first():
    service = make_retrieval(top_k=3, max_chunks_per_source=3)
    evidence = await service.retrieve(
        query="alpha beta",
        documents=[
            document(0, "zeta zeta zeta"),
            document(1, "alpha beta alpha beta"),
            document(2, "alpha gamma"),
        ],
    )
    assert evidence.status is RetrievalStatus.RETRIEVED
    scores = [chunk.similarity for chunk in evidence.chunks]
    assert scores == sorted(scores, reverse=True)
    assert evidence.chunks[0].source_index == 1


async def test_ranking_is_unaffected_by_vector_magnitude():
    """Same corpus, unnormalised document vectors: the order must not move."""
    docs = [document(0, "zeta"), document(1, "alpha beta"), document(2, "alpha gamma")]
    unit = await make_retrieval(StubEmbeddingProvider()).retrieve(
        query="alpha beta", documents=docs
    )
    scaled = await make_retrieval(StubEmbeddingProvider(scale=0.58)).retrieve(
        query="alpha beta", documents=docs
    )
    assert [c.source_index for c in unit.chunks] == [c.source_index for c in scaled.chunks]
    assert [c.similarity for c in unit.chunks] == pytest.approx(
        [c.similarity for c in scaled.chunks]
    )


async def test_top_k_bounds_the_evidence_budget():
    service = make_retrieval(top_k=2, max_chunks_per_source=2)
    evidence = await service.retrieve(
        query="alpha",
        documents=[document(i, "alpha beta") for i in range(6)],
    )
    assert len(evidence.chunks) == 2
    assert evidence.chunks_indexed == 6


# --- guardrail: per-source diversity cap ------------------------------------


async def test_one_long_source_cannot_fill_the_whole_budget():
    """Without the cap, the 40-chunk page would take every slot."""
    long_page = SourceDocument(
        source_index=0,
        text="\n\n".join(f"alpha beta paragraph number {i} of many" for i in range(40)),
    )
    service = make_retrieval(top_k=4, max_chunks_per_source=2, chunk_size=100, chunk_overlap=10)
    evidence = await service.retrieve(
        query="alpha beta",
        documents=[long_page, document(1, "alpha beta"), document(2, "alpha beta")],
    )
    assert sum(chunk.source_index == 0 for chunk in evidence.chunks) == 2
    assert set(evidence.source_indexes) == {0, 1, 2}


async def test_the_cap_is_relaxed_when_diversity_is_unavailable():
    """A preference, not a quota: one source must not leave the budget unfilled."""
    long_page = SourceDocument(
        source_index=0,
        text="\n\n".join(f"alpha beta paragraph number {i} of many" for i in range(40)),
    )
    service = make_retrieval(top_k=4, max_chunks_per_source=2, chunk_size=100, chunk_overlap=10)
    evidence = await service.retrieve(query="alpha beta", documents=[long_page])
    assert len(evidence.chunks) == 4
    assert evidence.source_indexes == [0]


# --- guardrail: similarity floor -------------------------------------------


async def test_chunks_below_the_floor_are_excluded():
    service = make_retrieval(top_k=5, min_similarity=0.5, max_chunks_per_source=5)
    evidence = await service.retrieve(
        query="alpha beta",
        documents=[document(0, "alpha beta"), document(1, "zeta zeta")],
    )
    assert evidence.status is RetrievalStatus.RETRIEVED
    assert evidence.source_indexes == [0]
    assert all(chunk.similarity >= 0.5 for chunk in evidence.chunks)


async def test_a_thin_result_above_the_floor_is_reported_as_such():
    service = make_retrieval(top_k=8, min_similarity=0.5, max_chunks_per_source=8)
    evidence = await service.retrieve(
        query="alpha beta",
        documents=[document(0, "alpha beta"), document(1, "zeta")],
    )
    assert any("similarity floor" in warning for warning in evidence.warnings)


async def test_nothing_above_the_floor_yields_weak_evidence_not_an_empty_set():
    """Thin evidence the agents can see beats an empty set they cannot read."""
    service = make_retrieval(top_k=3, min_similarity=0.9, max_chunks_per_source=3)
    evidence = await service.retrieve(
        query="alpha beta",
        documents=[document(0, "gamma delta"), document(1, "zeta epsilon")],
    )
    assert evidence.status is RetrievalStatus.WEAK_EVIDENCE
    assert evidence.chunks


async def test_weak_evidence_never_claims_anything_is_false():
    """A low score means these sources do not discuss it - nothing more."""
    service = make_retrieval(min_similarity=0.99)
    evidence = await service.retrieve(query="alpha", documents=[document(0, "zeta")])
    warning = " ".join(evidence.warnings).casefold()
    assert "may not discuss" in warning
    assert "not evidence that anything about it is false" in warning


# --- guardrail: duplicate removal ------------------------------------------


async def test_text_repeated_across_sources_is_indexed_once():
    """Site furniture would otherwise look like independent corroboration."""
    boilerplate = "alpha beta accept all cookies and continue browsing this site"
    service = make_retrieval(top_k=5, max_chunks_per_source=5)
    evidence = await service.retrieve(
        query="alpha beta",
        documents=[
            SourceDocument(source_index=0, text=boilerplate),
            SourceDocument(source_index=1, text=boilerplate),
            SourceDocument(source_index=2, text=boilerplate.upper()),
        ],
    )
    assert evidence.chunks_indexed == 1
    assert any("repeated verbatim" in warning for warning in evidence.warnings)


# --- guardrail: per-run corpus cap ------------------------------------------


def long_page(index: int, paragraphs: int = 60) -> SourceDocument:
    return SourceDocument(
        source_index=index,
        text="\n\n".join(
            f"alpha beta source {index} paragraph {position} of many"
            for position in range(paragraphs)
        ),
    )


async def test_the_corpus_is_capped_before_anything_is_embedded():
    """Embedding is charged in requests, so an unbounded corpus is an unbounded
    bill - and on the free tier, a rate limit rather than a bill."""
    provider = StubEmbeddingProvider()
    service = make_retrieval(
        provider, chunk_size=100, chunk_overlap=10, max_chunks_per_run=20, top_k=4
    )
    evidence = await service.retrieve(
        query="alpha beta", documents=[long_page(0), long_page(1)]
    )
    assert evidence.chunks_indexed == 20
    assert any("embedding budget" in warning for warning in evidence.warnings)


async def test_the_cap_takes_from_every_source_not_just_the_first():
    """A passage that is never embedded can never be selected, so truncating
    the tail would make the last source unrepresentable in the evidence."""
    service = make_retrieval(chunk_size=100, chunk_overlap=10, max_chunks_per_run=20)
    evidence = await service.retrieve(
        query="alpha beta",
        documents=[long_page(0), long_page(1), long_page(2)],
    )
    # Every source reached the index. Which of them wins a selection slot is a
    # separate question, decided later by score and the per-source cap.
    assert evidence.documents_indexed == 3
    assert evidence.chunks_indexed == 20


async def test_a_corpus_inside_the_cap_is_untouched():
    provider = StubEmbeddingProvider()
    service = make_retrieval(provider, max_chunks_per_run=500)
    evidence = await service.retrieve(
        query="alpha beta", documents=[document(0, "alpha beta"), document(1, "alpha")]
    )
    assert not any("embedding budget" in warning for warning in evidence.warnings)


# --- empty and failing inputs ----------------------------------------------


async def test_no_documents_is_no_evidence_and_costs_no_embedding_call():
    provider = StubEmbeddingProvider()
    evidence = await make_retrieval(provider).retrieve(query="alpha", documents=[])
    assert evidence.status is RetrievalStatus.NO_EVIDENCE
    assert not evidence.chunks
    assert provider.document_calls == 0


async def test_documents_with_no_usable_text_are_no_evidence():
    evidence = await make_retrieval().retrieve(
        query="alpha",
        documents=[SourceDocument(source_index=0, text="   \n\n  ")],
    )
    assert evidence.status is RetrievalStatus.NO_EVIDENCE


async def test_an_embedding_failure_propagates_rather_than_reading_as_no_evidence():
    """The caller must record a failure, not report that the sources said nothing."""
    service = make_retrieval(StubEmbeddingProvider(fail=True))
    with pytest.raises(RuntimeError, match="stub embedding failure"):
        await service.retrieve(query="alpha", documents=[document(0, "alpha")])


async def test_a_short_vector_batch_is_rejected_rather_than_mis_paired():
    """Vectors are paired with chunks by position, so a silent drop would
    attribute evidence to the wrong source."""
    service = make_retrieval(StubEmbeddingProvider(drop_last=True))
    with pytest.raises(EmbeddingProviderError, match="vectors for"):
        await service.retrieve(
            query="alpha",
            documents=[document(0, "alpha"), document(1, "beta")],
        )


# --- evidence set ----------------------------------------------------------


def test_not_attempted_records_the_reason():
    evidence = EvidenceSet.not_attempted("alpha", "Retrieval disabled by configuration.")
    assert evidence.status is RetrievalStatus.NOT_ATTEMPTED
    assert not evidence.is_usable
    assert evidence.warnings == ["Retrieval disabled by configuration."]


async def test_source_indexes_are_distinct_and_in_rank_order():
    service = make_retrieval(top_k=4, max_chunks_per_source=2)
    evidence = await service.retrieve(
        query="alpha beta",
        documents=[document(0, "zeta"), document(1, "alpha beta"), document(2, "alpha")],
    )
    assert evidence.source_indexes == list(dict.fromkeys(evidence.source_indexes))
    assert evidence.source_indexes[0] == evidence.chunks[0].source_index


# --- the "no vector database" claim ----------------------------------------


async def test_an_exact_scan_is_fast_enough_at_realistic_corpus_size():
    """The module claims a vector store would be over-engineering here. This is
    the measurement behind that: a full 768-dimension scan over a corpus larger
    than any single run produces, well inside a millisecond budget that makes
    an approximate index pointless."""
    query = [0.01 * i for i in range(768)]
    corpus = [[0.01 * ((i + j) % 97) for i in range(768)] for j in range(300)]

    started = time.perf_counter()
    scores = [cosine_similarity(vector, query) for vector in corpus]
    elapsed = time.perf_counter() - started

    assert len(scores) == 300
    assert all(math.isfinite(score) for score in scores)
    assert elapsed < 1.0, f"exact scan took {elapsed:.3f}s"


# --- construction from settings --------------------------------------------


def test_embeddings_need_gemini_key_even_when_generation_uses_groq():
    """Groq has no embeddings API, so retrieval depends on Gemini's key."""
    settings = Settings(llm_provider="groq", groq_api_key="groq-key", gemini_api_key="")
    assert settings.llm_configured is True
    assert settings.embeddings_configured is False
    assert settings.retrieval_available is False
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        build_embedding_provider(settings)


def test_a_groq_run_with_a_gemini_key_can_still_retrieve():
    settings = Settings(llm_provider="groq", groq_api_key="groq-key", gemini_api_key="gemini-key")
    assert settings.retrieval_available is True
    assert isinstance(build_retrieval_service(settings), RetrievalService)


def test_disabling_retrieval_overrides_a_usable_key():
    settings = Settings(gemini_api_key="gemini-key", retrieval_enabled=False)
    assert settings.embeddings_configured is True
    assert settings.retrieval_available is False


def test_overlap_at_or_above_chunk_size_is_rejected_at_configuration_time():
    """Caught at startup rather than on the first run that tries to chunk."""
    with pytest.raises(ValueError, match="retrieval_chunk_overlap"):
        Settings(retrieval_chunk_size=500, retrieval_chunk_overlap=500)


def test_settings_defaults_match_the_documented_budget():
    settings = Settings()
    assert settings.retrieval_top_k == 8
    assert settings.retrieval_chunk_size == 1200
    assert settings.embedding_dimensions == 768
