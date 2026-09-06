"""Chunking: size guarantees, boundary preference and whitespace handling.

Pure functions, so every case here is exact rather than approximate. The size
invariant matters most: it is what lets the retrieval layer reason about the
token budget before spending a single embedding call.
"""

from __future__ import annotations

import pytest

from app.services.chunking import (
    chunk_document,
    chunk_text,
    normalise,
)

PARAGRAPH = (
    "Retrieval augmented generation grounds a model in text it was given "
    "rather than in what it memorised during training. "
)


def paragraphs(count: int, *, sentences: int = 3) -> str:
    return "\n\n".join((PARAGRAPH * sentences).strip() for _ in range(count))


# --- normalisation ---------------------------------------------------------


def test_normalise_collapses_blank_line_runs():
    assert normalise("one\n\n\n\n\ntwo") == "one\n\ntwo"


def test_normalise_collapses_non_breaking_spaces():
    """HTML extraction leaves these behind; they cost budget and mean nothing."""
    nbsp = chr(0xA0)
    assert normalise(f"one{nbsp} {nbsp} two") == "one two"


def test_normalise_removes_markdown_images_entirely():
    """An image carries nothing an embedding can use."""
    assert normalise("before ![Linkedin](https://cdn.example.com/x.png) after") == (
        "before after"
    )


def test_normalise_keeps_link_text_and_drops_the_target():
    assert normalise("read [the report](https://example.com/r) now") == (
        "read the report now"
    )


def test_normalise_removes_bare_urls():
    assert normalise("see https://example.com/page for more") == "see for more"
    assert normalise("visit www.example.com today") == "visit today"


def test_a_navigation_block_collapses_to_almost_nothing():
    """Measured on a real page: a social-link block outranked every substantive
    passage on the same topic, because boilerplate on a topical page embeds
    close to the topic. The floor cannot catch that; removing the markup can."""
    nav = "[![Linkedin](https://storage.googleapis.com/a.png)](https://linkedin.com/x) " * 8
    assert len(normalise(nav)) < 80


def test_normalise_strips_control_characters():
    assert normalise("clean\x00\x07text") == "cleantext"


def test_normalise_preserves_paragraph_structure():
    """Paragraph breaks are the primary split point, so they must survive."""
    assert normalise("first para\n\nsecond para") == "first para\n\nsecond para"


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t\n"])
def test_blank_documents_produce_no_chunks(text):
    assert chunk_text(text) == []


# --- size and overlap invariants -------------------------------------------


def test_no_chunk_exceeds_the_configured_size():
    chunks = chunk_text(paragraphs(40), chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_no_chunk_exceeds_the_size_when_a_single_sentence_is_oversized():
    """The hard-window fallback must respect the budget too."""
    monster = "word " * 4000
    chunks = chunk_text(monster, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_consecutive_chunks_share_overlapping_text():
    """A statement spanning a boundary must be retrievable from either side."""
    chunks = chunk_text(paragraphs(30), chunk_size=400, overlap=100)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        carried = current[:100].strip()
        assert carried, "expected leading overlap text"
        assert carried.split()[0] in previous


def test_zero_overlap_produces_disjoint_chunks():
    chunks = chunk_text(paragraphs(20), chunk_size=400, overlap=0)
    assert len(chunks) > 1
    rejoined = "".join(chunks)
    assert len(rejoined) <= len(normalise(paragraphs(20)))


def test_text_shorter_than_the_budget_is_one_untouched_chunk():
    body = "A short page with a single claim in it."
    assert chunk_text(body, chunk_size=500, overlap=50) == [body]


def test_a_lone_short_chunk_survives_the_minimum():
    """A brief page is thin evidence, not noise - it must not be discarded."""
    assert chunk_text("Short.", chunk_size=500, overlap=50, min_chunk_chars=100) == ["Short."]


def test_trailing_fragments_are_pruned_once_there_is_real_content():
    body = paragraphs(6) + "\n\nNav"
    chunks = chunk_text(body, chunk_size=300, overlap=50, min_chunk_chars=100)
    assert not any(chunk.strip().endswith("Nav") and len(chunk) < 100 for chunk in chunks)


@pytest.mark.parametrize("overlap", [300, 400])
def test_overlap_at_or_above_chunk_size_is_rejected(overlap):
    """Otherwise no chunk would have room for new content."""
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        chunk_text(paragraphs(5), chunk_size=300, overlap=overlap)


def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError, match="not be negative"):
        chunk_text(paragraphs(5), chunk_size=300, overlap=-1)


# --- boundary preference ---------------------------------------------------


def test_paragraphs_that_fit_are_not_split_mid_paragraph():
    """Splitting a claim in half leaves two halves that support nothing."""
    body = "\n\n".join(f"Paragraph {i} makes exactly one point." for i in range(8))
    chunks = chunk_text(body, chunk_size=200, overlap=0)
    for chunk in chunks:
        for line in chunk.split("\n\n"):
            assert line.startswith("Paragraph "), chunk
            assert line.endswith("point."), chunk


def test_an_oversized_paragraph_falls_back_to_sentence_splits():
    body = " ".join(f"Sentence number {i} carries its own small fact." for i in range(40))
    chunks = chunk_text(body, chunk_size=200, overlap=0)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    # Sentence boundaries are preferred over arbitrary cuts.
    assert sum(chunk.rstrip().endswith(".") for chunk in chunks) >= len(chunks) - 1


# --- document tagging ------------------------------------------------------


def test_chunk_document_tags_source_and_ascending_positions():
    chunks = chunk_document(paragraphs(20), source_index=3, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert {chunk.source_index for chunk in chunks} == {3}
    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))


def test_chunk_document_on_blank_text_yields_nothing():
    assert chunk_document("   \n\n  ", source_index=0) == []
