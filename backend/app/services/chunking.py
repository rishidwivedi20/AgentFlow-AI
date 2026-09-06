"""Splitting source documents into retrievable chunks.

Pure functions, no I/O and no model calls, so the behaviour that decides what
evidence can be retrieved at all is directly testable.

Two properties matter more than clever segmentation:

* **Bounded size.** Every chunk is at most ``chunk_size`` characters, which is
  what lets the caller reason about the token budget before embedding anything.
* **Preserved boundaries.** Splits are attempted at paragraph breaks first, then
  sentence breaks, and only fall back to a hard character window when a single
  sentence is longer than the budget. A claim cut in half is a claim neither
  half supports.
"""

from __future__ import annotations

import re
from typing import Final

from app.models.evidence import DocumentChunk

# Characters carried from the end of one chunk into the start of the next, so a
# statement spanning a boundary is retrievable from either side.
DEFAULT_CHUNK_SIZE: Final = 1200
DEFAULT_CHUNK_OVERLAP: Final = 150

# Below this, a chunk is almost always page furniture - a nav label, a cookie
# line, a lone heading - rather than something that could support a claim.
DEFAULT_MIN_CHUNK_CHARS: Final = 80

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
# Extracted page text arrives as markdown. Images carry nothing an embedding
# can use; links carry their text but not their target.
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BARE_URL = re.compile(r"\b(?:https?://|www\.)\S+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
# Whitespace that is not a newline. Written as a negated class rather than a
# literal set so it also collapses the non-breaking spaces and thin spaces
# that HTML extraction leaves behind.
_INLINE_WHITESPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")
# Everything unprintable except newline and tab. Extracted page text routinely
# carries stray control bytes that survive into embeddings as noise.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalise(text: str) -> str:
    """Strip the markup and whitespace noise typical of extracted page text.

    Blank-line runs and non-breaking spaces consume chunk budget without
    carrying meaning. URLs are worse: a block of social links embeds close to
    whatever the page is about, so it competes for evidence slots against text
    that actually makes a claim. Measured on a real page, a navigation block
    outranked every substantive passage.

    Stripping URLs also keeps them out of what a model sees, which matters
    because the citation design assumes it can only reference sources by index.
    """
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL_CHARS.sub("", cleaned)
    cleaned = _MARKDOWN_IMAGE.sub("", cleaned)
    # Keep the link's visible text, discard its target.
    cleaned = _MARKDOWN_LINK.sub(r"\1", cleaned)
    cleaned = _BARE_URL.sub("", cleaned)
    cleaned = _INLINE_WHITESPACE.sub(" ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    cleaned = _BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> list[str]:
    """Split ``text`` into overlapping chunks of at most ``chunk_size`` chars.

    ``overlap`` characters of the previous chunk are prepended to each
    subsequent chunk, so the new content in a chunk is budgeted at
    ``chunk_size - overlap``. That keeps the hard size guarantee: no returned
    string exceeds ``chunk_size``.
    """
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    body = normalise(text)
    if not body:
        return []

    content_limit = chunk_size - overlap
    segments = _split_to_limit(body, content_limit)

    groups = _pack(segments, content_limit)
    chunks: list[str] = []
    for index, group in enumerate(groups):
        tail = _overlap_tail(groups[index - 1], overlap) if index else ""
        chunks.append(f"{tail}{group}" if tail else group)

    # A single short chunk is still evidence - a brief page is not noise. The
    # minimum only prunes fragments once the document has real content.
    if len(chunks) == 1:
        return chunks
    return [
        chunk
        for chunk, group in zip(chunks, groups, strict=True)
        if len(group) >= min_chunk_chars
    ]


def chunk_document(
    text: str,
    *,
    source_index: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> list[DocumentChunk]:
    """Chunk one document, tagging each slice with its source and position."""
    return [
        DocumentChunk(source_index=source_index, position=position, text=chunk)
        for position, chunk in enumerate(
            chunk_text(
                text,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_chars=min_chunk_chars,
            )
        )
    ]


def _split_to_limit(text: str, limit: int) -> list[str]:
    """Break ``text`` into segments no longer than ``limit``.

    Paragraphs first, then sentences, then a hard window. Each fallback is only
    reached when the previous one left a segment that still does not fit.
    """
    segments: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= limit:
            segments.append(paragraph)
            continue
        for sentence in _SENTENCE_BREAK.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= limit:
                segments.append(sentence)
            else:
                segments.extend(_hard_window(sentence, limit))
    return segments


def _hard_window(text: str, limit: int) -> list[str]:
    """Last resort: fixed windows, snapped back to a word boundary."""
    windows: list[str] = []
    start = 0
    while start < len(text):
        end = start + limit
        if end >= len(text):
            windows.append(text[start:].strip())
            break
        boundary = text.rfind(" ", start, end)
        # Only honour the boundary if it leaves a substantial window; a space
        # near the start would produce a sliver and loop for a long time.
        if boundary > start + limit // 2:
            end = boundary
        windows.append(text[start:end].strip())
        start = end
    return [window for window in windows if window]


def _pack(segments: list[str], limit: int) -> list[str]:
    """Greedily join segments into groups no longer than ``limit``."""
    groups: list[str] = []
    current: list[str] = []
    length = 0
    for segment in segments:
        # +2 for the "\n\n" that will rejoin them.
        projected = length + len(segment) + (2 if current else 0)
        if current and projected > limit:
            groups.append("\n\n".join(current))
            current, length = [segment], len(segment)
        else:
            current.append(segment)
            length = projected
    if current:
        groups.append("\n\n".join(current))
    return groups


def _overlap_tail(previous: str, overlap: int) -> str:
    """Trailing text of ``previous`` to carry forward, cut at a word boundary."""
    if overlap <= 0 or not previous:
        return ""
    # One character short of the budget, because a separating space is appended
    # below and the total must still fit inside ``overlap``.
    tail = previous[-(overlap - 1) :]
    space = tail.find(" ")
    if space != -1:
        tail = tail[space + 1 :]
    tail = tail.strip()
    return f"{tail} " if tail else ""
