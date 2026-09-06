"""Provider-agnostic contract for text embeddings.

The third protocol in this codebase, alongside ``llm.base.LLMProvider`` and
``research.base.ResearchProvider``, and it follows the same rule: the retrieval
service depends on this module, never on a vendor SDK.

Two methods rather than one with a mode flag. Retrieval quality improves
measurably when documents and queries are embedded asymmetrically, and every
provider that supports it spells that differently - ``task_type`` here, an
instruction prefix elsewhere. Splitting the method keeps the distinction in the
contract and the vocabulary in the adapter.

Vectors are returned unnormalised. Callers must not assume unit length: Gemini
returns unit vectors at full width but *not* at the truncated widths this
project uses (measured: 0.58 at 768 dimensions), so a raw dot product would
silently understate every similarity.
"""

from __future__ import annotations

from typing import Protocol

# Marker placed in AgentFlowError.details by adapters when a failure is worth
# retrying. Mirrors the LLM and research layers so all three read alike.
RETRYABLE_DETAIL_KEY = "retryable"

# Vector = list[float]. Kept as a plain alias rather than a wrapper type: these
# are passed around in the thousands and carry no behaviour of their own.
Vector = list[float]


class EmbeddingProvider(Protocol):
    """What every embedding adapter must offer."""

    @property
    def name(self) -> str:
        """Short provider identifier, e.g. 'gemini'."""
        ...

    @property
    def model(self) -> str:
        """Embedding model this adapter is configured to call."""
        ...

    @property
    def dimensions(self) -> int:
        """Width of the vectors this adapter returns."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed corpus text, one vector per input, in the same order.

        Implementations must batch internally to respect provider limits, and
        must raise rather than return a short list: a caller pairing vectors
        with chunks by position cannot detect a silent drop.

        Implementations must raise:

        * ``ConfigurationError`` when the API key is missing or rejected.
        * ``EmbeddingProviderError`` when the provider fails. Set
          ``details[RETRYABLE_DETAIL_KEY] = True`` only for transient faults.
        """
        ...

    async def embed_query(self, text: str) -> Vector:
        """Embed a search query, using the provider's query-side task type."""
        ...
