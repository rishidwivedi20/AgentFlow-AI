"""Provider-agnostic contract for web search.

Mirrors ``app.services.llm.base``: the Researcher agent depends on this
protocol, never on a vendor SDK or HTTP client, so swapping search backends is
one adapter.

The contract returns :class:`~app.models.research.Source` objects built from a
real HTTP response. Nothing in this layer accepts model output, which is what
makes fabricated citations impossible rather than merely discouraged.
"""

from __future__ import annotations

from typing import Protocol

from app.models.research import Source

# Marker placed in AgentFlowError.details by adapters when a failure is worth
# retrying. Mirrors the LLM layer's convention so the two read alike.
RETRYABLE_DETAIL_KEY = "retryable"


class ResearchProvider(Protocol):
    """What every search adapter must offer."""

    @property
    def name(self) -> str:
        """Short provider identifier, e.g. 'tavily'."""
        ...

    async def search(self, query: str, *, max_results: int) -> list[Source]:
        """Return real search results for ``query``.

        Implementations must raise:

        * ``ConfigurationError`` when the API key is missing or rejected.
        * ``ResearchProviderError`` when the provider fails. Set
          ``details[RETRYABLE_DETAIL_KEY] = True`` only for transient faults.

        An empty list is a valid result - a query can legitimately match
        nothing - and is not an error.
        """
        ...
