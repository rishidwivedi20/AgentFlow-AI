"""Centralised web search access.

Mirrors ``llm_service``: the agent calls this, never an adapter. Deduplication
and the "no results is not an error" rule live here so every provider behaves
alike.

No retries. A search failure costs the run its research, and the graph already
degrades gracefully; retrying would spend another of the 1,000 monthly free
credits to make the same request.
"""

from __future__ import annotations

import logging

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.models.research import SearchResults, Source
from app.services.research.base import ResearchProvider
from app.services.research.tavily import TavilyProvider

logger = logging.getLogger(__name__)


class ResearchService:
    """Provider-agnostic entry point for web search."""

    def __init__(
        self,
        provider: ResearchProvider,
        *,
        max_results: int,
        include_page_text: bool = False,
    ) -> None:
        self._provider = provider
        self._max_results = max_results
        self._include_page_text = include_page_text

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def includes_page_text(self) -> bool:
        """Whether searches ask for full page text as well as snippets."""
        return self._include_page_text

    async def search(self, query: str) -> SearchResults:
        """Search for ``query``, returning de-duplicated results.

        Duplicate URLs are collapsed because sources are cited by index: two
        identical entries would let the same evidence appear to corroborate a
        claim twice.
        """
        results = await self._provider.search(
            query,
            max_results=self._max_results,
            include_page_text=self._include_page_text,
        )
        return _deduplicate(results)


def _deduplicate(results: SearchResults) -> SearchResults:
    """Drop repeated URLs, preserving provider ranking.

    Documents address sources by position, so collapsing the source list means
    re-pointing every document at its source's new index. Getting this wrong
    fails silently, attributing a page's text to the wrong site.
    """
    first_index: dict[str, int] = {}
    remap: dict[int, int] = {}
    unique: list[Source] = []

    for old_index, source in enumerate(results.sources):
        key = str(source.url)
        if key in first_index:
            # Point at the surviving copy rather than discarding: the two
            # entries can differ in what the provider extracted.
            remap[old_index] = first_index[key]
            continue
        first_index[key] = len(unique)
        remap[old_index] = len(unique)
        unique.append(source)

    documents = [
        document.model_copy(update={"source_index": remap[document.source_index]})
        for document in results.documents
        if document.source_index in remap
    ]

    if len(unique) != len(results.sources):
        logger.debug(
            "Dropped duplicate sources",
            extra={"removed": len(results.sources) - len(unique)},
        )
    return SearchResults(sources=unique, documents=documents)


def build_research_provider(settings: Settings) -> ResearchProvider:
    """Instantiate the configured search adapter."""
    if not settings.research_configured:
        raise ConfigurationError(
            "No API key configured for web search. Set TAVILY_API_KEY in "
            "backend/.env. Without it the Researcher cannot run: any source it "
            "produced would be fabricated.",
            details={"provider": "tavily"},
        )
    return TavilyProvider(
        api_key=settings.tavily_api_key,
        search_depth=settings.tavily_search_depth,
    )


def build_research_service(settings: Settings | None = None) -> ResearchService:
    """Build the service from configuration.

    Not cached: tests construct :class:`ResearchService` with a stub provider,
    and the graph receives its service by injection.
    """
    resolved = settings or get_settings()
    return ResearchService(
        build_research_provider(resolved),
        max_results=resolved.tavily_max_results,
        # Page text is only worth fetching if something is going to retrieve
        # over it. With retrieval off it is bandwidth, and possibly credits,
        # spent on text nothing reads.
        include_page_text=resolved.retrieval_available,
    )
