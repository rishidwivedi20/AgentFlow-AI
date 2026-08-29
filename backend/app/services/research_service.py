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
from app.models.research import Source
from app.services.research.base import ResearchProvider
from app.services.research.tavily import TavilyProvider

logger = logging.getLogger(__name__)


class ResearchService:
    """Provider-agnostic entry point for web search."""

    def __init__(self, provider: ResearchProvider, *, max_results: int) -> None:
        self._provider = provider
        self._max_results = max_results

    @property
    def provider_name(self) -> str:
        return self._provider.name

    async def search(self, query: str) -> list[Source]:
        """Search for ``query``, returning de-duplicated sources.

        Duplicate URLs are collapsed because the Researcher cites sources by
        index: two identical entries would let the same evidence appear to
        corroborate a claim twice.
        """
        sources = await self._provider.search(query, max_results=self._max_results)
        return _deduplicate(sources)


def _deduplicate(sources: list[Source]) -> list[Source]:
    """Drop repeated URLs, preserving provider ranking."""
    seen: set[str] = set()
    unique: list[Source] = []
    for source in sources:
        key = str(source.url)
        if key not in seen:
            seen.add(key)
            unique.append(source)
    if len(unique) != len(sources):
        logger.debug("Dropped duplicate sources", extra={"removed": len(sources) - len(unique)})
    return unique


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
    )
