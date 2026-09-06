"""Tavily search adapter.

The only module that knows how to talk to Tavily. Its REST endpoint is a single
authenticated POST, so it is called directly with ``httpx`` rather than through
``tavily-python`` - that package pulls in ``requests`` and ``tiktoken``, neither
of which this project would use.
"""

from __future__ import annotations

import logging
from typing import Any, Final

import httpx
from pydantic import ValidationError

from app.core.exceptions import (
    ConfigurationError,
    ResearchProviderError,
    ResearchRateLimitError,
)
from app.models.evidence import SourceDocument
from app.models.research import SearchResults, Source
from app.services.research.base import RETRYABLE_DETAIL_KEY

logger = logging.getLogger(__name__)

PROVIDER_NAME: Final = "tavily"
_ENDPOINT: Final = "https://api.tavily.com/search"

SNIPPET_TIMEOUT_SECONDS: Final = 20.0
"""Ample for a snippet-only search, which Tavily answers from its index."""

PAGE_TEXT_TIMEOUT_SECONDS: Final = 90.0
"""Requesting page text makes Tavily fetch and extract each result live, which
is a different order of work. Measured: five results with page text exceeded
the 20s snippet budget outright, while the same query without it returned in a
few seconds. Generous on purpose - a timeout here costs the run its sources."""


def _map_status(status_code: int, detail: str) -> ResearchProviderError | ConfigurationError:
    """Translate an HTTP status into a meaningful domain category.

    Only the status code and a short provider message are used; the raw
    response body is never propagated, matching the LLM adapter's rule.
    """
    if status_code == 429:
        return ResearchRateLimitError(
            "Tavily credit or rate limit reached. The free tier allows 1,000 "
            "credits per month.",
            details={"provider": PROVIDER_NAME, "status": status_code},
        )
    if status_code in (401, 403):
        return ConfigurationError(
            "Tavily rejected the API key. Check TAVILY_API_KEY in backend/.env.",
            details={"provider": PROVIDER_NAME, "status": status_code},
        )
    if status_code >= 500:
        return ResearchProviderError(
            "Tavily is temporarily unavailable.",
            details={
                RETRYABLE_DETAIL_KEY: True,
                "provider": PROVIDER_NAME,
                "status": status_code,
            },
        )
    return ResearchProviderError(
        f"Tavily rejected the search request: {detail}",
        details={"provider": PROVIDER_NAME, "status": status_code},
    )


def _to_results(payload: dict[str, Any]) -> SearchResults:
    """Build sources, and their page text where present, from the response body.

    Individual malformed results are skipped rather than failing the search: a
    single result missing a URL should not lose the other four. A result that
    cannot be validated simply does not become a citable source.

    ``raw_content`` is absent whenever Tavily could not extract the page - a
    PDF, a paywall, a JavaScript-only site. That is normal, not an error: the
    source stays citable through its snippet and contributes no chunks.
    """
    sources: list[Source] = []
    documents: list[SourceDocument] = []
    for raw in payload.get("results") or []:
        if not isinstance(raw, dict):
            continue
        try:
            source = Source(
                title=str(raw.get("title") or "").strip() or "Untitled",
                url=raw.get("url"),
                snippet=str(raw.get("content") or "").strip(),
            )
        except ValidationError:
            logger.warning(
                "Skipping unusable search result",
                extra={"provider": PROVIDER_NAME, "url": str(raw.get("url"))[:120]},
            )
            continue

        # The index is assigned here, after the result is known to be citable,
        # so a skipped result never leaves a gap that shifts every later index.
        page_text = str(raw.get("raw_content") or "").strip()
        if page_text:
            documents.append(
                SourceDocument(source_index=len(sources), text=page_text)
            )
        sources.append(source)

    return SearchResults(sources=sources, documents=documents)


class TavilyProvider:
    """Search adapter over the Tavily REST API."""

    def __init__(
        self,
        *,
        api_key: str,
        search_depth: str = "basic",
        timeout_seconds: float = SNIPPET_TIMEOUT_SECONDS,
        page_text_timeout_seconds: float = PAGE_TEXT_TIMEOUT_SECONDS,
    ) -> None:
        self._api_key = api_key
        self._search_depth = search_depth
        self._timeout_seconds = timeout_seconds
        self._page_text_timeout_seconds = page_text_timeout_seconds

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    async def search(
        self, query: str, *, max_results: int, include_page_text: bool = False
    ) -> SearchResults:
        """Execute one search and return real, validated sources."""
        body = {
            "query": query,
            "search_depth": self._search_depth,
            "max_results": max_results,
            # No generated answer: the Researcher agent extracts facts itself,
            # and an extra summary would cost credits for nothing.
            "include_answer": False,
            # Page text is roughly 15x the size of a snippet (measured: ~21,000
            # characters per page against ~1,300), so it is requested only when
            # retrieval is actually going to chunk and embed it.
            "include_raw_content": include_page_text,
        }

        # Extracting whole pages takes materially longer than returning
        # snippets, so the budget follows the request rather than penalising
        # snippet-only searches with a timeout they never need.
        timeout = (
            self._page_text_timeout_seconds if include_page_text else self._timeout_seconds
        )

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    _ENDPOINT,
                    json=body,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
        except httpx.TimeoutException as exc:
            raise ResearchProviderError(
                f"Tavily did not respond within {timeout:g}s.",
                details={"provider": PROVIDER_NAME},
            ) from exc
        except httpx.HTTPError as exc:
            raise ResearchProviderError(
                "Could not reach Tavily.",
                details={RETRYABLE_DETAIL_KEY: True, "provider": PROVIDER_NAME},
            ) from exc

        if response.status_code != httpx.codes.OK:
            raise _map_status(response.status_code, response.reason_phrase)

        try:
            payload = response.json()
        except ValueError as exc:
            raise ResearchProviderError(
                "Tavily returned a response that was not valid JSON.",
                details={"provider": PROVIDER_NAME},
            ) from exc

        results = _to_results(payload)
        logger.info(
            "Search completed",
            extra={
                "provider": PROVIDER_NAME,
                "query": query[:80],
                "sources": len(results.sources),
                "documents": len(results.documents),
            },
        )
        return results
