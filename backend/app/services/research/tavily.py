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
from app.models.research import Source
from app.services.research.base import RETRYABLE_DETAIL_KEY

logger = logging.getLogger(__name__)

PROVIDER_NAME: Final = "tavily"
_ENDPOINT: Final = "https://api.tavily.com/search"


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


def _to_sources(payload: dict[str, Any]) -> list[Source]:
    """Build Source objects from the response body.

    Individual malformed results are skipped rather than failing the search: a
    single result missing a URL should not lose the other four. A result that
    cannot be validated simply does not become a citable source.
    """
    sources: list[Source] = []
    for raw in payload.get("results") or []:
        if not isinstance(raw, dict):
            continue
        try:
            sources.append(
                Source(
                    title=str(raw.get("title") or "").strip() or "Untitled",
                    url=raw.get("url"),
                    snippet=str(raw.get("content") or "").strip(),
                )
            )
        except ValidationError:
            logger.warning(
                "Skipping unusable search result",
                extra={"provider": PROVIDER_NAME, "url": str(raw.get("url"))[:120]},
            )
    return sources


class TavilyProvider:
    """Search adapter over the Tavily REST API."""

    def __init__(
        self,
        *,
        api_key: str,
        search_depth: str = "basic",
        timeout_seconds: float = 20.0,
    ) -> None:
        self._api_key = api_key
        self._search_depth = search_depth
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    async def search(self, query: str, *, max_results: int) -> list[Source]:
        """Execute one search and return real, validated sources."""
        body = {
            "query": query,
            "search_depth": self._search_depth,
            "max_results": max_results,
            # No generated answer: the Researcher agent extracts facts itself,
            # and an extra summary would cost credits for nothing.
            "include_answer": False,
            "include_raw_content": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(
                    _ENDPOINT,
                    json=body,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
        except httpx.TimeoutException as exc:
            raise ResearchProviderError(
                f"Tavily did not respond within {self._timeout_seconds:g}s.",
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

        sources = _to_sources(payload)
        logger.info(
            "Search completed",
            extra={"provider": PROVIDER_NAME, "query": query[:80], "sources": len(sources)},
        )
        return sources
