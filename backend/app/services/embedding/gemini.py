"""Google Gemini embedding adapter.

Shares ``google.genai`` with ``app.services.llm.gemini`` but stays a separate
adapter: the two speak different protocols, map errors onto different domain
categories, and - the practical reason - draw on separate quotas. Embedding
calls have been observed succeeding while text generation was rate limited.

This adapter is used whatever ``LLM_PROVIDER`` is set to, because Groq exposes
no embeddings API. Retrieval therefore needs ``GEMINI_API_KEY`` even on a Groq
run; see ``Settings.embeddings_configured``.
"""

from __future__ import annotations

import logging
from typing import Final

from google import genai
from google.genai import errors, types

from app.core.exceptions import (
    AgentFlowError,
    ConfigurationError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
)
from app.services.embedding.base import RETRYABLE_DETAIL_KEY, Vector

logger = logging.getLogger(__name__)

PROVIDER_NAME: Final = "gemini"

DEFAULT_EMBEDDING_MODEL: Final = "gemini-embedding-001"
"""The only current Gemini embedding model. ``text-embedding-004`` was
retired and now answers 404 NOT_FOUND."""

DEFAULT_DIMENSIONS: Final = 768
"""Truncated from the model's native 3072. Retrieval quality at 768 is
comparable for a corpus this small, and the vectors are a quarter the size."""

MAX_BATCH_SIZE: Final = 100
"""Hard provider limit, quoted verbatim by the API on overflow:
"BatchEmbedContentsRequest.requests: at most 100 requests can be in one batch".

The free tier also bills 100 embedded items per minute, counting each text in
a batch separately, so one full batch spends a minute's quota. Batching saves
round trips, not quota, which is why the corpus is capped before it reaches
here."""

_DOCUMENT_TASK_TYPE: Final = "RETRIEVAL_DOCUMENT"
_QUERY_TASK_TYPE: Final = "RETRIEVAL_QUERY"

_RATE_LIMIT_STATUS_CODES: Final = frozenset({429})
_AUTH_STATUS_CODES: Final = frozenset({401, 403})


def _map_api_error(exc: errors.APIError) -> AgentFlowError:
    """Translate a provider API error into a meaningful domain category.

    Only ``exc.code`` and ``exc.message`` are used. ``exc.details`` holds the
    raw response body and is deliberately never propagated - the same rule the
    generation adapter follows.
    """
    code = exc.code

    if code in _RATE_LIMIT_STATUS_CODES:
        return EmbeddingRateLimitError(
            "Gemini embedding rate limit or quota exceeded. The free tier allows "
            "100 embedded passages per minute; lower RETRIEVAL_MAX_CHUNKS_PER_RUN "
            "or TAVILY_MAX_RESULTS to fit inside it.",
            details={"provider": PROVIDER_NAME, "status": code},
        )

    if code in _AUTH_STATUS_CODES:
        return ConfigurationError(
            "Gemini rejected the API key used for embeddings. Check "
            "GEMINI_API_KEY in backend/.env - it is required for retrieval even "
            "when LLM_PROVIDER is groq, because Groq has no embeddings API.",
            details={"provider": PROVIDER_NAME, "status": code},
        )

    if code is not None and code >= 500:
        return EmbeddingProviderError(
            f"Gemini embeddings are temporarily unavailable: {exc.message}",
            details={RETRYABLE_DETAIL_KEY: True, "provider": PROVIDER_NAME, "status": code},
        )

    return EmbeddingProviderError(
        f"Gemini rejected the embedding request: {exc.message}",
        details={"provider": PROVIDER_NAME, "status": code},
    )


def _extract_vectors(
    response: types.EmbedContentResponse, *, expected: int, dimensions: int
) -> list[Vector]:
    """Pull vectors out of a response, refusing anything that does not line up.

    The caller pairs vectors with chunks by position, so a short or ragged
    batch must fail loudly rather than silently mis-attribute evidence to the
    wrong source.
    """
    embeddings = response.embeddings or []
    if len(embeddings) != expected:
        raise EmbeddingProviderError(
            f"Gemini returned {len(embeddings)} embeddings for {expected} inputs.",
            details={RETRYABLE_DETAIL_KEY: True, "provider": PROVIDER_NAME},
        )

    vectors: list[Vector] = []
    for embedding in embeddings:
        values = embedding.values
        if not values or len(values) != dimensions:
            raise EmbeddingProviderError(
                f"Gemini returned a {len(values or [])}-dimension vector, expected "
                f"{dimensions}.",
                details={RETRYABLE_DETAIL_KEY: True, "provider": PROVIDER_NAME},
            )
        vectors.append(list(values))
    return vectors


class GeminiEmbeddingProvider:
    """Embedding adapter over the Google Gen AI SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed corpus text, batching to the provider's 100-input limit."""
        if not texts:
            return []

        vectors: list[Vector] = []
        for start in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[start : start + MAX_BATCH_SIZE]
            vectors.extend(await self._embed(batch, task_type=_DOCUMENT_TASK_TYPE))

        logger.debug(
            "Embedded documents",
            extra={
                "provider": PROVIDER_NAME,
                "model": self._model,
                "count": len(vectors),
                "batches": -(-len(texts) // MAX_BATCH_SIZE),
            },
        )
        return vectors

    async def embed_query(self, text: str) -> Vector:
        """Embed one query with the query-side task type."""
        vectors = await self._embed([text], task_type=_QUERY_TASK_TYPE)
        return vectors[0]

    async def _embed(self, texts: list[str], *, task_type: str) -> list[Vector]:
        # No auto_truncate: the SDK rejects it outright on the Developer API
        # ("only supported in Gemini Enterprise Agent Platform mode"). Nothing
        # is lost - chunking bounds every input to at most a few thousand
        # characters, far below the model's input window.
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._dimensions,
        )

        try:
            response = await self._client.aio.models.embed_content(
                model=self._model,
                contents=texts,
                config=config,
            )
        except errors.APIError as exc:
            raise _map_api_error(exc) from exc

        return _extract_vectors(response, expected=len(texts), dimensions=self._dimensions)
