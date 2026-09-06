"""Gemini embedding adapter: batching, error mapping and response validation.

No network. The SDK's HTTP layer is never exercised - what is tested is
everything we put around it, in the same spirit as ``test_groq_provider``.
"""

from __future__ import annotations

import pytest
from google.genai import errors, types

from app.core.exceptions import (
    ConfigurationError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
)
from app.services.embedding.base import RETRYABLE_DETAIL_KEY
from app.services.embedding.gemini import (
    MAX_BATCH_SIZE,
    PROVIDER_NAME,
    GeminiEmbeddingProvider,
    _map_api_error,
)

DIMENSIONS = 4


def api_error(code: int, message: str = "boom") -> errors.APIError:
    """Build a real APIError the way the SDK does, from a response body."""
    return errors.APIError(code, {"error": {"message": message, "status": "ERROR"}})


class FakeModels:
    """Records every call and replays canned responses."""

    def __init__(self, *, dimensions: int = DIMENSIONS, error: Exception | None = None) -> None:
        self._dimensions = dimensions
        self._error = error
        self.calls: list[dict] = []

    async def embed_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        if self._error is not None:
            raise self._error
        return types.EmbedContentResponse(
            embeddings=[
                types.ContentEmbedding(values=[float(index)] * self._dimensions)
                for index in range(len(contents))
            ]
        )


def provider_with(models: FakeModels, *, dimensions: int = DIMENSIONS) -> GeminiEmbeddingProvider:
    """A real provider with its SDK client swapped for the fake."""
    provider = GeminiEmbeddingProvider(
        api_key="test-key", model="gemini-embedding-001", dimensions=dimensions
    )
    provider._client = type("Client", (), {"aio": type("Aio", (), {"models": models})()})()
    return provider


# --- error mapping ---------------------------------------------------------


def test_rate_limit_maps_to_its_own_category():
    """Distinct from the LLM's rate limit: embeddings draw on a separate quota,
    and have been observed working while generation was throttled."""
    mapped = _map_api_error(api_error(429, "Quota exceeded"))
    assert isinstance(mapped, EmbeddingRateLimitError)
    assert mapped.error_type == "embedding_rate_limited"
    assert mapped.status_code == 429


def test_rate_limit_is_not_retryable():
    mapped = _map_api_error(api_error(429))
    assert mapped.details.get(RETRYABLE_DETAIL_KEY) is not True


@pytest.mark.parametrize("code", [401, 403])
def test_auth_failures_name_the_key_and_explain_the_groq_case(code):
    mapped = _map_api_error(api_error(code, "API key not valid"))
    assert isinstance(mapped, ConfigurationError)
    assert mapped.status_code == 503
    assert "GEMINI_API_KEY" in mapped.message
    assert "groq" in mapped.message.casefold()


@pytest.mark.parametrize("code", [500, 503])
def test_server_errors_are_retryable(code):
    mapped = _map_api_error(api_error(code))
    assert isinstance(mapped, EmbeddingProviderError)
    assert mapped.details[RETRYABLE_DETAIL_KEY] is True


def test_other_client_errors_are_not_retryable():
    mapped = _map_api_error(api_error(400, "bad request"))
    assert isinstance(mapped, EmbeddingProviderError)
    assert mapped.details.get(RETRYABLE_DETAIL_KEY) is not True
    assert mapped.details["provider"] == PROVIDER_NAME


def test_raw_provider_details_never_reach_the_message():
    """``exc.details`` holds the response body, which can echo request text."""
    mapped = _map_api_error(api_error(400, "secret-request-echo"))
    assert "secret-request-echo" in mapped.message
    assert "error" not in mapped.details


# --- batching --------------------------------------------------------------


async def test_a_small_batch_is_one_call():
    models = FakeModels()
    vectors = await provider_with(models).embed_documents(["a", "b", "c"])
    assert len(models.calls) == 1
    assert len(vectors) == 3


async def test_batches_are_split_at_the_provider_limit():
    """The API rejects anything larger: "at most 100 requests can be in one batch"."""
    models = FakeModels()
    texts = [f"chunk {index}" for index in range(MAX_BATCH_SIZE + 45)]
    vectors = await provider_with(models).embed_documents(texts)

    assert len(models.calls) == 2
    assert len(models.calls[0]["contents"]) == MAX_BATCH_SIZE
    assert len(models.calls[1]["contents"]) == 45
    assert len(vectors) == len(texts)


async def test_batch_order_is_preserved_across_the_split():
    """Vectors are paired with chunks by position, so order is load-bearing."""
    models = FakeModels()
    texts = [f"chunk {index}" for index in range(MAX_BATCH_SIZE + 5)]
    await provider_with(models).embed_documents(texts)
    sent = models.calls[0]["contents"] + models.calls[1]["contents"]
    assert sent == texts


async def test_no_inputs_costs_no_call():
    models = FakeModels()
    assert await provider_with(models).embed_documents([]) == []
    assert models.calls == []


# --- task types ------------------------------------------------------------


async def test_documents_and_queries_use_asymmetric_task_types():
    """Retrieval quality improves measurably when the two sides differ."""
    models = FakeModels()
    provider = provider_with(models)
    await provider.embed_documents(["a"])
    await provider.embed_query("a")

    assert models.calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert models.calls[1]["config"].task_type == "RETRIEVAL_QUERY"


async def test_the_configured_width_is_requested():
    models = FakeModels(dimensions=768)
    provider = provider_with(models, dimensions=768)
    await provider.embed_documents(["a"])
    assert models.calls[0]["config"].output_dimensionality == 768


async def test_the_config_carries_no_vertex_only_fields():
    """Regression guard. ``auto_truncate`` looked harmless and passed every
    mocked test, then raised ValueError against the real Developer API:
    "only supported in Gemini Enterprise Agent Platform mode". A fake client
    accepts any config, so the shape of the config has to be asserted directly.
    """
    models = FakeModels()
    await provider_with(models).embed_documents(["a"])
    config = models.calls[0]["config"]
    assert config.model_dump(exclude_none=True).keys() == {
        "task_type",
        "output_dimensionality",
    }


async def test_embed_query_returns_a_single_vector():
    provider = provider_with(FakeModels())
    assert len(await provider.embed_query("alpha")) == DIMENSIONS


# --- response validation ---------------------------------------------------


class ShortModels(FakeModels):
    async def embed_content(self, **kwargs):
        response = await super().embed_content(**kwargs)
        return types.EmbedContentResponse(embeddings=response.embeddings[:-1])


class NarrowModels(FakeModels):
    async def embed_content(self, **kwargs):
        await super().embed_content(**kwargs)
        return types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[1.0])])


async def test_a_short_response_is_rejected_rather_than_mis_paired():
    provider = provider_with(ShortModels())
    with pytest.raises(EmbeddingProviderError, match="embeddings for"):
        await provider.embed_documents(["a", "b"])


async def test_a_wrong_width_response_is_rejected():
    """A truncated vector would score against the query at the wrong width."""
    provider = provider_with(NarrowModels())
    with pytest.raises(EmbeddingProviderError, match="dimension vector"):
        await provider.embed_documents(["a"])


async def test_an_empty_response_is_rejected():
    models = FakeModels()

    async def empty(**kwargs):
        models.calls.append(kwargs)
        return types.EmbedContentResponse(embeddings=[])

    models.embed_content = empty
    with pytest.raises(EmbeddingProviderError, match="embeddings for"):
        await provider_with(models).embed_documents(["a"])


async def test_api_errors_are_mapped_not_leaked():
    provider = provider_with(FakeModels(error=api_error(429, "Quota exceeded")))
    with pytest.raises(EmbeddingRateLimitError):
        await provider.embed_documents(["a"])
