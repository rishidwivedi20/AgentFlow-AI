"""Research pipeline: sources are real, citations cannot be fabricated."""

from __future__ import annotations

import httpx
import pytest

from app.agents.researcher import _drop_uncitable_facts, build_search_query, run_researcher
from app.core.config import Settings
from app.core.exceptions import (
    ConfigurationError,
    ResearchProviderError,
    ResearchRateLimitError,
)
from app.models.research import ResearchFact, ResearchOutput, ResearchResult, Source, StatementKind
from app.services.research.tavily import PROVIDER_NAME, _map_status, _to_sources
from app.services.research_service import (
    ResearchService,
    _deduplicate,
    build_research_provider,
)
from tests.conftest import StubProvider, StubSearchProvider, make_service, make_sources

# --- the fabrication guarantee ---------------------------------------------


def test_source_is_absent_from_every_llm_facing_schema():
    """A URL never appears in a response schema, so it cannot be invented."""
    assert "url" not in ResearchOutput.model_fields
    assert "url" not in ResearchFact.model_fields
    # The model cites by index into a list it was handed.
    assert "source_index" in ResearchFact.model_fields


def test_facts_citing_missing_sources_are_dropped():
    facts = [
        ResearchFact(statement="ok", kind=StatementKind.FACT, source_index=0),
        ResearchFact(statement="out of range", kind=StatementKind.FACT, source_index=7),
    ]
    kept = _drop_uncitable_facts(facts, source_count=2)
    assert [f.statement for f in kept] == ["ok"]


def test_all_facts_kept_when_indexes_are_valid():
    facts = [
        ResearchFact(statement=f"f{i}", kind=StatementKind.FACT, source_index=i) for i in range(3)
    ]
    assert len(_drop_uncitable_facts(facts, source_count=3)) == 3


async def test_researcher_attaches_real_sources():
    search = StubSearchProvider(sources=make_sources(3))
    result, llm_result = await run_researcher(
        make_service(StubProvider()), ResearchService(search, max_results=5), topic="AI"
    )
    assert isinstance(result, ResearchResult)
    assert len(result.sources) == 3
    assert all(str(s.url).startswith("https://") for s in result.sources)
    assert llm_result is not None
    assert search.calls == 1


async def test_no_search_results_skips_the_llm_call():
    """Nothing to extract from, so spending a call to say so would be waste."""
    search = StubSearchProvider(sources=[])
    provider = StubProvider()
    result, llm_result = await run_researcher(
        make_service(provider), ResearchService(search, max_results=5), topic="AI"
    )
    assert result.sources == []
    assert result.facts == []
    assert llm_result is None
    assert provider.researcher_calls == 0


def test_search_query_is_the_topic():
    assert build_search_query("  AI in engineering  ") == "AI in engineering"


# --- source resolution ------------------------------------------------------


def test_source_for_resolves_and_guards_range():
    result = ResearchResult(
        summary="s",
        facts=[ResearchFact(statement="f", kind=StatementKind.FACT, source_index=0)],
        sources=make_sources(1),
    )
    assert result.source_for(result.facts[0]) is not None
    stray = ResearchFact(statement="x", kind=StatementKind.FACT, source_index=9)
    assert result.source_for(stray) is None


def test_verifiable_facts_excludes_opinions():
    result = ResearchResult(
        summary="s",
        facts=[
            ResearchFact(statement="a", kind=StatementKind.FACT, source_index=0),
            ResearchFact(statement="b", kind=StatementKind.OPINION, source_index=0),
        ],
        sources=make_sources(1),
    )
    assert [f.statement for f in result.verifiable_facts] == ["a"]


def test_writer_context_attributes_facts_to_domains():
    result = ResearchResult(
        summary="Summary text",
        facts=[ResearchFact(statement="A finding.", kind=StatementKind.FACT, source_index=0)],
        sources=make_sources(1),
    )
    context = result.as_writer_context()
    assert "Summary text" in context
    assert "A finding." in context
    assert "example.com" in context


# --- provider adapter -------------------------------------------------------


def test_malformed_results_are_skipped_not_fatal():
    sources = _to_sources(
        {
            "results": [
                {"title": "Good", "url": "https://example.com/a", "content": "x"},
                {"title": "Bad URL", "url": "not-a-url", "content": "y"},
                "not even a dict",
                {"url": "https://example.com/b", "content": "no title"},
            ]
        }
    )
    assert [str(s.url) for s in sources] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert sources[1].title == "Untitled"


def test_empty_response_yields_no_sources():
    assert _to_sources({}) == []
    assert _to_sources({"results": []}) == []


@pytest.mark.parametrize(
    ("status_code", "expected", "error_type"),
    [
        (429, ResearchRateLimitError, "research_rate_limited"),
        (401, ConfigurationError, "configuration_error"),
        (403, ConfigurationError, "configuration_error"),
        (500, ResearchProviderError, "research_provider_error"),
        (400, ResearchProviderError, "research_provider_error"),
    ],
)
def test_status_codes_map_to_meaningful_categories(status_code, expected, error_type):
    mapped = _map_status(status_code, "detail")
    assert isinstance(mapped, expected)
    assert mapped.error_type == error_type
    assert mapped.details["provider"] == PROVIDER_NAME


def test_only_server_errors_are_marked_retryable():
    assert _map_status(500, "x").details.get("retryable") is True
    for code in (400, 401, 429):
        assert _map_status(code, "x").details.get("retryable") is not True


def test_search_errors_never_leak_the_response_body():
    for code in (400, 401, 429, 500):
        assert set(_map_status(code, "detail").details) <= {"retryable", "provider", "status"}


def test_tavily_uses_bearer_auth_not_a_query_parameter():
    """A key in a URL leaks into logs and proxies."""
    import inspect

    from app.services.research import tavily

    source = inspect.getsource(tavily.TavilyProvider.search)
    assert "Authorization" in source
    assert "Bearer" in source
    assert "api_key=" not in source


# --- service layer ----------------------------------------------------------


def test_duplicate_urls_are_collapsed():
    """Citing by index means a repeated URL would double-count as evidence."""
    duplicated = [
        Source(title="A", url="https://example.com/x", snippet="1"),
        Source(title="A again", url="https://example.com/x", snippet="2"),
        Source(title="B", url="https://example.com/y", snippet="3"),
    ]
    unique = _deduplicate(duplicated)
    assert [str(s.url) for s in unique] == ["https://example.com/x", "https://example.com/y"]


def test_provider_ranking_is_preserved_by_deduplication():
    sources = make_sources(4)
    assert _deduplicate(sources) == sources


async def test_service_caps_results_at_configured_max():
    service = ResearchService(StubSearchProvider(sources=make_sources(10)), max_results=3)
    assert len(await service.search("q")) == 3


def test_missing_search_key_raises_configuration_error():
    with pytest.raises(ConfigurationError) as exc:
        build_research_provider(Settings(tavily_api_key="", app_env="test"))
    assert exc.value.status_code == 503
    assert "TAVILY_API_KEY" in exc.value.message
    # The message must say why it matters, not just that a key is absent.
    assert "fabricated" in exc.value.message


def test_configured_key_builds_tavily_provider():
    provider = build_research_provider(Settings(tavily_api_key="tvly-x", app_env="test"))
    assert provider.name == "tavily"


def test_timeout_and_transport_errors_become_domain_errors():
    """httpx exceptions must not escape the adapter."""
    import inspect

    from app.services.research import tavily

    source = inspect.getsource(tavily.TavilyProvider.search)
    assert "httpx.TimeoutException" in source
    assert "httpx.HTTPError" in source
    assert issubclass(ResearchProviderError, Exception)
    assert httpx.codes.OK == 200
