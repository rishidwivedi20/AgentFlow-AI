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
from app.models.evidence import SourceDocument
from app.models.research import (
    ResearchFact,
    ResearchOutput,
    ResearchResult,
    SearchResults,
    Source,
    StatementKind,
)
from app.services.research import tavily
from app.services.research.tavily import (
    PAGE_TEXT_TIMEOUT_SECONDS,
    PROVIDER_NAME,
    SNIPPET_TIMEOUT_SECONDS,
    _map_status,
    _to_results,
)
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
    result, llm_result = await run_researcher(
        make_service(StubProvider()), topic="AI", sources=make_sources(3)
    )
    assert isinstance(result, ResearchResult)
    assert len(result.sources) == 3
    assert all(str(s.url).startswith("https://") for s in result.sources)
    assert llm_result is not None


async def test_no_search_results_skips_the_llm_call():
    """Nothing to extract from, so spending a call to say so would be waste."""
    provider = StubProvider()
    result, llm_result = await run_researcher(make_service(provider), topic="AI", sources=[])
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
    sources = _to_results(
        {
            "results": [
                {"title": "Good", "url": "https://example.com/a", "content": "x"},
                {"title": "Bad URL", "url": "not-a-url", "content": "y"},
                "not even a dict",
                {"url": "https://example.com/b", "content": "no title"},
            ]
        }
    ).sources
    assert [str(s.url) for s in sources] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert sources[1].title == "Untitled"


def test_empty_response_yields_no_sources():
    assert _to_results({}).sources == []
    assert _to_results({"results": []}).sources == []


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


@pytest.mark.parametrize(
    ("include_page_text", "expected"),
    [(False, SNIPPET_TIMEOUT_SECONDS), (True, PAGE_TEXT_TIMEOUT_SECONDS)],
)
async def test_the_timeout_follows_what_the_search_asks_for(
    monkeypatch, include_page_text, expected
):
    """Page text makes Tavily fetch and extract each result live. Measured: five
    results with page text blew straight through the 20s snippet budget, which
    cost the run every source it had already paid for."""
    captured: dict[str, float] = {}

    class FakeClient:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def post(self, *_, **__):
            raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(tavily.httpx, "AsyncClient", FakeClient)
    provider = tavily.TavilyProvider(api_key="test-key")

    with pytest.raises(ResearchProviderError, match=f"within {expected:g}s"):
        await provider.search("q", max_results=3, include_page_text=include_page_text)
    assert captured["timeout"] == expected


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
    unique = _deduplicate(SearchResults(sources=duplicated)).sources
    assert [str(s.url) for s in unique] == ["https://example.com/x", "https://example.com/y"]


def test_deduplication_repoints_documents_at_their_sources_new_index():
    """Documents address sources by position. If collapsing the source list did
    not remap them, a page's text would be attributed to a different site - the
    exact failure the provenance design exists to prevent, and a silent one."""
    results = SearchResults(
        sources=[
            Source(title="A", url="https://example.com/x", snippet="1"),
            Source(title="A again", url="https://example.com/x", snippet="2"),
            Source(title="B", url="https://example.com/y", snippet="3"),
        ],
        documents=[SourceDocument(source_index=2, text="text belonging to B")],
    )
    deduped = _deduplicate(results)
    assert len(deduped.sources) == 2
    # B moved from position 2 to position 1.
    assert deduped.documents[0].source_index == 1
    assert str(deduped.sources[1].url) == "https://example.com/y"


def test_a_duplicates_page_text_is_kept_and_pointed_at_the_survivor():
    """The two entries can differ in what the provider managed to extract."""
    results = SearchResults(
        sources=[
            Source(title="A", url="https://example.com/x", snippet="1"),
            Source(title="A again", url="https://example.com/x", snippet="2"),
        ],
        documents=[SourceDocument(source_index=1, text="page text")],
    )
    deduped = _deduplicate(results)
    assert len(deduped.sources) == 1
    assert deduped.documents[0].source_index == 0


def test_page_text_is_parsed_only_when_the_provider_returns_it():
    """Absent raw_content is normal - a PDF, a paywall, a JS-only site - and
    must leave the source citable rather than failing the search."""
    results = _to_results(
        {
            "results": [
                {"title": "A", "url": "https://example.com/a", "content": "snip"},
                {
                    "title": "B",
                    "url": "https://example.com/b",
                    "content": "snip",
                    "raw_content": "the full page text",
                },
            ]
        }
    )
    assert len(results.sources) == 2
    assert len(results.documents) == 1
    assert results.documents[0].source_index == 1
    assert results.documents[0].text == "the full page text"


def test_a_skipped_result_does_not_shift_later_document_indexes():
    """The index is assigned after a result is known citable, so an unusable
    result leaves no gap for every later document to be misattributed by."""
    results = _to_results(
        {
            "results": [
                {"title": "Bad", "url": "not-a-url", "raw_content": "orphan"},
                {
                    "title": "Good",
                    "url": "https://example.com/b",
                    "raw_content": "real page",
                },
            ]
        }
    )
    assert len(results.sources) == 1
    assert results.documents[0].source_index == 0
    assert results.documents[0].text == "real page"


def test_provider_ranking_is_preserved_by_deduplication():
    sources = make_sources(4)
    assert _deduplicate(SearchResults(sources=sources)).sources == sources


async def test_service_caps_results_at_configured_max():
    service = ResearchService(StubSearchProvider(sources=make_sources(10)), max_results=3)
    assert len((await service.search("q")).sources) == 3


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
