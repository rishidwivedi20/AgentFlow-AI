"""Phase 6: the single-agent baseline and the evaluation harness."""

from __future__ import annotations

import json

import pytest

from app.core.exceptions import LLMProviderError, LLMRateLimitError
from app.evaluation import runner
from app.evaluation.__main__ import _load_topics, _parse_args
from app.evaluation.judge import neutral_research
from app.evaluation.models import (
    EvaluationReport,
    Grounding,
    SystemName,
    summarise,
)
from app.evaluation.report import render_markdown
from app.evaluation.runner import RateLimitPolicy, evaluate, evaluate_topic
from app.models.enums import Platform, Tone
from app.models.factcheck import FactCheckReport
from app.services.pricing import ModelPricing
from tests.conftest import (
    StubProvider,
    StubSearchProvider,
    make_fact_check,
    make_research,
    make_service,
    make_sources,
)

PRICING = ModelPricing(model="stub-model", input_per_mtok=1.0, output_per_mtok=2.0, source="test")


async def _evaluate(provider: StubProvider, settings, *, search=None, topic="AI in teams"):
    return await evaluate_topic(
        make_service(provider),
        make_research(search),
        topic=topic,
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
        settings=settings,
        retrieval=None,
        pricing=PRICING,
    )


# --- one topic -------------------------------------------------------------


async def test_both_systems_run_and_are_scored(settings):
    provider = StubProvider()
    result = await _evaluate(provider, settings)

    multi, single = result.multi_agent, result.single_agent
    assert multi.succeeded and single.succeeded
    # researcher, planner, writer, editor, fact checker - retriever and
    # finalize are shared infrastructure, not generation.
    assert multi.llm_calls == 5
    assert single.llm_calls == 1
    assert single.revisions == 0
    assert multi.workflow_status is not None
    assert single.workflow_status is None

    # One Writer call each, then one judge and one grounding check per system.
    assert provider.writer_calls == 2
    assert provider.judge_calls == 2
    assert provider.fact_check_calls == 1 + 2
    assert multi.quality is not None and single.quality is not None
    assert multi.grounding is not None and single.grounding is not None
    assert result.evaluator_tokens > 0


async def test_costs_exclude_the_shared_retriever(settings):
    result = await _evaluate(StubProvider(), settings)
    # Every stub call reports 100 prompt and 250 billed output tokens.
    per_call = (100 * 1.0 + 250 * 2.0) / 1_000_000
    assert result.single_agent.cost_usd == pytest.approx(per_call)
    assert result.multi_agent.cost_usd == pytest.approx(per_call * 5)


async def test_quality_winner_follows_the_judge(settings):
    # The multi-agent output is judged first.
    result = await _evaluate(StubProvider(judge_scores=[6.0, 8.0]), settings)
    assert result.quality_winner is SystemName.SINGLE_AGENT

    tied = await _evaluate(StubProvider(judge_scores=[7.0, 7.0]), settings)
    assert tied.quality_winner is None


async def test_baseline_sees_the_same_sources_in_one_prompt(settings):
    provider = StubProvider()
    await _evaluate(provider, settings)

    baseline_prompt = provider.prompts["writer"][-1]
    for index in range(3):
        assert f"Snippet from source {index}." in baseline_prompt
    # The multi-agent Writer works from distilled research instead.
    assert "Snippet from source 0." not in provider.prompts["writer"][0]


async def test_judge_is_blind_to_which_system_wrote_the_posts(settings):
    provider = StubProvider()
    await _evaluate(provider, settings)

    for prompt in provider.prompts["judge"]:
        lowered = prompt.lower()
        assert "multi" not in lowered
        assert "single" not in lowered
        assert "baseline" not in lowered


async def test_grounding_uses_neutral_research_not_the_researchers(settings):
    provider = StubProvider()
    await _evaluate(provider, settings)

    pipeline_check, *grounding_checks = provider.prompts["fact_checker"]
    assert "Sources broadly agree" in pipeline_check
    for prompt in grounding_checks:
        assert "Sources broadly agree" not in prompt
        assert "Snippet from source 0." in prompt


async def test_failed_judge_leaves_no_score_rather_than_zero(settings):
    result = await _evaluate(StubProvider(fail_on=["judge"]), settings)

    assert result.multi_agent.quality is None
    assert result.single_agent.quality is None
    assert len(result.evaluator_errors) == 2
    # Grounding still ran.
    assert result.multi_agent.grounding is not None


async def test_failed_baseline_is_recorded_and_not_scored(settings, monkeypatch):
    async def broken(*_args, **_kwargs):
        raise LLMProviderError("Provider unavailable.")

    monkeypatch.setattr(runner, "run_single_agent", broken)
    provider = StubProvider()
    result = await _evaluate(provider, settings)

    assert not result.single_agent.succeeded
    assert result.single_agent.error == "Provider unavailable."
    assert result.single_agent.quality is None
    assert provider.judge_calls == 1  # the multi-agent output only


async def test_no_sources_skips_the_baseline(settings):
    result = await _evaluate(StubProvider(), settings, search=StubSearchProvider(sources=[]))

    assert result.source_count == 0
    assert not result.single_agent.succeeded
    assert "no sources" in (result.single_agent.error or "")


async def test_rate_limits_are_waited_out(settings, monkeypatch):
    calls = 0
    real = runner.run_single_agent

    async def throttled(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMRateLimitError("Slow down.")
        return await real(*args, **kwargs)

    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(runner, "run_single_agent", throttled)
    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)

    result = await evaluate_topic(
        make_service(StubProvider()),
        make_research(),
        topic="AI in teams",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
        settings=settings,
        retrieval=None,
        pricing=PRICING,
        rate_limits=RateLimitPolicy(wait_seconds=42.0, retries=1),
    )

    assert result.single_agent.succeeded
    assert slept == [42.0]
    # The wait is not generation time.
    assert result.single_agent.latency_seconds < 42.0


async def test_rate_limit_retries_are_bounded(settings, monkeypatch):
    async def always_throttled(*_args, **_kwargs):
        raise LLMRateLimitError("Slow down.")

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(runner, "run_single_agent", always_throttled)
    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)

    result = await evaluate_topic(
        make_service(StubProvider()),
        make_research(),
        topic="AI in teams",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
        settings=settings,
        retrieval=None,
        pricing=PRICING,
        rate_limits=RateLimitPolicy(wait_seconds=1.0, retries=2),
    )
    assert result.single_agent.error == "Slow down."


async def test_transient_outages_are_waited_out_too(settings, monkeypatch):
    calls = 0
    real = runner.run_single_agent

    async def overloaded_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMProviderError("High demand.", details={"retryable": True})
        return await real(*args, **kwargs)

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(runner, "run_single_agent", overloaded_once)
    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)
    result = await _evaluate(StubProvider(), settings)
    assert result.single_agent.succeeded
    assert calls == 2


async def test_an_empty_workflow_says_why(settings):
    result = await _evaluate(StubProvider(fail_on=["planner"]), settings)

    assert not result.multi_agent.succeeded
    assert "planner" in (result.multi_agent.error or "")
    # The baseline still runs: the search succeeded.
    assert result.single_agent.succeeded


async def test_other_failures_are_not_retried(settings, monkeypatch):
    calls = 0

    async def broken(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise LLMProviderError("Bad key.")

    monkeypatch.setattr(runner, "run_single_agent", broken)
    await _evaluate(StubProvider(), settings)
    assert calls == 1


# --- the evaluators --------------------------------------------------------


def test_neutral_research_is_the_snippets_verbatim():
    sources = make_sources(2)
    research = neutral_research(sources)

    assert [fact.statement for fact in research.facts] == [s.snippet for s in sources]
    assert [fact.source_index for fact in research.facts] == [0, 1]
    assert research.sources == sources


def test_claim_free_output_has_no_supported_rate():
    assert Grounding(report=FactCheckReport(claims=[])).supported_rate is None
    mixed = Grounding(report=make_fact_check(verified=3, unsupported=1))
    assert mixed.supported_rate == 0.75
    assert mixed.unsupported == 1


# --- many topics -----------------------------------------------------------


async def test_evaluate_runs_every_topic_and_reports_progress(settings):
    seen: list[tuple[int, int, str]] = []
    report = await evaluate(
        make_service(StubProvider(judge_scores=[8.0, 6.0])),
        make_research(),
        topics=["Topic one", "Topic two"],
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
        settings=settings,
        retrieval=None,
        pricing=PRICING,
        on_progress=lambda i, n, t: seen.append((i, n, t)),
    )

    assert [r.topic for r in report.results] == ["Topic one", "Topic two"]
    assert seen == [(1, 2, "Topic one"), (2, 2, "Topic two")]
    assert report.retrieval_enabled is False

    multi = next(s for s in report.summaries if s.system is SystemName.MULTI_AGENT)
    assert multi.topics_completed == 2
    assert multi.mean_llm_calls == 5.0
    # Judge scores repeat their last value, so only the first topic is a win.
    assert multi.quality_wins == 1
    assert report.ties == 1


async def test_summary_excludes_failed_topics(settings, monkeypatch):
    async def broken(*_args, **_kwargs):
        raise LLMProviderError("Provider unavailable.")

    monkeypatch.setattr(runner, "run_single_agent", broken)
    result = await _evaluate(StubProvider(), settings)

    single = summarise(SystemName.SINGLE_AGENT, [result])
    assert single.topics_completed == 0
    assert single.topics_failed == 1
    assert single.mean_quality is None
    assert single.mean_tokens is None


async def test_report_renders_and_round_trips(settings):
    report = await evaluate(
        make_service(StubProvider()),
        make_research(),
        topics=["A | piped topic"],
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
        settings=settings,
        retrieval=None,
        pricing=None,
    )

    markdown = render_markdown(report)
    assert "# Single-agent vs multi-agent evaluation" in markdown
    assert "| Mean quality (0-10) |" in markdown
    assert "A \\| piped topic" in markdown  # a pipe would break the table
    assert "costs not estimated" in markdown

    restored = EvaluationReport.model_validate_json(
        report.model_dump_json(exclude_computed_fields=True)
    )
    assert restored.results[0].topic == "A | piped topic"


# --- command line ----------------------------------------------------------


def test_topics_file_must_be_a_list_of_strings(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(["one", "two", "three"]), encoding="utf-8")
    assert _load_topics(good, 2) == ["one", "two"]

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"topic": "one"}), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_topics(bad, None)


def test_post_count_is_bounded():
    assert _parse_args(["--posts", "3"]).posts == 3
    with pytest.raises(SystemExit):
        _parse_args(["--posts", "9"])
