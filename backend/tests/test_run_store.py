"""Phase 7: SQLite run history and the aggregates computed over it."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.core.config import Settings
from app.models.enums import Platform, Tone
from app.models.events import EventType, WorkflowEvent
from app.schemas.workflow import WorkflowRunResponse, build_workflow_response
from app.services.pricing import ModelPricing
from app.services.run_store import RunStore, get_run_store
from tests.conftest import StubProvider, make_fact_check, make_verdict, run_pipeline

RATES = ModelPricing(model="stub-model", input_per_mtok=1.0, output_per_mtok=2.0, source="test")
T0 = datetime(2026, 11, 1, 9, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path) -> RunStore:
    return RunStore(tmp_path / "history.db")


async def _result(settings, run_kwargs, provider=None) -> WorkflowRunResponse:
    state = await run_pipeline(provider or StubProvider(), settings, run_kwargs)
    return build_workflow_response(state, RATES)


def _events(count: int = 3) -> list[WorkflowEvent]:
    return [
        WorkflowEvent(sequence=i, type=EventType.WORKFLOW_STARTED, timestamp=T0)
        for i in range(count)
    ]


def _save(store: RunStore, run_id: str, *, result=None, created=T0, state="completed", **extra):
    store.save(
        run_id=run_id,
        topic=f"Topic {run_id}",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
        state=state,
        created_at=created,
        finished_at=created + timedelta(seconds=30),
        provider="stub",
        model="stub-model",
        result=result,
        events=_events(),
        **extra,
    )


async def test_a_finished_run_round_trips(store, settings, run_kwargs):
    # The fact check carries computed fields, which must not break reloading.
    provider = StubProvider(fact_checks=[make_fact_check(verified=2)])
    result = await _result(settings, run_kwargs, provider)
    _save(store, "abc", result=result)

    stored = store.get("abc")
    assert stored is not None
    assert stored.result == result
    assert stored.result.fact_check.overall_status == result.fact_check.overall_status
    assert [e.sequence for e in stored.events] == [0, 1, 2]

    summary = stored.summary
    assert summary.topic == "Topic abc"
    assert summary.duration_seconds == 30.0
    assert summary.approved is True
    assert summary.quality_score == result.quality_score
    assert summary.cost_usd == result.cost.total_usd
    assert summary.total_tokens == sum(
        m.usage.total_tokens for m in result.agent_metrics if m.usage
    )


def test_a_crashed_run_is_stored_without_a_result(store):
    _save(store, "dead", state="failed", error_type="workflow_error", error_message="Boom.")

    stored = store.get("dead")
    assert stored is not None
    assert stored.result is None
    assert stored.error_message == "Boom."
    assert stored.summary.workflow_status is None
    assert stored.summary.approved is None


def test_unknown_runs_are_none(store):
    assert store.get("missing") is None


def test_history_lists_newest_first_with_paging(store):
    for index in range(5):
        _save(store, f"r{index}", created=T0 + timedelta(minutes=index))

    assert [run.run_id for run in store.list(limit=2)] == ["r4", "r3"]
    assert [run.run_id for run in store.list(limit=2, offset=2)] == ["r2", "r1"]
    assert store.count() == 5


async def test_saving_again_replaces_rather_than_duplicates(store, settings, run_kwargs):
    result = await _result(settings, run_kwargs)
    _save(store, "same", result=result)
    _save(store, "same", result=result)

    assert store.count() == 1
    executions = sum(agent.executions for agent in store.summary().agents)
    assert executions == len(result.agent_metrics)


async def test_summary_averages_only_runs_with_a_verdict(store, settings, run_kwargs):
    approved = await _result(settings, run_kwargs)
    rejected = await _result(
        settings, run_kwargs, StubProvider(verdicts=[make_verdict(approved=False, score=5.0)])
    )
    _save(store, "good", result=approved)
    _save(store, "capped", result=rejected, created=T0 + timedelta(minutes=1))
    _save(store, "crashed", state="failed", created=T0 + timedelta(minutes=2))

    summary = store.summary()
    assert summary.total_runs == 3
    assert summary.runs_by_state == {"completed": 2, "failed": 1}
    # The crashed run has no verdict, so it is not counted as unapproved.
    assert summary.approval_rate == 0.5
    assert summary.mean_quality == pytest.approx((approved.quality_score + 5.0) / 2)
    assert summary.mean_revisions == pytest.approx(rejected.revision_count / 2)
    assert summary.total_cost_usd == pytest.approx(
        approved.cost.total_usd + rejected.cost.total_usd, abs=1e-6
    )
    assert summary.first_run_at == T0
    assert summary.last_run_at == T0 + timedelta(minutes=2)


async def test_agent_aggregates_follow_graph_order(store, settings, run_kwargs):
    _save(store, "one", result=await _result(settings, run_kwargs))

    agents = store.summary().agents
    names = [aggregate.agent.value for aggregate in agents]
    assert names[0] == "retriever"
    assert names[-1] == "finalize"

    retriever = agents[0]
    assert retriever.mean_tokens is None  # makes no LLM call
    writer = next(a for a in agents if a.agent.value == "writer")
    assert writer.failure_rate == 0.0
    assert writer.mean_tokens == 350.0


def test_empty_history_summarises_to_nothing(store):
    summary = store.summary()
    assert summary.total_runs == 0
    assert summary.approval_rate is None
    assert summary.agents == []


def test_history_can_be_switched_off(tmp_path):
    settings = Settings(run_store_enabled=False, run_store_path=tmp_path / "x.db")
    assert get_run_store(settings) is None
    assert not (tmp_path / "x.db").exists()


def test_an_unusable_path_disables_history_instead_of_raising(tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory", encoding="utf-8")
    settings = Settings(run_store_path=blocker / "runs.db")
    assert get_run_store(settings) is None
