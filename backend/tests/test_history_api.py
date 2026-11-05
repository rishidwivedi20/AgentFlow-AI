"""Phase 7: history and metrics endpoints, and runs outliving the registry."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from app.services.run_registry import reset_run_registry
from tests.conftest import StubProvider, make_research, make_service, make_verdict
from tests.test_streaming import VALID_BODY, read_sse


def _settings(**overrides) -> Settings:
    return Settings(
        gemini_api_key="test-key",
        tavily_api_key="test-search-key",
        max_revisions=3,
        app_env="test",
        **overrides,
    )


@pytest.fixture
def client(monkeypatch):
    provider = StubProvider(verdicts=[make_verdict(approved=True, score=9.0)])
    monkeypatch.setattr("app.routes.runs.build_llm_service", lambda _s=None: make_service(provider))
    monkeypatch.setattr("app.routes.runs.build_research_service", lambda _s=None: make_research())
    reset_run_registry()
    app.dependency_overrides[get_settings] = lambda: _settings()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    reset_run_registry()


def _finish_run(client: TestClient) -> str:
    """Start a run, follow it to the end, and wait for it to reach history.

    The stream closes the moment the run completes; the history write follows
    a moment later on the same task.
    """
    run_id = client.post("/api/workflows", json=VALID_BODY).json()["run_id"]
    client.get(f"/api/workflows/{run_id}/events")
    for _ in range(100):
        if any(run["run_id"] == run_id for run in client.get("/api/workflows").json()["runs"]):
            return run_id
        time.sleep(0.02)
    raise AssertionError("run never reached history")


def test_finished_runs_appear_in_history(client):
    run_id = _finish_run(client)

    body = client.get("/api/workflows").json()
    assert body["history_enabled"] is True
    assert body["total"] == 1
    (run,) = body["runs"]
    assert run["run_id"] == run_id
    assert run["topic"] == VALID_BODY["topic"]
    assert run["state"] == "completed"
    assert run["approved"] is True


def test_history_paging_is_validated(client):
    assert client.get("/api/workflows?limit=0").status_code == 422
    assert client.get("/api/workflows?offset=-1").status_code == 422


def test_a_run_survives_the_registry_forgetting_it(client):
    run_id = _finish_run(client)
    live = client.get(f"/api/workflows/{run_id}").json()

    reset_run_registry()  # as after a restart or eviction

    stored = client.get(f"/api/workflows/{run_id}").json()
    assert stored["state"] == "completed"
    assert stored["result"]["final_posts"] == live["result"]["final_posts"]
    assert stored["result"]["fact_check"]["overall_status"] == (
        live["result"]["fact_check"]["overall_status"]
    )
    assert stored["event_count"] == live["event_count"]


def test_stored_events_replay_from_the_cursor(client):
    run_id = _finish_run(client)
    reset_run_registry()

    response = client.get(f"/api/workflows/{run_id}/events")
    assert response.status_code == 200
    events = read_sse(response.text)
    assert events[0]["type"] == "workflow_started"
    assert events[-1]["type"] == "workflow_completed"

    resumed = read_sse(client.get(f"/api/workflows/{run_id}/events?cursor=2").text)
    assert [e["sequence"] for e in resumed] == [e["sequence"] for e in events if e["sequence"] > 2]


def test_unknown_runs_are_still_404(client):
    assert client.get("/api/workflows/nope").status_code == 404
    assert client.get("/api/workflows/nope/events").status_code == 404


def test_metrics_summary_aggregates_history(client):
    _finish_run(client)
    _finish_run(client)

    body = client.get("/api/metrics/summary").json()
    assert body["history_enabled"] is True
    summary = body["summary"]
    assert summary["total_runs"] == 2
    assert summary["approval_rate"] == 1.0
    assert {a["agent"] for a in summary["agents"]} >= {"writer", "editor", "fact_checker"}
    # gemini-3.6-flash is in the price table.
    assert body["pricing"]["model"] == "gemini-3.6-flash"
    assert summary["total_cost_usd"] > 0


def test_history_switched_off(client):
    app.dependency_overrides[get_settings] = lambda: _settings(run_store_enabled=False)

    assert client.get("/api/workflows").json() == {
        "history_enabled": False,
        "total": 0,
        "limit": 50,
        "offset": 0,
        "runs": [],
    }
    body = client.get("/api/metrics/summary").json()
    assert body["history_enabled"] is False
    assert body["summary"] is None
