"""Event emission, the SSE endpoints, and what events may never contain."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from app.models.enums import AgentName, AgentStatus, WorkflowStatus
from app.models.events import EventType, WorkflowEvent
from app.routes.runs import _format_sse, _resolve_cursor
from app.services.event_bus import NullEmitter, RunEventEmitter
from app.services.run_registry import RunRegistry, RunState, get_run_registry, reset_run_registry
from tests.conftest import (
    StubProvider,
    StubSearchProvider,
    make_fact_check,
    make_research,
    make_service,
    make_verdict,
)

VALID_BODY = {
    "topic": "How AI is changing software engineering",
    "platform": "linkedin",
    "tone": "professional",
    "number_of_posts": 2,
}


def make_settings() -> Settings:
    return Settings(
        gemini_api_key="test-key",
        tavily_api_key="test-search-key",
        max_revisions=3,
        app_env="test",
    )


@pytest.fixture
def registry():
    reset_run_registry()
    yield get_run_registry()
    reset_run_registry()


@pytest.fixture
def client(monkeypatch, registry):
    """Client whose runs use stub providers. No network, no API key."""
    provider = StubProvider(verdicts=[make_verdict(approved=True, score=9.0)])

    monkeypatch.setattr(
        "app.routes.runs.build_llm_service", lambda _s=None: make_service(provider)
    )
    monkeypatch.setattr(
        "app.routes.runs.build_research_service", lambda _s=None: make_research()
    )
    # The synchronous endpoint builds its services from its own module.
    monkeypatch.setattr(
        "app.routes.workflows.build_llm_service", lambda _s=None: make_service(provider)
    )
    monkeypatch.setattr(
        "app.routes.workflows.build_research_service", lambda _s=None: make_research()
    )
    app.dependency_overrides[get_settings] = make_settings
    with TestClient(app) as c:
        c.provider = provider
        yield c
    app.dependency_overrides.clear()


def read_sse(text: str) -> list[dict]:
    """Parse an SSE response body into its data payloads."""
    events = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


# --- emitter ----------------------------------------------------------------


async def test_null_emitter_discards_everything():
    """The default: streaming must cost nothing when unused."""
    assert await NullEmitter().emit(EventType.WORKFLOW_STARTED) is None


async def test_emitter_assigns_gapless_sequences():
    registry = RunRegistry()
    record = registry.create(topic="t")
    emitter = RunEventEmitter(record.run_id, registry)

    await emitter.emit(EventType.WORKFLOW_STARTED)
    await emitter.emit(EventType.AGENT_STARTED, agent=AgentName.WRITER)
    await emitter.emit(EventType.AGENT_COMPLETED, agent=AgentName.WRITER)

    assert [e.sequence for e in record.events] == [0, 1, 2]
    assert emitter.next_sequence == 3


async def test_emitting_to_an_unknown_run_is_harmless():
    """A late event must not raise into a finished run's teardown."""
    registry = RunRegistry()
    await RunEventEmitter("ghost", registry).emit(EventType.WORKFLOW_STARTED)


# --- the graph emits a correct, ordered lifecycle ---------------------------


async def test_workflow_emits_an_ordered_lifecycle():
    registry = RunRegistry()
    record = registry.create(topic="t")
    emitter = RunEventEmitter(record.run_id, registry)

    from app.graph.workflow import run_workflow
    from app.models.enums import Platform, Tone

    await run_workflow(
        make_service(StubProvider()),
        make_research(),
        topic="t",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
        settings=make_settings(),
        emitter=emitter,
    )

    types = [e.type for e in record.events]
    assert types[0] is EventType.WORKFLOW_STARTED
    assert types[-1] is EventType.WORKFLOW_COMPLETED
    assert [e.sequence for e in record.events] == list(range(len(record.events)))

    # Each agent is started then completed, in order.
    agents = [(e.type, e.agent) for e in record.events if e.agent is not None]
    assert agents[0] == (EventType.AGENT_STARTED, AgentName.RETRIEVER)
    assert agents[1] == (EventType.AGENT_COMPLETED, AgentName.RETRIEVER)
    assert agents[2] == (EventType.AGENT_STARTED, AgentName.RESEARCHER)
    assert record.events[-1].workflow_status is WorkflowStatus.COMPLETED


async def test_repeated_writer_runs_are_distinguishable():
    """A second writer event with a higher execution index is the loop signal."""
    registry = RunRegistry()
    record = registry.create(topic="t")
    emitter = RunEventEmitter(record.run_id, registry)

    from app.graph.workflow import run_workflow
    from app.models.enums import Platform, Tone

    await run_workflow(
        make_service(
            StubProvider(
                verdicts=[
                    make_verdict(approved=False, score=5.0),
                    make_verdict(approved=True, score=9.0),
                ]
            )
        ),
        make_research(),
        topic="t",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
        settings=make_settings(),
        emitter=emitter,
    )

    writer_starts = [
        e
        for e in record.events
        if e.type is EventType.AGENT_STARTED and e.agent is AgentName.WRITER
    ]
    assert len(writer_starts) == 2
    first, second = (e.agent_execution for e in writer_starts)
    assert first is not None and second is not None
    assert second > first, "each writer pass must be individually identifiable"


async def test_failed_agent_emits_a_failure_event_without_erasing_history():
    registry = RunRegistry()
    record = registry.create(topic="t")
    emitter = RunEventEmitter(record.run_id, registry)

    from app.graph.workflow import run_workflow
    from app.models.enums import Platform, Tone

    await run_workflow(
        make_service(StubProvider(fail_on=["editor"])),
        make_research(),
        topic="t",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
        settings=make_settings(),
        emitter=emitter,
    )

    failures = [e for e in record.events if e.type is EventType.AGENT_FAILED]
    assert len(failures) == 1
    assert failures[0].agent is AgentName.EDITOR
    assert failures[0].status is AgentStatus.FAILED
    assert failures[0].message

    # Everything that succeeded before the failure is still in the stream.
    completed = {e.agent for e in record.events if e.type is EventType.AGENT_COMPLETED}
    assert AgentName.RESEARCHER in completed
    assert AgentName.PLANNER in completed
    assert AgentName.WRITER in completed
    assert record.events[-1].type is EventType.WORKFLOW_COMPLETED


async def test_events_never_leak_prompts_or_internals():
    """The safety contract, asserted rather than assumed."""
    registry = RunRegistry()
    record = registry.create(topic="t")
    emitter = RunEventEmitter(record.run_id, registry)

    from app.graph.workflow import run_workflow
    from app.models.enums import Platform, Tone

    await run_workflow(
        make_service(StubProvider(fail_on=["writer"])),
        make_research(),
        topic="t",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
        settings=make_settings(),
        emitter=emitter,
    )

    blob = json.dumps([json.loads(e.model_dump_json()) for e in record.events])
    for forbidden in ("Traceback", "system_prompt", "user_prompt", "test-key", "api_key"):
        assert forbidden not in blob, f"{forbidden!r} leaked into the event stream"

    # Only the declared fields exist - no accidental payload passthrough.
    assert set(WorkflowEvent.model_fields) == {
        "sequence",
        "type",
        "timestamp",
        "agent",
        "status",
        "agent_execution",
        "latency_seconds",
        "usage",
        "error_type",
        "message",
        "workflow_status",
    }


# --- SSE formatting and cursor resolution ----------------------------------


def test_sse_frame_carries_id_event_and_data():
    frame = _format_sse(
        WorkflowEvent(
            sequence=7,
            type=EventType.AGENT_COMPLETED,
            timestamp="2026-01-01T00:00:00Z",
            agent=AgentName.WRITER,
        )
    )
    assert frame.startswith("id: 7\n")
    assert "event: agent_completed\n" in frame
    assert frame.endswith("\n\n")
    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload["sequence"] == 7


@pytest.mark.parametrize(
    ("header", "query", "expected"),
    [
        (None, None, -1),
        (None, 4, 4),
        ("9", None, 9),
        ("9", 4, 9),  # header wins: reconnects reuse the original query string
        ("garbage", 4, 4),  # malformed header falls back
        ("garbage", None, -1),
    ],
)
def test_cursor_resolution(header, query, expected):
    assert _resolve_cursor(header, query) == expected


# --- endpoints --------------------------------------------------------------


def test_create_run_returns_202_with_urls(client):
    response = client.post("/api/workflows", json=VALID_BODY)
    assert response.status_code == 202
    body = response.json()
    assert body["state"] == RunState.QUEUED.value
    assert body["events_url"] == f"/api/workflows/{body['run_id']}/events"
    assert body["status_url"] == f"/api/workflows/{body['run_id']}"


def test_create_run_validates_input(client):
    assert client.post("/api/workflows", json={**VALID_BODY, "topic": "  "}).status_code == 422
    assert (
        client.post("/api/workflows", json={**VALID_BODY, "number_of_posts": 9}).status_code == 422
    )


def test_synchronous_endpoint_is_unchanged(client):
    """Phase 3's endpoint must keep working exactly as before."""
    response = client.post("/api/workflows/run", json=VALID_BODY)
    assert response.status_code == 200
    assert response.json()["workflow_status"] == WorkflowStatus.COMPLETED.value


def test_unknown_run_is_404(client):
    for path in ("/api/workflows/nope", "/api/workflows/nope/events"):
        response = client.get(path)
        assert response.status_code == 404
        assert "retained for a limited time" in response.json()["error"]["message"]


def test_run_reaches_completion_and_returns_its_result(client):
    run_id = client.post("/api/workflows", json=VALID_BODY).json()["run_id"]
    # Consuming the stream to completion also drives the background task.
    client.get(f"/api/workflows/{run_id}/events")

    body = client.get(f"/api/workflows/{run_id}").json()
    assert body["state"] == RunState.COMPLETED.value
    assert body["result"]["workflow_status"] == WorkflowStatus.COMPLETED.value
    assert body["event_count"] > 0
    assert body["finished_at"]


def test_stream_delivers_the_whole_lifecycle(client):
    run_id = client.post("/api/workflows", json=VALID_BODY).json()["run_id"]
    response = client.get(f"/api/workflows/{run_id}/events")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"

    events = read_sse(response.text)
    assert events[0]["type"] == EventType.WORKFLOW_STARTED.value
    assert events[-1]["type"] == EventType.WORKFLOW_COMPLETED.value
    assert [e["sequence"] for e in events] == list(range(len(events)))


def test_stream_replays_from_a_cursor(client):
    run_id = client.post("/api/workflows", json=VALID_BODY).json()["run_id"]
    everything = read_sse(client.get(f"/api/workflows/{run_id}/events").text)

    resumed = read_sse(client.get(f"/api/workflows/{run_id}/events?cursor=2").text)
    assert [e["sequence"] for e in resumed] == [e["sequence"] for e in everything[3:]]


def test_stream_honours_last_event_id_over_the_query(client):
    run_id = client.post("/api/workflows", json=VALID_BODY).json()["run_id"]
    everything = read_sse(client.get(f"/api/workflows/{run_id}/events").text)

    resumed = read_sse(
        client.get(
            f"/api/workflows/{run_id}/events?cursor=0", headers={"Last-Event-ID": "3"}
        ).text
    )
    assert [e["sequence"] for e in resumed] == [e["sequence"] for e in everything[4:]]


def test_finished_run_can_still_be_streamed(client):
    """Refreshing the page after a run ends must replay the whole thing."""
    run_id = client.post("/api/workflows", json=VALID_BODY).json()["run_id"]
    first = read_sse(client.get(f"/api/workflows/{run_id}/events").text)
    second = read_sse(client.get(f"/api/workflows/{run_id}/events").text)
    assert [e["sequence"] for e in first] == [e["sequence"] for e in second]


def test_missing_configuration_is_rejected_before_a_run_is_created(monkeypatch, registry):
    """A missing key must not create a run that exists only to fail."""
    app.dependency_overrides[get_settings] = lambda: Settings(
        gemini_api_key="", tavily_api_key="", app_env="test"
    )
    try:
        with TestClient(app) as c:
            response = c.post("/api/workflows", json=VALID_BODY)
        assert response.status_code == 503
        assert response.json()["error"]["type"] == "configuration_error"
        assert len(registry) == 0, "no run should have been created"
    finally:
        app.dependency_overrides.clear()


def test_search_failure_is_recorded_on_the_run(monkeypatch, registry):
    provider = StubProvider()
    monkeypatch.setattr(
        "app.routes.runs.build_llm_service", lambda _s=None: make_service(provider)
    )
    monkeypatch.setattr(
        "app.routes.runs.build_research_service",
        lambda _s=None: make_research(StubSearchProvider(fail=True)),
    )
    app.dependency_overrides[get_settings] = make_settings
    try:
        with TestClient(app) as c:
            run_id = c.post("/api/workflows", json=VALID_BODY).json()["run_id"]
            c.get(f"/api/workflows/{run_id}/events")
            body = c.get(f"/api/workflows/{run_id}").json()

        # The run itself completed; the workflow degraded gracefully.
        assert body["state"] == RunState.COMPLETED.value
        assert body["result"]["workflow_status"] == WorkflowStatus.FAILED.value
        # The search belongs to the Retriever since Phase 5, so that is where
        # a search failure is now attributed.
        assert body["result"]["errors"][0]["agent"] == "retriever"
    finally:
        app.dependency_overrides.clear()


def test_unsupported_claims_run_streams_and_reports(monkeypatch, registry):
    provider = StubProvider(
        verdicts=[make_verdict(approved=True, score=9.0)],
        fact_checks=[make_fact_check(unsupported=1)],
    )
    monkeypatch.setattr(
        "app.routes.runs.build_llm_service", lambda _s=None: make_service(provider)
    )
    monkeypatch.setattr(
        "app.routes.runs.build_research_service", lambda _s=None: make_research()
    )
    app.dependency_overrides[get_settings] = make_settings
    try:
        with TestClient(app) as c:
            run_id = c.post("/api/workflows", json=VALID_BODY).json()["run_id"]
            events = read_sse(c.get(f"/api/workflows/{run_id}/events").text)
            body = c.get(f"/api/workflows/{run_id}").json()

        assert events[-1]["workflow_status"] == WorkflowStatus.COMPLETED_UNVERIFIED.value
        assert body["result"]["warnings"]
    finally:
        app.dependency_overrides.clear()
