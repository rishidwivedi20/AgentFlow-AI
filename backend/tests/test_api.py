"""API contract: input validation, status codes and response shape."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from app.models.enums import WorkflowStatus
from app.services.llm_service import LLMService
from tests.conftest import StubProvider, make_research, make_verdict

VALID_BODY = {
    "topic": "How AI is changing software engineering",
    "platform": "linkedin",
    "tone": "professional",
    "number_of_posts": 2,
}


@pytest.fixture
def client(monkeypatch):
    """Client with the LLM service replaced by a stub. No network, no API key."""
    provider = StubProvider(verdicts=[make_verdict(approved=True, score=9.0)])

    def fake_build(_settings=None):
        return LLMService(provider, timeout_seconds=5.0, max_retries=0)

    monkeypatch.setattr("app.routes.workflows.build_llm_service", fake_build)
    monkeypatch.setattr(
        "app.routes.workflows.build_research_service", lambda _s=None: make_research()
    )
    app.dependency_overrides[get_settings] = lambda: Settings(
        gemini_api_key="test-key",
        tavily_api_key="test-search-key",
        max_revisions=3,
        app_env="test",
    )
    with TestClient(app) as c:
        c.provider = provider
        yield c
    app.dependency_overrides.clear()


# --- validation ------------------------------------------------------------

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("topic", ""),
        ("topic", "   "),          # whitespace-only passes min_length, must still fail
        ("topic", "ab"),
        ("platform", "twitter"),
        ("tone", "sarcastic"),
        ("number_of_posts", 0),
        ("number_of_posts", 6),
        ("number_of_posts", "two"),
    ],
)
def test_invalid_input_is_rejected(client, field, value):
    response = client.post("/api/workflows/run", json={**VALID_BODY, field: value})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["type"] == "validation_error"
    assert body["error"]["details"]["fields"]


def test_missing_body_is_rejected(client):
    assert client.post("/api/workflows/run", json={}).status_code == 422


def test_topic_is_stripped(client):
    response = client.post("/api/workflows/run", json={**VALID_BODY, "topic": "  Spaced topic  "})
    assert response.status_code == 200
    assert response.json()["topic"] == "Spaced topic"


def test_number_of_posts_defaults(client):
    body = {k: v for k, v in VALID_BODY.items() if k != "number_of_posts"}
    assert client.post("/api/workflows/run", json=body).status_code == 200


# --- successful run --------------------------------------------------------

def test_successful_run_returns_full_payload(client):
    response = client.post("/api/workflows/run", json=VALID_BODY)
    assert response.status_code == 200

    body = response.json()
    assert body["workflow_status"] == WorkflowStatus.COMPLETED.value
    assert body["revision_count"] == 0
    assert body["max_revisions"] == 3
    assert body["approved"] is True
    assert body["quality_score"] == 9.0
    assert len(body["final_posts"]) == 2
    assert body["errors"] == []

    # Every required observability field is present.
    assert {m["agent"] for m in body["agent_metrics"]} == {
        "researcher",
        "planner",
        "writer",
        "editor",
        "fact_checker",
        "finalize",
    }
    assert all("latency_seconds" in m for m in body["agent_metrics"])
    assert body["editor_feedback"]["quality_score"] == 9.0

    # Phase 3 additions: research, sources and verification reach the client.
    assert body["research_summary"]
    assert body["sources"], "real sources must be returned"
    assert all(s["url"].startswith("http") for s in body["sources"])
    assert body["content_plan"]["angles"]
    assert body["fact_check"] is not None


def test_post_shape(client):
    post = client.post("/api/workflows/run", json=VALID_BODY).json()["final_posts"][0]
    assert set(post) == {"angle", "content", "hashtags"}


# --- failure semantics -----------------------------------------------------

def test_failed_run_returns_200_with_structured_errors(monkeypatch):
    """A run that executes but fails is a 200 carrying its metrics and errors."""
    provider = StubProvider(fail_on=["writer"])
    monkeypatch.setattr(
        "app.routes.workflows.build_llm_service",
        lambda _s=None: LLMService(provider, timeout_seconds=5.0, max_retries=0),
    )
    monkeypatch.setattr(
        "app.routes.workflows.build_research_service", lambda _s=None: make_research()
    )
    app.dependency_overrides[get_settings] = lambda: Settings(
        gemini_api_key="test-key",
        tavily_api_key="test-search-key",
        max_revisions=3,
        app_env="test",
    )
    try:
        with TestClient(app) as c:
            response = c.post("/api/workflows/run", json=VALID_BODY)
        assert response.status_code == 200
        body = response.json()
        assert body["workflow_status"] == WorkflowStatus.FAILED.value
        assert body["final_posts"] == []
        assert body["errors"][0]["agent"] == "writer"
        assert "Traceback" not in body["errors"][0]["message"]
        writer_metric = next(m for m in body["agent_metrics"] if m["agent"] == "writer")
        assert writer_metric["status"] == "failed"
    finally:
        app.dependency_overrides.clear()


def test_missing_api_key_returns_503_before_running(monkeypatch):
    """Configuration problems are non-2xx and cost no LLM call."""
    app.dependency_overrides[get_settings] = lambda: Settings(
        gemini_api_key="",
        groq_api_key="",
        tavily_api_key="",
        llm_provider="gemini",
        app_env="test",
    )
    try:
        with TestClient(app) as c:
            response = c.post("/api/workflows/run", json=VALID_BODY)
        assert response.status_code == 503
        assert response.json()["error"]["type"] == "configuration_error"
    finally:
        app.dependency_overrides.clear()


def test_health_still_works(client):
    assert client.get("/api/health").status_code == 200
