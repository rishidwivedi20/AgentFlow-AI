"""Planner behaviour and the prerequisite routing that guards the pipeline."""

from __future__ import annotations

from app.agents.planner import PLANNER_TEMPERATURE, run_planner
from app.agents.prompts import planner as planner_prompts
from app.graph.routing import (
    PlanRoute,
    ResearchRoute,
    route_after_planner,
    route_after_researcher,
)
from app.graph.state import initial_state
from app.models.enums import AgentName, Platform, Tone, WorkflowStatus
from app.models.plan import ContentPlan, ContentType
from tests.conftest import (
    StubProvider,
    StubSearchProvider,
    make_plan,
    make_research_result,
    make_service,
    run_pipeline,
)


def _state():
    return initial_state(
        topic="topic",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
        max_revisions=3,
    )


# --- prompts ----------------------------------------------------------------


def test_planner_prompt_demands_variety_not_rewording():
    prompt = planner_prompts.build_system_prompt(Platform.LINKEDIN, Tone.PROFESSIONAL)
    assert "VARIETY" in prompt
    for content_type in ContentType:
        assert content_type.value in prompt


def test_planner_prompt_restricts_points_to_the_research():
    prompt = planner_prompts.build_user_prompt(
        topic="AI", research=make_research_result(), number_of_posts=3
    )
    assert "exactly 3 angle(s)" in prompt
    assert "RESEARCH" in prompt


def test_plan_renders_a_brief_the_writer_can_follow():
    brief = make_plan(2).as_writer_brief()
    assert "POST 1" in brief
    assert "POST 2" in brief
    assert "angle:" in brief


# --- agent ------------------------------------------------------------------


async def test_planner_requests_the_plan_schema():
    class Recording(StubProvider):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.captured = []

        async def generate_structured(self, **kwargs):
            self.captured.append(kwargs)
            return await super().generate_structured(**kwargs)

    provider = Recording()
    result = await run_planner(
        make_service(provider),
        topic="AI",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        research=make_research_result(),
        number_of_posts=2,
    )
    assert isinstance(result.value, ContentPlan)
    assert provider.captured[0]["schema"] is ContentPlan
    assert provider.captured[0]["temperature"] == PLANNER_TEMPERATURE
    # Variety needs headroom the near-deterministic critics do not.
    assert PLANNER_TEMPERATURE > 0.5


async def test_angle_count_mismatch_is_logged_not_fatal(caplog):
    provider = StubProvider(angle_count=2)
    result = await run_planner(
        make_service(provider),
        topic="AI",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        research=make_research_result(),
        number_of_posts=5,
    )
    assert len(result.value.angles) == 2
    assert "unexpected angle count" in caplog.text


# --- prerequisite routing ---------------------------------------------------


def test_missing_research_stops_the_run():
    """Writing without research would make every claim unverifiable."""
    assert route_after_researcher(_state()) is ResearchRoute.FINALIZE


def test_research_present_continues_to_planning():
    state = _state()
    state["research"] = make_research_result()
    assert route_after_researcher(state) is ResearchRoute.PLAN


def test_missing_plan_stops_the_run():
    assert route_after_planner(_state()) is PlanRoute.FINALIZE


def test_plan_present_continues_to_writing():
    state = _state()
    state["content_plan"] = make_plan(2)
    assert route_after_planner(state) is PlanRoute.WRITE


# --- end to end -------------------------------------------------------------


async def test_search_failure_ends_the_run_without_writing(settings, run_kwargs):
    provider = StubProvider()
    state = await run_pipeline(
        provider, settings, run_kwargs, search=StubSearchProvider(fail=True)
    )

    assert state["workflow_status"] is WorkflowStatus.FAILED
    assert state["final_output"] == []
    assert provider.writer_calls == 0, "nothing may be written without research"
    # The Retriever owns the search since Phase 5.
    assert state["errors"][0].agent is AgentName.RETRIEVER


async def test_planner_failure_ends_the_run_without_writing(settings, run_kwargs):
    provider = StubProvider(fail_on=["planner"])
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["workflow_status"] is WorkflowStatus.FAILED
    assert provider.writer_calls == 0
    assert state["errors"][0].agent is AgentName.PLANNER


async def test_empty_search_results_still_produce_content(settings, run_kwargs):
    """No sources is not a failure - the run proceeds, ungrounded but honest."""
    provider = StubProvider()
    state = await run_pipeline(
        provider, settings, run_kwargs, search=StubSearchProvider(sources=[])
    )

    assert state["research"] is not None
    assert state["research"].sources == []
    assert provider.researcher_calls == 0, "no sources means nothing to extract"
    assert provider.writer_calls >= 1
    assert state["final_output"]


async def test_research_and_plan_reach_the_writer(settings, run_kwargs):
    provider = StubProvider()
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["research_context"], "the Writer's grounding text must be populated"
    assert state["content_plan"] is not None
    assert state["research"].sources
