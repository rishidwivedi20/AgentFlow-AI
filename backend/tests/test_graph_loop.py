"""End-to-end graph behaviour with a stubbed provider.

These cover the requirement list: approval on first draft, rejection then
revision, termination on approval, termination at the revision cap, and agent
failure handling.
"""

from __future__ import annotations

from app.graph.workflow import recursion_limit_for
from app.models.enums import AgentName, AgentStatus, WorkflowStatus
from tests.conftest import StubProvider, make_verdict, run_pipeline


async def test_editor_approves_first_draft(approving_provider, settings, run_kwargs):
    state = await run_pipeline(approving_provider, settings, run_kwargs)

    assert state["workflow_status"] is WorkflowStatus.COMPLETED
    assert state["approved"] is True
    assert state["revision_count"] == 0
    assert approving_provider.writer_calls == 1
    assert approving_provider.editor_calls == 1
    assert len(state["final_output"]) == 2
    assert state["errors"] == []


async def test_rejection_triggers_revision_then_approval(settings, run_kwargs):
    provider = StubProvider(
        verdicts=[
            make_verdict(approved=False, score=5.0),
            make_verdict(approved=True, score=8.6),
        ]
    )
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["workflow_status"] is WorkflowStatus.COMPLETED
    assert state["revision_count"] == 1
    assert provider.writer_calls == 2
    assert provider.editor_calls == 2
    assert len(state["drafts"]) == 2


async def test_terminates_at_revision_cap(rejecting_provider, settings, run_kwargs):
    state = await run_pipeline(rejecting_provider, settings, run_kwargs)

    assert state["workflow_status"] is WorkflowStatus.MAX_REVISIONS_REACHED
    assert state["approved"] is False
    assert state["revision_count"] == settings.max_revisions == 3
    # 1 initial draft + 3 revisions
    assert rejecting_provider.writer_calls == 4
    assert rejecting_provider.editor_calls == 4
    assert len(state["drafts"]) == 4
    assert state["final_output"], "a best draft must still be returned"


async def test_cap_returns_best_scoring_draft_not_the_last(settings, run_kwargs):
    """The final revision is not necessarily the best one."""
    provider = StubProvider(
        verdicts=[
            make_verdict(approved=False, score=4.0),
            make_verdict(approved=False, score=7.2),  # best
            make_verdict(approved=False, score=3.0),
            make_verdict(approved=False, score=5.0),
        ]
    )
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["workflow_status"] is WorkflowStatus.MAX_REVISIONS_REACHED
    best = max(state["drafts"], key=lambda d: d.quality_score)
    assert best.quality_score == 7.2
    assert best.revision == 1
    assert state["final_output"] == best.posts


async def test_max_revisions_of_one_is_respected(run_kwargs):
    from app.core.config import Settings

    settings = Settings(gemini_api_key="k", max_revisions=1, app_env="test")
    provider = StubProvider(verdicts=[make_verdict(approved=False, score=5.0)])
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["revision_count"] == 1
    assert provider.writer_calls == 2
    assert state["workflow_status"] is WorkflowStatus.MAX_REVISIONS_REACHED


async def test_writer_failure_degrades_without_raising(settings, run_kwargs):
    provider = StubProvider(fail_on=["writer"])
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["workflow_status"] is WorkflowStatus.FAILED
    assert state["final_output"] == []
    assert len(state["errors"]) == 1
    assert state["errors"][0].agent is AgentName.WRITER
    # No stack trace or internals leak into the client-safe message.
    assert "Traceback" not in state["errors"][0].message
    # The loop must not have retried a failing writer.
    assert provider.writer_calls == 1


async def test_editor_failure_keeps_usable_draft(settings, run_kwargs):
    provider = StubProvider(fail_on=["editor"])
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["workflow_status"] is WorkflowStatus.COMPLETED_WITH_ERRORS
    assert len(state["final_output"]) == 2, "the unevaluated draft is still returned"
    assert state["errors"][0].agent is AgentName.EDITOR
    # Critically: a broken editor must not consume the revision budget.
    assert provider.writer_calls == 1


async def test_metrics_recorded_for_every_execution(settings, run_kwargs):
    provider = StubProvider(
        verdicts=[
            make_verdict(approved=False, score=5.0),
            make_verdict(approved=True, score=9.0),
        ]
    )
    state = await run_pipeline(provider, settings, run_kwargs)

    agents = [m.agent for m in state["agent_metrics"]]
    assert agents == [
        AgentName.RETRIEVER,
        AgentName.RESEARCHER,
        AgentName.PLANNER,
        AgentName.WRITER,
        AgentName.EDITOR,
        AgentName.WRITER,
        AgentName.EDITOR,
        AgentName.FACT_CHECKER,
        AgentName.FINALIZE,
    ]
    assert all(m.latency_seconds >= 0 for m in state["agent_metrics"])
    assert all(m.status is AgentStatus.COMPLETED for m in state["agent_metrics"])

    # The retriever and finalize make no LLM call, so they must report no token
    # usage at all. Zero would read as "this was free", which is a different
    # claim from "no tokens were spent here".
    no_llm = {AgentName.RETRIEVER, AgentName.FINALIZE}
    llm_metrics = [m for m in state["agent_metrics"] if m.agent not in no_llm]
    assert all(m.usage is not None and m.usage.total_tokens == 350 for m in llm_metrics)
    assert all(m.usage is None for m in state["agent_metrics"] if m.agent in no_llm)


async def test_failed_metric_carries_the_error(settings, run_kwargs):
    provider = StubProvider(fail_on=["writer"])
    state = await run_pipeline(provider, settings, run_kwargs)

    writer_metric = next(
        m for m in state["agent_metrics"] if m.agent is AgentName.WRITER
    )
    assert writer_metric.status is AgentStatus.FAILED
    assert writer_metric.error
    assert writer_metric.usage is None


def test_recursion_limit_exceeds_what_routing_permits():
    """The backstop must be looser than the predicate, never the enforcer."""
    for cap in (1, 3, 10):
        assert recursion_limit_for(cap) > (cap + 1) * 3
