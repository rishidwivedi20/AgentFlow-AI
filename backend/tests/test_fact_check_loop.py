"""The Fact Checker's feedback loop and its shared revision budget."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.graph.routing import FactCheckRoute, route_after_fact_check
from app.graph.state import initial_state
from app.models.enums import AgentName, AgentStatus, Platform, Tone, WorkflowStatus
from app.models.factcheck import ClaimVerdict, FactCheckClaim, FactCheckReport
from app.models.metrics import AgentMetric
from tests.conftest import StubProvider, make_fact_check, make_verdict, run_pipeline


def _state(*, report, revision_count=0, max_revisions=3, checker_failed=False):
    state = initial_state(
        topic="topic",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
        max_revisions=max_revisions,
    )
    state["fact_check"] = report
    state["revision_count"] = revision_count
    if checker_failed:
        now = datetime.now(UTC)
        state["agent_metrics"] = [
            AgentMetric(
                sequence=0,
                agent=AgentName.FACT_CHECKER,
                status=AgentStatus.FAILED,
                started_at=now,
                ended_at=now,
                latency_seconds=0.0,
            )
        ]
    return state


# --- the critical-issue rule -----------------------------------------------


def test_only_unsupported_claims_are_critical():
    """Uncertain claims are reported but must not force a rewrite."""
    report = FactCheckReport(
        claims=[
            FactCheckClaim(
                claim="c",
                verdict=ClaimVerdict.UNCERTAIN,
                reasoning="r",
                post_index=1,
                supporting_source_index=-1,
            )
        ]
    )
    assert report.has_critical_issues is False
    assert route_after_fact_check(_state(report=report)) is FactCheckRoute.FINALIZE


def test_unsupported_claim_triggers_revision():
    report = make_fact_check(unsupported=1)
    assert report.has_critical_issues is True
    assert route_after_fact_check(_state(report=report)) is FactCheckRoute.REVISE


def test_verified_only_report_finalizes():
    assert route_after_fact_check(_state(report=make_fact_check())) is FactCheckRoute.FINALIZE


def test_empty_report_finalizes():
    assert (
        route_after_fact_check(_state(report=FactCheckReport(claims=[])))
        is FactCheckRoute.FINALIZE
    )


def test_missing_report_finalizes():
    assert route_after_fact_check(_state(report=None)) is FactCheckRoute.FINALIZE


def test_checker_failure_finalizes_rather_than_looping():
    """A broken checker must not consume the revision budget."""
    state = _state(report=make_fact_check(unsupported=2), checker_failed=True)
    assert route_after_fact_check(state) is FactCheckRoute.FINALIZE


@pytest.mark.parametrize("revision_count", [0, 1, 2])
def test_revises_while_budget_remains(revision_count):
    state = _state(report=make_fact_check(unsupported=1), revision_count=revision_count)
    assert route_after_fact_check(state) is FactCheckRoute.REVISE


def test_cap_reached_finalizes_despite_unsupported_claims():
    state = _state(report=make_fact_check(unsupported=1), revision_count=3)
    assert route_after_fact_check(state) is FactCheckRoute.FINALIZE


def test_routing_does_not_mutate_state():
    state = _state(report=make_fact_check(unsupported=1))
    before = dict(state)
    route_after_fact_check(state)
    assert dict(state) == before


# --- report helpers ---------------------------------------------------------


def test_verdict_partitions_are_disjoint_and_complete():
    report = make_fact_check(unsupported=2, verified=3)
    report.claims.append(
        FactCheckClaim(
            claim="u",
            verdict=ClaimVerdict.UNCERTAIN,
            reasoning="r",
            post_index=1,
            supporting_source_index=-1,
        )
    )
    assert len(report.verified) == 3
    assert len(report.unsupported) == 2
    assert len(report.uncertain) == 1
    assert len(report.verified) + len(report.unsupported) + len(report.uncertain) == len(
        report.claims
    )


def test_writer_feedback_names_only_unsupported_claims():
    report = make_fact_check(unsupported=1, verified=2)
    feedback = report.as_writer_feedback()
    assert "Invented statistic 0" in feedback
    assert "Verified claim 0" not in feedback


# --- end to end -------------------------------------------------------------


async def test_unsupported_claims_send_the_draft_back_to_the_writer(settings, run_kwargs):
    provider = StubProvider(
        verdicts=[make_verdict(approved=True, score=9.0)],
        fact_checks=[make_fact_check(unsupported=1), make_fact_check()],
    )
    state = await run_pipeline(provider, settings, run_kwargs)

    assert provider.writer_calls == 2, "the fact check must have forced a rewrite"
    assert provider.fact_check_calls == 2
    assert state["revision_count"] == 1
    assert state["workflow_status"] is WorkflowStatus.COMPLETED


async def test_persistently_unsupported_claims_still_terminate(settings, run_kwargs):
    """The fact-check loop must be bounded exactly like the editor loop."""
    provider = StubProvider(
        verdicts=[make_verdict(approved=True, score=9.0)],
        fact_checks=[make_fact_check(unsupported=1)],
    )
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["revision_count"] == 3
    assert provider.writer_calls == 4
    # Posts are still returned, but the run must not claim they are verified.
    assert state["final_output"]
    assert state["workflow_status"] is WorkflowStatus.COMPLETED_UNVERIFIED
    assert state["errors"] == [], "nothing crashed; this is not an error state"
    assert any("not supported" in w for w in state["warnings"])
    assert any("revision limit" in w for w in state["warnings"])


async def test_both_critics_share_one_revision_budget(settings, run_kwargs):
    """Editor rejections and fact-check rejections cannot alternate forever."""
    provider = StubProvider(
        verdicts=[
            make_verdict(approved=False, score=5.0),  # rejection 1 -> revision 1
            make_verdict(approved=True, score=9.0),
        ],
        fact_checks=[make_fact_check(unsupported=1)],  # rejects every time
    )
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["revision_count"] == 3, "one shared budget, not two"
    assert provider.writer_calls == 4
    assert state["workflow_status"] is WorkflowStatus.COMPLETED_UNVERIFIED


async def test_fact_checker_runs_only_after_approval(settings, run_kwargs):
    """A rejected draft is not worth verifying; it is about to be rewritten."""
    provider = StubProvider(verdicts=[make_verdict(approved=False, score=4.0)])
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["workflow_status"] is WorkflowStatus.MAX_REVISIONS_REACHED
    assert provider.editor_calls == 4
    assert provider.fact_check_calls == 0, "never approved, so never verified"


# --- requirement 4: the Editor must not be bypassed on a fact-check revision -


async def test_fact_check_revision_goes_back_through_the_editor(settings, run_kwargs):
    """The rewritten draft must be re-reviewed, not sent straight to checking.

    Asserted on call counts rather than inferred: after a fact-check rejection
    the Editor runs again before the Fact Checker sees the new draft.
    """
    provider = StubProvider(
        verdicts=[make_verdict(approved=True, score=9.0)],
        fact_checks=[make_fact_check(unsupported=1), make_fact_check()],
    )
    state = await run_pipeline(provider, settings, run_kwargs)

    assert provider.writer_calls == 2
    assert provider.editor_calls == 2, "the revised draft must be re-reviewed"
    assert provider.fact_check_calls == 2
    assert state["revision_count"] == 1
    # Both revisions were evaluated, so both are selection candidates.
    assert [d.revision for d in state["drafts"]] == [0, 1]


def test_no_edge_bypasses_the_editor_from_the_writer():
    """Structural guarantee: the Writer's only forward edge is the Editor."""
    from app.graph.workflow import build_workflow
    from tests.conftest import make_research, make_service

    graph = build_workflow(make_service(StubProvider()), make_research()).get_graph()
    writer_targets = {e.target for e in graph.edges if e.source == "writer"}
    assert "fact_checker" not in writer_targets, "fact checking must never skip review"
    assert writer_targets == {"editor", "finalize"}


async def test_fact_checker_failure_is_recorded_as_an_error(settings, run_kwargs):
    """Requirement 9, at the state level."""
    provider = StubProvider(fail_on=["fact_checker"])
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["errors"], "a crashed checker must be recorded"
    assert state["errors"][0].agent.value == "fact_checker"
    assert state["fact_check"] is None
    # And it must not have burned the revision budget retrying.
    assert provider.writer_calls == 1
