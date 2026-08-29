"""Best-draft selection, and the honesty rules around verification.

The rule under test: a draft verified free of unsupported claims beats any
draft carrying them, however well it scored editorially. An unchecked draft is
not clean - absence of verification is not evidence of correctness.
"""

from __future__ import annotations

from app.graph.state import initial_state, merge_drafts, select_best_draft
from app.models.content import DraftRecord
from app.models.enums import Platform, Tone, WorkflowStatus
from app.models.factcheck import VerificationStatus
from tests.conftest import (
    StubProvider,
    make_fact_check,
    make_posts,
    make_verdict,
    run_pipeline,
)


def draft(revision: int, score: float, *, unsupported: int | None = None) -> DraftRecord:
    """Build a draft. ``unsupported=None`` means the Fact Checker never ran."""
    return DraftRecord(
        revision=revision,
        posts=make_posts(1, tag=f"r{revision}"),
        evaluation=make_verdict(approved=True, score=score),
        fact_check=None if unsupported is None else make_fact_check(unsupported=unsupported),
    )


def _state_with(drafts: list[DraftRecord]):
    state = initial_state(
        topic="t",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=1,
        max_revisions=3,
    )
    state["drafts"] = drafts
    return state


# --- tier 1: factual cleanliness outranks editorial score -------------------


def test_clean_draft_beats_a_higher_scoring_unclean_one():
    """The whole point: 9.2 with invented statistics loses to a clean 8.8."""
    unclean = draft(0, 9.2, unsupported=3)
    clean = draft(1, 8.8, unsupported=0)
    best = select_best_draft(_state_with([unclean, clean]))
    assert best.revision == 1
    assert best.quality_score == 8.8


def test_clean_draft_beats_an_unchecked_one_even_at_lower_score():
    """Unchecked is not clean; absence of a check is not evidence."""
    unchecked = draft(0, 9.5, unsupported=None)
    clean = draft(1, 7.0, unsupported=0)
    assert select_best_draft(_state_with([unchecked, clean])).revision == 1


def test_unchecked_draft_ranks_with_unclean_ones():
    unchecked = draft(0, 9.5, unsupported=None)
    unclean = draft(1, 6.0, unsupported=2)
    # Neither is clean, so the higher score wins within that tier.
    assert select_best_draft(_state_with([unchecked, unclean])).revision == 0


# --- tier 2: score, within a tier ------------------------------------------


def test_highest_score_wins_among_clean_drafts():
    best = select_best_draft(
        _state_with([draft(0, 7.0, unsupported=0), draft(1, 9.1, unsupported=0)])
    )
    assert best.quality_score == 9.1


def test_highest_score_wins_among_unclean_drafts():
    best = select_best_draft(
        _state_with([draft(0, 5.0, unsupported=1), draft(1, 8.0, unsupported=1)])
    )
    assert best.quality_score == 8.0


# --- tier 3: deterministic tie-breaking ------------------------------------


def test_ties_resolve_to_the_earliest_draft():
    best = select_best_draft(
        _state_with([draft(0, 8.0, unsupported=0), draft(1, 8.0, unsupported=0)])
    )
    assert best.revision == 0


def test_selection_is_deterministic_across_input_order():
    drafts = [draft(0, 9.2, unsupported=3), draft(1, 8.8, unsupported=0)]
    forwards = select_best_draft(_state_with(drafts))
    backwards = select_best_draft(_state_with(list(reversed(drafts))))
    assert forwards.revision == backwards.revision == 1


def test_no_drafts_returns_none():
    assert select_best_draft(_state_with([])) is None


# --- the merge reducer that makes one candidate per revision ---------------


def test_fact_check_replaces_rather_than_duplicates_a_revision():
    """Editor writes the record; Fact Checker updates the same revision."""
    from_editor = draft(0, 9.0, unsupported=None)
    from_checker = draft(0, 9.0, unsupported=0)
    merged = merge_drafts([from_editor], [from_checker])
    assert len(merged) == 1, "one revision must yield one candidate"
    assert merged[0].fact_check is not None


def test_merge_appends_new_revisions_in_order():
    merged = merge_drafts([draft(0, 8.0)], [draft(2, 7.0), draft(1, 9.0)])
    assert [d.revision for d in merged] == [0, 1, 2]


# --- verification status on drafts -----------------------------------------


def test_verification_status_reflects_the_report():
    assert draft(0, 8.0, unsupported=None).verification_status is VerificationStatus.NOT_CHECKED
    assert draft(0, 8.0, unsupported=0).verification_status is VerificationStatus.VERIFIED
    assert (
        draft(0, 8.0, unsupported=2).verification_status
        is VerificationStatus.UNSUPPORTED_CLAIMS
    )


def test_unchecked_draft_is_not_factually_clean():
    assert draft(0, 9.9, unsupported=None).is_factually_clean is False
    assert draft(0, 5.0, unsupported=0).is_factually_clean is True


# --- end to end: the run must not overclaim --------------------------------


async def test_clean_verified_run_is_completed(settings, run_kwargs):
    state = await run_pipeline(StubProvider(), settings, run_kwargs)
    assert state["workflow_status"] is WorkflowStatus.COMPLETED
    assert state["warnings"] == [], "a clean run needs no caveats"


async def test_fact_checker_failure_does_not_claim_verification(settings, run_kwargs):
    """Requirement 9: a broken checker must never read as verified."""
    provider = StubProvider(fail_on=["fact_checker"])
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["workflow_status"] is not WorkflowStatus.COMPLETED
    assert state["final_output"], "posts are still returned"
    assert any("not fact checked" in w for w in state["warnings"])
    best = select_best_draft(state)
    assert best.verification_status is VerificationStatus.NOT_CHECKED


async def test_unsupported_claims_at_the_cap_are_surfaced_not_hidden(settings, run_kwargs):
    provider = StubProvider(
        verdicts=[make_verdict(approved=True, score=9.0)],
        fact_checks=[make_fact_check(unsupported=2)],
    )
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["workflow_status"] is WorkflowStatus.COMPLETED_UNVERIFIED
    assert state["errors"] == [], "nothing crashed - this is not an error state"
    assert any("not supported" in w for w in state["warnings"])
    assert any("revision limit" in w for w in state["warnings"])


async def test_missing_sources_are_warned_about(settings, run_kwargs):
    from tests.conftest import StubSearchProvider

    state = await run_pipeline(
        StubProvider(), settings, run_kwargs, search=StubSearchProvider(sources=[])
    )
    assert any("No sources were found" in w for w in state["warnings"])
