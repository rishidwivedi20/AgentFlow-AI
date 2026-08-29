"""The Editor cannot approve a draft that carries a high-severity issue.

Enforced on EditorOutput itself, so routing, the draft history and the API
response all observe the same corrected verdict.
"""

from __future__ import annotations

import pytest

from app.models.enums import WorkflowStatus
from app.models.review import EditorIssue, EditorOutput, IssueSeverity, ReviewDimension
from tests.conftest import StubProvider, run_pipeline


def issue(severity: IssueSeverity) -> EditorIssue:
    return EditorIssue(
        dimension=ReviewDimension.RELEVANCE,
        severity=severity,
        description="Post 2 does not address the topic.",
    )


def verdict(*, approved: bool, severities: list[IssueSeverity], score: float = 9.0) -> EditorOutput:
    return EditorOutput(
        approved=approved,
        quality_score=score,
        issues=[issue(s) for s in severities],
        feedback=["Refocus post 2 on the topic."],
    )


# --- the rule --------------------------------------------------------------


def test_approval_with_high_severity_issue_is_overridden():
    assert verdict(approved=True, severities=[IssueSeverity.HIGH]).approved is False


def test_high_severity_overrides_even_with_a_perfect_score():
    """Severity wins over score; a 10/10 with a critical flaw is still rejected."""
    assert verdict(approved=True, severities=[IssueSeverity.HIGH], score=10.0).approved is False


def test_high_severity_among_lower_ones_still_overrides():
    result = verdict(
        approved=True,
        severities=[IssueSeverity.LOW, IssueSeverity.MEDIUM, IssueSeverity.HIGH],
    )
    assert result.approved is False


@pytest.mark.parametrize("severity", [IssueSeverity.LOW, IssueSeverity.MEDIUM])
def test_non_high_severity_issues_do_not_block_approval(severity):
    """The rule is narrow: only HIGH blocks. Minor nits may still be approved."""
    assert verdict(approved=True, severities=[severity]).approved is True


def test_clean_approval_is_untouched():
    assert verdict(approved=True, severities=[]).approved is True


def test_rejection_stays_rejected():
    assert verdict(approved=False, severities=[IssueSeverity.HIGH]).approved is False
    assert verdict(approved=False, severities=[]).approved is False


def test_issues_are_preserved_so_the_contradiction_stays_visible():
    """No flag records the override; the data itself shows it."""
    result = verdict(approved=True, severities=[IssueSeverity.HIGH])
    assert result.approved is False
    assert result.has_blocking_issues is True


def test_override_is_logged(caplog):
    with caplog.at_level("WARNING"):
        verdict(approved=True, severities=[IssueSeverity.HIGH])
    assert "overriding to rejected" in caplog.text


def test_rule_survives_deserialization():
    """Applies to provider JSON too, not only to objects built in Python."""
    payload = {
        "approved": True,
        "quality_score": 9.5,
        "issues": [
            {"dimension": "clarity", "severity": "high", "description": "Unreadable opening."}
        ],
        "feedback": ["Rewrite the hook."],
    }
    assert EditorOutput.model_validate(payload).approved is False


# --- effect on the workflow ------------------------------------------------


async def test_contradictory_approval_triggers_a_revision(settings, run_kwargs):
    """The real failure this prevents: finalising with zero revisions."""
    provider = StubProvider(
        verdicts=[
            verdict(approved=True, severities=[IssueSeverity.HIGH]),
            verdict(approved=True, severities=[]),
        ]
    )
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["revision_count"] == 1, "the contradictory verdict must not finalise the run"
    assert provider.writer_calls == 2
    assert state["workflow_status"] is WorkflowStatus.COMPLETED
    assert state["approved"] is True


async def test_persistent_contradiction_still_terminates(settings, run_kwargs):
    """Overriding approval must not create an unbounded loop."""
    provider = StubProvider(verdicts=[verdict(approved=True, severities=[IssueSeverity.HIGH])])
    state = await run_pipeline(provider, settings, run_kwargs)

    assert state["workflow_status"] is WorkflowStatus.MAX_REVISIONS_REACHED
    assert state["revision_count"] == 3
    assert provider.writer_calls == 4
