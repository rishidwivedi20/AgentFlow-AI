"""Which agents evidence reaches, and repair of citations that point nowhere."""

from __future__ import annotations

import pytest

from app.agents.fact_checker import _repair_citations
from app.agents.prompts import fact_checker as fact_checker_prompts
from app.agents.prompts import researcher as researcher_prompts
from app.agents.prompts import writer as writer_prompts
from app.models.evidence import RetrievalStatus
from app.models.factcheck import ClaimVerdict, FactCheckClaim, FactCheckReport
from tests.conftest import make_research_result, make_sources
from tests.test_evidence_prompt import INJECTION, make_evidence

PAGE_TEXT = "Adoption reached 41 percent among surveyed teams in the second quarter."


# --- Which agents see raw page text ---------------------------------------


def test_the_researcher_receives_the_excerpts():
    prompt = researcher_prompts.build_user_prompt(
        topic="alpha", sources=make_sources(2), evidence=make_evidence(PAGE_TEXT)
    )
    assert PAGE_TEXT in prompt
    assert "UNTRUSTED DATA" in prompt


def test_the_fact_checker_receives_the_excerpts():
    """The last check before publication, judged against the sources' own
    words rather than a paraphrase of them."""
    prompt = fact_checker_prompts.build_user_prompt(
        research=make_research_result(),
        posts=[],
        evidence=make_evidence(PAGE_TEXT),
    )
    assert PAGE_TEXT in prompt
    assert "UNTRUSTED DATA" in prompt


def test_the_writer_never_receives_raw_page_text():
    """The Writer produces the prose a reader sees, and is fed only by
    ``research_context`` - facts already extracted under a constrained schema
    and index-checked. If the Writer prompt grows an ``evidence`` parameter
    this fails, so the tradeoff gets re-argued rather than assumed.
    """
    assert "evidence" not in writer_prompts.build_user_prompt.__code__.co_varnames

    prompt = writer_prompts.build_user_prompt(
        topic="alpha",
        number_of_posts=2,
        research_context=make_research_result().as_writer_context(),
    )
    assert INJECTION not in prompt
    assert PAGE_TEXT not in prompt


# --- Token economy of the researcher prompt -------------------------------


def test_snippets_are_dropped_for_sources_the_excerpts_already_cover():
    """A snippet previews text the excerpt already carries in full."""
    sources = make_sources(3)
    evidence = make_evidence(PAGE_TEXT, source_indexes=(1,))
    prompt = researcher_prompts.build_user_prompt(
        topic="alpha", sources=sources, evidence=evidence
    )

    assert sources[1].snippet not in prompt
    # Sources retrieval never reached keep theirs: it is all there is for them.
    assert sources[0].snippet in prompt
    assert sources[2].snippet in prompt


def test_every_source_stays_citable_even_without_an_excerpt():
    """An index that no longer resolves loses every fact citing it."""
    sources = make_sources(3)
    prompt = researcher_prompts.build_user_prompt(
        topic="alpha",
        sources=sources,
        evidence=make_evidence(PAGE_TEXT, source_indexes=(1,)),
    )
    for index in range(3):
        assert f"[{index}]" in prompt


def test_all_snippets_survive_when_retrieval_did_not_run():
    sources = make_sources(2)
    prompt = researcher_prompts.build_user_prompt(
        topic="alpha", sources=sources, evidence=None
    )
    assert sources[0].snippet in prompt
    assert sources[1].snippet in prompt


def test_weak_evidence_still_reaches_the_researcher():
    """Weak is not absent - withholding it gives the model less than the run
    actually found."""
    prompt = researcher_prompts.build_user_prompt(
        topic="alpha",
        sources=make_sources(2),
        evidence=make_evidence(PAGE_TEXT, status=RetrievalStatus.WEAK_EVIDENCE),
    )
    assert PAGE_TEXT in prompt


# --- Citation repair ------------------------------------------------------


def claim(verdict: ClaimVerdict, index: int) -> FactCheckClaim:
    return FactCheckClaim(
        claim="Adoption reached 41 percent.",
        verdict=verdict,
        reasoning="Stated by the source.",
        post_index=1,
        supporting_source_index=index,
    )


def test_a_verified_claim_citing_a_nonexistent_source_is_downgraded():
    """Verified means a real source supports it; an index into empty space
    does not."""
    report = _repair_citations(
        FactCheckReport(claims=[claim(ClaimVerdict.VERIFIED, 9)]), source_count=3
    )
    repaired = report.claims[0]
    assert repaired.verdict is ClaimVerdict.UNCERTAIN
    assert repaired.supporting_source_index == -1


def test_a_broken_citation_never_becomes_unsupported():
    """A bad index shows the stated backing is not real, not that the claim is
    wrong. Marking it unsupported would spend a revision deleting something
    possibly true."""
    report = _repair_citations(
        FactCheckReport(claims=[claim(ClaimVerdict.VERIFIED, 9)]), source_count=1
    )
    assert report.claims[0].verdict is not ClaimVerdict.UNSUPPORTED
    assert not report.has_critical_issues


def test_an_unsupported_claim_carrying_a_citation_is_normalised():
    """A citation on an unsupported claim contradicts its own verdict."""
    report = _repair_citations(
        FactCheckReport(claims=[claim(ClaimVerdict.UNSUPPORTED, 2)]), source_count=3
    )
    assert report.claims[0].supporting_source_index == -1
    assert report.claims[0].verdict is ClaimVerdict.UNSUPPORTED
    assert report.has_critical_issues


@pytest.mark.parametrize(
    "verdict", [ClaimVerdict.VERIFIED, ClaimVerdict.UNCERTAIN]
)
def test_valid_citations_are_left_alone(verdict: ClaimVerdict):
    original = FactCheckReport(claims=[claim(verdict, 2)])
    assert _repair_citations(original, source_count=3) == original


def test_repair_never_upgrades_a_verdict():
    """Every transition is downward or lateral."""
    for verdict in ClaimVerdict:
        for index in (-1, 0, 5):
            report = _repair_citations(
                FactCheckReport(claims=[claim(verdict, index)]), source_count=2
            )
            result = report.claims[0].verdict
            if verdict is ClaimVerdict.UNCERTAIN:
                assert result is ClaimVerdict.UNCERTAIN
            elif verdict is ClaimVerdict.UNSUPPORTED:
                assert result is ClaimVerdict.UNSUPPORTED
            else:
                assert result in {ClaimVerdict.VERIFIED, ClaimVerdict.UNCERTAIN}


def test_repair_keeps_every_claim():
    """Unlike research facts, a miscited claim is still a real finding about
    the draft, so it is repaired rather than dropped."""
    claims = [
        claim(ClaimVerdict.VERIFIED, 0),
        claim(ClaimVerdict.VERIFIED, 99),
        claim(ClaimVerdict.UNSUPPORTED, -1),
    ]
    report = _repair_citations(FactCheckReport(claims=claims), source_count=2)
    assert len(report.claims) == 3


def test_no_sources_at_all_invalidates_every_citation():
    report = _repair_citations(
        FactCheckReport(claims=[claim(ClaimVerdict.VERIFIED, 0)]), source_count=0
    )
    assert report.claims[0].supporting_source_index == -1
    assert report.claims[0].verdict is ClaimVerdict.UNCERTAIN
