"""Fact Checker agent: judges the draft's claims against the research.

Verifies only against the research gathered for this run. It performs no
search of its own and does not fall back on model knowledge - a claim the
sources do not support is unsupported, however true it might be.

Whether an unsupported claim sends the draft back to the Writer is decided by
the graph's routing, not here.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Final

from app.agents.prompts import fact_checker as prompts
from app.models.content import Post
from app.models.evidence import EvidenceSet
from app.models.factcheck import ClaimVerdict, FactCheckClaim, FactCheckReport
from app.models.research import ResearchResult
from app.services.llm.base import LLMResult
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

FACT_CHECKER_TEMPERATURE: Final = 0.1
"""Lowest in the system: the same claim against the same research must reach
the same verdict, or the loop becomes non-deterministic."""


def _repair_citations(report: FactCheckReport, source_count: int) -> FactCheckReport:
    """Strip citations that point at sources which were never supplied.

    The mirror of ``_drop_uncitable_facts`` in the Researcher, but it repairs
    rather than discards: the finding that a claim exists is worth keeping even
    when the model attached the wrong evidence to it.

    A ``verified`` verdict resting on a source that does not exist is
    downgraded to ``uncertain``. Verified means "a source directly supports
    this", and a citation into empty space is not support - leaving the verdict
    intact would let a fabricated reference certify a claim, which is the exact
    failure the cite-by-index design exists to prevent. It is never downgraded
    to ``unsupported``: nothing here establishes the claim is wrong, only that
    the stated backing is not real.

    Nothing is ever upgraded, so this cannot manufacture confidence.
    """
    repaired: list[FactCheckClaim] = []
    stripped = 0
    downgraded = 0

    for claim in report.claims:
        index = claim.supporting_source_index
        out_of_range = index >= source_count
        # The prompt requires -1 on unsupported claims; a citation there
        # contradicts the verdict, so it is not trustworthy either.
        contradictory = claim.verdict is ClaimVerdict.UNSUPPORTED and index >= 0

        if not out_of_range and not contradictory:
            repaired.append(claim)
            continue

        stripped += 1
        verdict = claim.verdict
        if out_of_range and verdict is ClaimVerdict.VERIFIED:
            verdict = ClaimVerdict.UNCERTAIN
            downgraded += 1
        repaired.append(
            claim.model_copy(update={"supporting_source_index": -1, "verdict": verdict})
        )

    if stripped:
        logger.warning(
            "Repaired fact check citations that referenced unsupplied sources",
            extra={
                "stripped": stripped,
                "downgraded": downgraded,
                "source_count": source_count,
            },
        )
    return report.model_copy(update={"claims": repaired})


async def run_fact_checker(
    llm: LLMService,
    *,
    research: ResearchResult,
    posts: list[Post],
    evidence: EvidenceSet | None = None,
) -> LLMResult[FactCheckReport]:
    """Extract the draft's factual claims and judge each against the evidence."""
    result = await llm.generate_structured(
        system_prompt=prompts.build_system_prompt(),
        user_prompt=prompts.build_user_prompt(
            research=research, posts=posts, evidence=evidence
        ),
        schema=FactCheckReport,
        temperature=FACT_CHECKER_TEMPERATURE,
    )
    return replace(
        result, value=_repair_citations(result.value, len(research.sources))
    )
