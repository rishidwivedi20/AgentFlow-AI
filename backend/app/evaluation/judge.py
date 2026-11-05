"""Evaluators applied identically to both systems' output.

Quality comes from a neutral rubric judge. Grounding comes from the existing
Fact Checker, run against a research base built from the search results
themselves rather than from either system's working notes - the multi-agent
Researcher's distilled facts would favour the output that was written from
them.
"""

from __future__ import annotations

from typing import Final

from app.agents.fact_checker import run_fact_checker
from app.evaluation import prompts
from app.evaluation.models import Grounding, JudgeOutput
from app.models.content import Post
from app.models.enums import Platform, Tone
from app.models.evidence import EvidenceSet
from app.models.factcheck import FactCheckReport
from app.models.research import ResearchFact, ResearchResult, Source, StatementKind
from app.services.llm.base import LLMResult
from app.services.llm_service import LLMService

JUDGE_TEMPERATURE: Final = 0.0
"""As deterministic as the provider allows: the same posts should get the same
score, or differences between systems drown in judge noise."""


def neutral_research(sources: list[Source]) -> ResearchResult:
    """A research base neither system produced: each search snippet, verbatim.

    Snippets are marked as facts so the Fact Checker holds claims to them; the
    retrieved excerpts, passed alongside, carry the rest of the evidence.
    """
    return ResearchResult(
        summary=(
            "Evaluation reference: the search results' own summaries, unedited. "
            "The source excerpts below are the remaining evidence."
        ),
        facts=[
            ResearchFact(statement=source.snippet, kind=StatementKind.FACT, source_index=index)
            for index, source in enumerate(sources)
            if source.snippet.strip()
        ],
        limitations=[],
        sources=sources,
    )


async def judge_quality(
    llm: LLMService,
    *,
    topic: str,
    platform: Platform,
    tone: Tone,
    number_of_posts: int,
    posts: list[Post],
) -> LLMResult[JudgeOutput]:
    """Score one set of posts. The judge is never told which system wrote them."""
    return await llm.generate_structured(
        system_prompt=prompts.build_judge_system_prompt(),
        user_prompt=prompts.build_judge_user_prompt(
            topic=topic,
            platform=platform,
            tone=tone,
            number_of_posts=number_of_posts,
            posts=posts,
        ),
        schema=JudgeOutput,
        temperature=JUDGE_TEMPERATURE,
    )


async def judge_grounding(
    llm: LLMService,
    *,
    sources: list[Source],
    evidence: EvidenceSet | None,
    posts: list[Post],
) -> tuple[Grounding, LLMResult[FactCheckReport]]:
    """Verify one output's claims against the shared, neutral evidence."""
    result = await run_fact_checker(
        llm, research=neutral_research(sources), posts=posts, evidence=evidence
    )
    return Grounding(report=result.value), result
