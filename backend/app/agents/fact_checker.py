"""Fact Checker agent: judges the draft's claims against the research.

Verifies only against the research gathered for this run. It performs no
search of its own and does not fall back on model knowledge - a claim the
sources do not support is unsupported, however true it might be.

Whether an unsupported claim sends the draft back to the Writer is decided by
the graph's routing, not here.
"""

from __future__ import annotations

from typing import Final

from app.agents.prompts import fact_checker as prompts
from app.models.content import Post
from app.models.factcheck import FactCheckReport
from app.models.research import ResearchResult
from app.services.llm.base import LLMResult
from app.services.llm_service import LLMService

FACT_CHECKER_TEMPERATURE: Final = 0.1
"""Lowest in the system: the same claim against the same research must reach
the same verdict, or the loop becomes non-deterministic."""


async def run_fact_checker(
    llm: LLMService,
    *,
    research: ResearchResult,
    posts: list[Post],
) -> LLMResult[FactCheckReport]:
    """Extract the draft's factual claims and judge each against the research."""
    return await llm.generate_structured(
        system_prompt=prompts.build_system_prompt(),
        user_prompt=prompts.build_user_prompt(research=research, posts=posts),
        schema=FactCheckReport,
        temperature=FACT_CHECKER_TEMPERATURE,
    )
