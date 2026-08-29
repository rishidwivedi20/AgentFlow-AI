"""Content Planner agent: assigns each post a distinct angle.

Decides what the posts are about; the Writer decides how they read. Contains no
orchestration logic.
"""

from __future__ import annotations

import logging
from typing import Final

from app.agents.prompts import planner as prompts
from app.models.enums import Platform, Tone
from app.models.plan import ContentPlan
from app.models.research import ResearchResult
from app.services.llm.base import LLMResult
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

PLANNER_TEMPERATURE: Final = 0.7
"""Moderately high: the Planner's whole job is finding varied angles, which a
near-deterministic setting works against."""


async def run_planner(
    llm: LLMService,
    *,
    topic: str,
    platform: Platform,
    tone: Tone,
    research: ResearchResult,
    number_of_posts: int,
) -> LLMResult[ContentPlan]:
    """Produce one angle per requested post."""
    result = await llm.generate_structured(
        system_prompt=prompts.build_system_prompt(platform, tone),
        user_prompt=prompts.build_user_prompt(
            topic=topic, research=research, number_of_posts=number_of_posts
        ),
        schema=ContentPlan,
        temperature=PLANNER_TEMPERATURE,
    )

    produced = len(result.value.angles)
    if produced != number_of_posts:
        # Logged, not fatal: the Writer is told how many posts to produce
        # regardless, and failing here would spend the run on a count slip.
        logger.warning(
            "Planner returned an unexpected angle count",
            extra={"requested": number_of_posts, "produced": produced},
        )

    return result
