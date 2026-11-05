"""The single-agent baseline: one LLM call from evidence to finished posts.

Exists only to be measured against. It is never served by the API: it hands
raw page text straight to the model that writes the posts, which the
production pipeline deliberately never does.
"""

from __future__ import annotations

from typing import Final

from app.agents.writer import WRITER_TEMPERATURE
from app.evaluation import prompts
from app.models.content import WriterOutput
from app.models.enums import Platform, Tone
from app.models.evidence import EvidenceSet
from app.models.research import Source
from app.services.llm.base import LLMResult
from app.services.llm_service import LLMService

BASELINE_TEMPERATURE: Final = WRITER_TEMPERATURE
"""Matches the Writer, so sampling variety is not a difference between the two."""


async def run_single_agent(
    llm: LLMService,
    *,
    topic: str,
    platform: Platform,
    tone: Tone,
    number_of_posts: int,
    sources: list[Source],
    evidence: EvidenceSet | None,
) -> LLMResult[WriterOutput]:
    """Research, plan, write and self-review in a single structured call."""
    return await llm.generate_structured(
        system_prompt=prompts.build_baseline_system_prompt(platform, tone),
        user_prompt=prompts.build_baseline_user_prompt(
            topic=topic,
            number_of_posts=number_of_posts,
            sources=sources,
            evidence=evidence,
        ),
        schema=WriterOutput,
        temperature=BASELINE_TEMPERATURE,
    )
