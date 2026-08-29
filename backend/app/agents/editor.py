"""Editor agent: critiques drafts, never rewrites them.

Returns a structured verdict only. The decision about what happens next
(approve, revise, stop) belongs to the graph's routing function, not here.
"""

from __future__ import annotations

from typing import Final

from app.agents.prompts import editor as prompts
from app.models.content import Post
from app.models.enums import Platform, Tone
from app.models.review import EditorOutput
from app.services.llm.base import LLMResult
from app.services.llm_service import LLMService

EDITOR_TEMPERATURE: Final = 0.2
"""Low: evaluation must be as reproducible as possible, so the same draft does
not oscillate between approved and rejected across runs."""


async def run_editor(
    llm: LLMService,
    *,
    topic: str,
    platform: Platform,
    tone: Tone,
    posts: list[Post],
    revision: int,
) -> LLMResult[EditorOutput]:
    """Evaluate a draft and return a structured verdict."""
    return await llm.generate_structured(
        system_prompt=prompts.build_system_prompt(platform, tone),
        user_prompt=prompts.build_user_prompt(topic=topic, posts=posts, revision=revision),
        schema=EditorOutput,
        temperature=EDITOR_TEMPERATURE,
    )
