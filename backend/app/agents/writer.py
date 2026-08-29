"""Writer agent: generates and revises posts.

Contains no orchestration logic. It does not decide whether a revision is
needed, does not touch the revision counter and does not know a graph exists -
it is handed everything it needs and returns content.
"""

from __future__ import annotations

import logging
from typing import Final

from app.agents.prompts import writer as prompts
from app.models.content import WriterOutput
from app.models.enums import Platform, Tone
from app.models.review import EditorOutput
from app.services.llm.base import LLMResult
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

WRITER_TEMPERATURE: Final = 0.8
"""Higher than the default: distinct angles across posts need variety, and a
revision must diverge from the draft the Editor rejected."""


async def run_writer(
    llm: LLMService,
    *,
    topic: str,
    platform: Platform,
    tone: Tone,
    number_of_posts: int,
    research_context: str | None = None,
    feedback: EditorOutput | None = None,
) -> LLMResult[WriterOutput]:
    """Produce one draft.

    ``feedback`` present means this is a revision; the previous critique is
    folded into the prompt. ``research_context`` is the Phase 3 seam - the
    Writer already grounds claims in it when supplied.
    """
    result = await llm.generate_structured(
        system_prompt=prompts.build_system_prompt(platform, tone),
        user_prompt=prompts.build_user_prompt(
            topic=topic,
            number_of_posts=number_of_posts,
            research_context=research_context,
            feedback=feedback,
        ),
        schema=WriterOutput,
        temperature=WRITER_TEMPERATURE,
    )

    produced = len(result.value.posts)
    if produced != number_of_posts:
        # Accepted rather than raised: a count mismatch still yields usable
        # content, and failing the run would spend the whole revision budget on
        # a formatting slip. The Editor sees the draft regardless.
        logger.warning(
            "Writer returned an unexpected post count",
            extra={"requested": number_of_posts, "produced": produced},
        )

    return result
