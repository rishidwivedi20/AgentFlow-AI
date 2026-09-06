"""Researcher agent: extracts structured facts from the evidence it is handed.

Since Phase 5 the search itself belongs to the Retriever, which gathers and
selects evidence without any LLM call. This agent does the interpreting: it
reads what was found and extracts statements, referencing sources by index.
The split keeps each node's metric honest, because embedding a corpus and
reasoning over it have very different costs.

Because URLs never appear in the model's output schema, a fabricated citation
is not possible - the worst the model can do is cite an index that does not
exist, which is dropped here.
"""

from __future__ import annotations

import logging
from typing import Final

from app.agents.prompts import researcher as prompts
from app.models.evidence import EvidenceSet
from app.models.research import ResearchFact, ResearchOutput, ResearchResult, Source
from app.services.llm.base import LLMResult
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

RESEARCHER_TEMPERATURE: Final = 0.2
"""Low: extraction should reproduce what the sources say, not embellish it."""


def build_search_query(topic: str) -> str:
    """Turn a content topic into a search query.

    Kept trivial on purpose. A query-expansion step would cost an extra LLM
    call and another search credit for a gain this project has no way to
    measure yet.
    """
    return topic.strip()


def _drop_uncitable_facts(facts: list[ResearchFact], source_count: int) -> list[ResearchFact]:
    """Remove facts whose source_index does not refer to a real source.

    The final guard on fabricated citations: an out-of-range index means the
    model referenced evidence it was not given, so the statement is discarded
    rather than surfaced with a broken or misleading attribution.
    """
    kept = [fact for fact in facts if 0 <= fact.source_index < source_count]
    dropped = len(facts) - len(kept)
    if dropped:
        logger.warning(
            "Dropped research facts citing sources that were not supplied",
            extra={"dropped": dropped, "source_count": source_count},
        )
    return kept


async def run_researcher(
    llm: LLMService,
    *,
    topic: str,
    sources: list[Source],
    evidence: EvidenceSet | None = None,
) -> tuple[ResearchResult, LLMResult[ResearchOutput] | None]:
    """Extract structured research from sources the Retriever gathered.

    Returns the assembled result plus the raw LLM result, so the caller can
    record token usage. When no sources were found, no LLM call is made and
    the second element is None - there is nothing to extract from, and
    spending a call to say so would be waste.
    """
    if not sources:
        logger.warning("No sources to research", extra={"topic": topic[:80]})
        return (
            ResearchResult(
                summary="No sources were found for this topic.",
                facts=[],
                limitations=["No sources were retrieved, so nothing here is grounded."],
                sources=[],
            ),
            None,
        )

    result = await llm.generate_structured(
        system_prompt=prompts.build_system_prompt(),
        user_prompt=prompts.build_user_prompt(
            topic=topic, sources=sources, evidence=evidence
        ),
        schema=ResearchOutput,
        temperature=RESEARCHER_TEMPERATURE,
    )

    return (
        ResearchResult(
            summary=result.value.summary,
            facts=_drop_uncitable_facts(result.value.facts, len(sources)),
            limitations=result.value.limitations,
            sources=sources,
        ),
        result,
    )
