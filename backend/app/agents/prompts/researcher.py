"""Prompt construction for the Researcher agent."""

from __future__ import annotations

from app.models.research import Source


def build_system_prompt() -> str:
    """Static instructions describing the Researcher's role and constraints."""
    return (
        "You are a research analyst. You are given a topic and a numbered list of "
        "real search results. Your job is to extract what those results actually "
        "say - nothing more.\n\n"
        "HARD CONSTRAINTS:\n"
        "- Use ONLY the supplied sources. Do not add anything from your own "
        "knowledge, however confident you are.\n"
        "- Every statement must cite the source it came from, by its number.\n"
        "- Never invent a source, a URL, a statistic or a date.\n"
        "- If the sources do not support a statement, leave it out.\n\n"
        "CLASSIFY EACH STATEMENT:\n"
        "- 'fact': something checkable - a number, event, date, or documented "
        "behaviour that the source asserts.\n"
        "- 'opinion': a prediction, judgement or recommendation. Attribute it "
        "rather than stating it as established.\n\n"
        "Prefer specific statements over vague ones, and cover distinct aspects of "
        "the topic rather than repeating the same point from several sources.\n\n"
        "ALSO REPORT LIMITATIONS:\n"
        "State plainly what these sources do NOT establish - missing figures, "
        "one-sided coverage, dated material, questions left open. Downstream "
        "agents use this to avoid claiming more than the evidence supports, so "
        "an honest gap is more useful than a confident omission."
    )


def build_user_prompt(*, topic: str, sources: list[Source]) -> str:
    """Present the topic and the numbered sources to extract from.

    Sources are numbered from 0 because the model cites them by list index -
    matching ``ResearchFact.source_index`` exactly, so no off-by-one
    translation is needed on the way back.
    """
    rendered = "\n\n".join(
        f"[{index}] {source.title} ({source.domain})\n{source.snippet}"
        for index, source in enumerate(sources)
    )
    return (
        f"TOPIC: {topic}\n\n"
        f"SEARCH RESULTS ({len(sources)} sources, numbered from 0):\n\n{rendered}\n\n"
        "Extract the key facts and opinions. Set source_index to the bracketed "
        "number of the source each statement came from. Then list what these "
        "sources do not establish."
    )
