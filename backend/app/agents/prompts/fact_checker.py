"""Prompt construction for the Fact Checker agent."""

from __future__ import annotations

from app.models.content import Post
from app.models.research import ResearchResult


def build_system_prompt() -> str:
    """Static instructions describing the Fact Checker's role and standard."""
    return (
        "You are a fact checker. You are given research and a draft written from "
        "it. Find every factual assertion in the draft and judge each one against "
        "the research ONLY.\n\n"
        "THE STANDARD:\n"
        "You are not judging whether a claim is true in the world. You are judging "
        "whether THIS research supports it. A claim you know to be correct is still "
        "unsupported if the research does not establish it.\n\n"
        "VERDICTS:\n"
        "- verified: a research fact directly supports this claim.\n"
        "- uncertain: related to the research and plausible, but not directly "
        "established by it.\n"
        "- unsupported: contradicted by the research, or a specific claim - a "
        "number, date, percentage, named study or attributed quote - with nothing "
        "in the research behind it.\n\n"
        "WHAT COUNTS AS A CLAIM:\n"
        "- Statistics, percentages, dates, named companies, studies, quotations.\n"
        "- Assertions about what is happening in an industry or what tools do.\n\n"
        "WHAT DOES NOT:\n"
        "- Opinions the draft presents as opinions.\n"
        "- Advice, questions, calls to action and rhetorical framing.\n"
        "- General statements no reader would take as evidence.\n\n"
        "CITING YOUR EVIDENCE:\n"
        "Set supporting_source_index to the bracketed number of the source that "
        "backs the claim. Use -1 when nothing supports it - always -1 for "
        "unsupported claims. Never point at a source that does not actually "
        "establish the claim.\n\n"
        "Be precise, not exhaustive. Flagging ordinary prose as unsupported forces "
        "pointless rewrites; missing an invented statistic defeats the purpose. If "
        "the draft makes no factual assertions, return an empty list."
    )


def build_user_prompt(*, research: ResearchResult, posts: list[Post]) -> str:
    """Present the research and the draft to check against it."""
    rendered = "\n\n".join(
        f"--- POST {index} ---\n{post.content}" for index, post in enumerate(posts, start=1)
    )
    numbered_sources = "\n".join(
        f"[{index}] {source.title} ({source.domain})"
        for index, source in enumerate(research.sources)
    )
    return (
        f"RESEARCH - the only evidence you may accept:\n{research.as_writer_context()}\n\n"
        f"SOURCES (cite by number):\n{numbered_sources or '(none)'}\n\n"
        f"DRAFT TO CHECK ({len(posts)} post(s)):\n\n{rendered}\n\n"
        "Extract each factual claim and assign a verdict. Set post_index to the "
        "post number the claim appears in, and supporting_source_index to the "
        "source that backs it, or -1."
    )
