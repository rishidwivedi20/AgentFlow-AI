"""Prompt construction for the Editor agent."""

from __future__ import annotations

from app.models.content import Post
from app.models.enums import Platform, Tone

APPROVAL_THRESHOLD = 7.5
"""Score at or above which a draft may be approved. Stated in the prompt so the
Editor applies one consistent bar across revisions."""


def build_system_prompt(platform: Platform, tone: Tone) -> str:
    """Static instructions describing the Editor's role and rubric."""
    return (
        "You are a demanding content editor. You review drafts and you NEVER rewrite "
        "them. Your entire output is a structured critique.\n\n"
        f"The draft targets {platform.value} and must sustain a {tone.value} tone.\n\n"
        "Assess exactly these dimensions:\n"
        "- clarity: is each point immediately understandable?\n"
        "- structure: hook, flow, paragraphing, closing?\n"
        "- engagement: would a reader stop scrolling and reply?\n"
        "- repetition: do posts overlap in idea, phrasing or example?\n"
        "- relevance: does every post genuinely address the topic?\n"
        "- tone_consistency: is the requested tone held throughout?\n\n"
        "SCORING (0-10):\n"
        "- 9-10 publishable as-is\n"
        "- 7.5-8.9 strong, minor polish only\n"
        "- 5-7.4 usable but needs real revision\n"
        "- below 5 significant problems\n\n"
        f"Set approved=true only when the draft scores at least {APPROVAL_THRESHOLD} AND has "
        "no high-severity issues. Otherwise set approved=false.\n\n"
        "RULES:\n"
        "- Do not produce rewritten sentences, replacement copy or examples of better "
        "wording. Describe what must change and why; the writer performs the rewrite.\n"
        "- Each feedback item must be a specific, actionable instruction.\n"
        "- Report only issues you can point to in the draft. Do not invent faults to "
        "look thorough. An excellent draft may legitimately have zero issues.\n"
        "- Flag any factual claim, statistic or citation that appears unsupported."
    )


def build_user_prompt(*, topic: str, posts: list[Post], revision: int) -> str:
    """Present the draft for review."""
    header = (
        f"TOPIC THE DRAFT MUST ADDRESS: {topic}\n"
        f"This is revision {revision} ({'initial draft' if revision == 0 else 'a rewrite'})."
    )
    rendered = "\n\n".join(
        f"--- POST {index} ---\n"
        f"angle: {post.angle}\n"
        f"hashtags: {', '.join(post.hashtags) if post.hashtags else '(none)'}\n\n"
        f"{post.content}"
        for index, post in enumerate(posts, start=1)
    )
    return f"{header}\n\nDRAFT ({len(posts)} post(s)):\n\n{rendered}"
