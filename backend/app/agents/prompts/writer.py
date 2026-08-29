"""Prompt construction for the Writer agent.

Kept separate from the agent so prompt wording can be tuned without touching
execution logic, and so Phase 3 can add research context by editing one
function.
"""

from __future__ import annotations

from app.models.enums import Platform, Tone
from app.models.plan import ContentPlan
from app.models.review import EditorOutput

_PLATFORM_RULES: dict[Platform, str] = {
    Platform.LINKEDIN: (
        "- Open with a hook of at most two lines; LinkedIn truncates the rest behind "
        '"see more".\n'
        "- Use short paragraphs of one to three lines separated by blank lines.\n"
        "- Aim for 120-220 words per post.\n"
        "- Plain text only: no markdown headings, bold markers or bullet characters "
        "other than simple dashes.\n"
        "- Close with a question or invitation that invites replies.\n"
        "- Suggest three to five relevant hashtags, without the '#' prefix."
    )
}


def build_system_prompt(platform: Platform, tone: Tone) -> str:
    """Static instructions describing the Writer's role and constraints."""
    return (
        "You are an expert content writer producing posts for a professional audience.\n\n"
        f"PLATFORM: {platform.value}\n"
        f"{_PLATFORM_RULES[platform]}\n\n"
        f"TONE: {tone.value}. Sustain it across every post.\n\n"
        "HARD CONSTRAINTS:\n"
        "- Never invent statistics, percentages, dates, company names, study results or "
        "quotations. If you have no verified source for a number, make the point "
        "qualitatively instead.\n"
        "- Every post must take a genuinely different angle. Do not restate the same "
        "idea with new wording.\n"
        "- Write finished copy, not outlines or placeholders.\n"
        "- Set 'angle' to a short label describing what makes that post distinct."
    )


def build_user_prompt(
    *,
    topic: str,
    number_of_posts: int,
    research_context: str | None = None,
    content_plan: ContentPlan | None = None,
    feedback: EditorOutput | None = None,
    fact_check_feedback: str | None = None,
) -> str:
    """Task instructions for one Writer run.

    When ``feedback`` is present this is a revision: the previous draft's
    critique is included and the Writer is told to address it. When
    ``research_context`` is present (Phase 3) the Writer must ground claims in
    it rather than in general knowledge.
    """
    sections = [f"TOPIC: {topic}", f"Write exactly {number_of_posts} post(s)."]

    if research_context:
        sections.append(
            "RESEARCH CONTEXT - the only source you may draw factual claims from:\n"
            f"{research_context}"
        )

    if feedback is not None:
        sections.append(_render_revision_request(feedback))

    return "\n\n".join(sections)


def _render_revision_request(feedback: EditorOutput) -> str:
    """Turn an Editor verdict into concrete revision instructions."""
    lines = [
        "This is a REVISION. An editor reviewed your previous draft and scored it "
        f"{feedback.quality_score:.1f}/10. Rewrite the posts so every point below is "
        "resolved. Keep what already worked."
    ]

    if feedback.issues:
        lines.append("\nISSUES FOUND:")
        lines.extend(
            f"- [{issue.severity.value}] {issue.dimension.value}: {issue.description}"
            for issue in feedback.issues
        )

    if feedback.feedback:
        lines.append("\nREQUIRED CHANGES:")
        lines.extend(f"- {item}" for item in feedback.feedback)

    return "\n".join(lines)
