"""Content plan: the distinct angle each post must take.

Exists to stop the Writer producing N restatements of one idea. In Phase 2 the
Writer chose its own angles inside a single call; from Phase 3 the Planner
assigns them up front, which makes repetition a checkable property rather than
a hope.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ContentType(StrEnum):
    """The shape a post takes.

    A fixed vocabulary so the Planner is pushed towards genuine variety instead
    of four posts that differ only in wording.
    """

    EDUCATIONAL = "educational"
    """Explains a concept or mechanism."""

    STORYTELLING = "storytelling"
    """Frames the topic through a narrative or scenario."""

    CAREER = "career"
    """What the topic means for someone's skills or career."""

    OPINION = "opinion"
    """Takes a defensible position and argues it."""

    PRACTICAL = "practical"
    """Concrete steps, checklists or things to try."""


class ContentAngle(BaseModel):
    """The brief for exactly one post."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    post_number: int = Field(
        ge=1, description="Which post this brief is for, numbered from 1."
    )
    angle: str = Field(
        min_length=1,
        description="One line naming what makes this post distinct from the others.",
    )
    content_type: ContentType = Field(description="The shape this post should take.")
    hook_strategy: str = Field(
        min_length=1,
        description="How the opening should earn attention, in one line. Not the hook itself.",
    )
    key_points: list[str] = Field(
        description="Two to four points this post should cover, drawn from the research."
    )
    target_audience: str = Field(
        min_length=1, description="Who this post speaks to, specifically."
    )
    intended_takeaway: str = Field(
        min_length=1, description="The one thing a reader should leave with."
    )


class ContentPlan(BaseModel):
    """One angle per requested post.

    Count is validated against the request in the planner agent rather than
    here: the schema cannot know how many posts were asked for.
    """

    model_config = ConfigDict(extra="forbid")

    angles: list[ContentAngle] = Field(
        min_length=1, description="Angles in the order the posts should be written."
    )

    def as_writer_brief(self) -> str:
        """Render the plan as instructions for the Writer prompt."""
        lines = []
        for index, angle in enumerate(self.angles, start=1):
            points = "\n".join(f"    - {point}" for point in angle.key_points)
            lines.append(
                f"POST {index} ({angle.content_type.value})\n"
                f"  angle: {angle.angle}\n"
                f"  cover:\n{points}"
            )
        return "\n\n".join(lines)
