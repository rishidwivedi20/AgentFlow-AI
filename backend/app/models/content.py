"""Generated content: the Writer's output and the evaluated draft history."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models.review import EditorOutput


class Post(BaseModel):
    """A single social post.

    LLM-facing: every field is required and none are nullable. See
    :class:`app.models.review.EditorOutput` for the full rationale.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    angle: str = Field(
        min_length=1,
        description=(
            "One short line naming this post's distinct angle. Lets the Editor detect "
            "repetition across posts, and is what the Phase 3 Planner will assign."
        ),
    )
    content: str = Field(min_length=1, description="Full post body, ready to publish.")
    hashtags: list[str] = Field(
        description="Hashtags without the leading '#'. Empty list when none apply."
    )


class WriterOutput(BaseModel):
    """One complete set of posts produced by a single Writer run."""

    model_config = ConfigDict(extra="forbid")

    posts: list[Post] = Field(min_length=1, description="Generated posts, in presentation order.")


class DraftRecord(BaseModel):
    """A draft together with the Editor's verdict on it.

    Appended by the *editor* node rather than the writer node: the Editor is the
    only participant that knows the quality score, so recording both together
    avoids two parallel lists that must stay index-aligned. This is what makes
    "finalise the best draft" a plain ``max`` over one collection.
    """

    revision: int = Field(ge=0, description="0 for the initial draft, then 1, 2, 3.")
    posts: list[Post] = Field(description="The draft exactly as the Writer produced it.")
    evaluation: EditorOutput = Field(description="The Editor's verdict on this draft.")

    @property
    def quality_score(self) -> float:
        """Convenience accessor used when selecting the best draft."""
        return self.evaluation.quality_score
