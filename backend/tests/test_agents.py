"""Agent behaviour: prompt construction and separation of concerns."""

from __future__ import annotations

from app.agents.editor import EDITOR_TEMPERATURE, run_editor
from app.agents.prompts import editor as editor_prompts
from app.agents.prompts import writer as writer_prompts
from app.agents.writer import WRITER_TEMPERATURE, run_writer
from app.models.content import WriterOutput
from app.models.enums import Platform, Tone
from app.models.review import EditorOutput
from tests.conftest import StubProvider, make_posts, make_service, make_verdict

REVIEW_DIMENSIONS = (
    "clarity",
    "structure",
    "engagement",
    "repetition",
    "relevance",
    "tone_consistency",
)


class RecordingProvider(StubProvider):
    """Captures the arguments each call was made with."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.captured = []

    async def generate_structured(self, **kwargs):
        self.captured.append(kwargs)
        return await super().generate_structured(**kwargs)


# --- prompt construction ---------------------------------------------------


def test_initial_writer_prompt_has_no_revision_language():
    prompt = writer_prompts.build_user_prompt(topic="AI", number_of_posts=3)
    assert "REVISION" not in prompt
    assert "exactly 3 post(s)" in prompt


def test_revision_prompt_includes_issues_and_feedback():
    verdict = make_verdict(approved=False, score=5.0)
    prompt = writer_prompts.build_user_prompt(topic="AI", number_of_posts=2, feedback=verdict)
    assert "REVISION" in prompt
    assert "5.0/10" in prompt
    assert verdict.issues[0].description in prompt
    assert verdict.feedback[0] in prompt


def test_research_context_is_rendered_when_supplied():
    """Phase 3 seam: the Writer already grounds claims in research."""
    prompt = writer_prompts.build_user_prompt(
        topic="AI", number_of_posts=1, research_context="FACT: adoption rose."
    )
    assert "RESEARCH CONTEXT" in prompt
    assert "FACT: adoption rose." in prompt


def test_research_context_absent_by_default():
    prompt = writer_prompts.build_user_prompt(topic="AI", number_of_posts=1)
    assert "RESEARCH CONTEXT" not in prompt


def test_writer_system_prompt_forbids_invented_facts():
    prompt = writer_prompts.build_system_prompt(Platform.LINKEDIN, Tone.PROFESSIONAL)
    assert "Never invent statistics" in prompt
    assert "professional" in prompt


def test_editor_system_prompt_forbids_rewriting():
    prompt = editor_prompts.build_system_prompt(Platform.LINKEDIN, Tone.ANALYTICAL)
    assert "NEVER rewrite" in prompt
    for dimension in REVIEW_DIMENSIONS:
        assert dimension in prompt


def test_editor_user_prompt_renders_every_post():
    posts = make_posts(3, tag="x")
    prompt = editor_prompts.build_user_prompt(topic="AI", posts=posts, revision=2)
    assert "POST 1" in prompt
    assert "POST 3" in prompt
    assert "revision 2" in prompt
    for post in posts:
        assert post.content in prompt


# --- agent execution -------------------------------------------------------


async def test_writer_requests_writer_schema_with_high_temperature():
    provider = RecordingProvider()
    result = await run_writer(
        make_service(provider),
        topic="AI",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=2,
    )
    assert isinstance(result.value, WriterOutput)
    call = provider.captured[0]
    assert call["schema"] is WriterOutput
    assert call["temperature"] == WRITER_TEMPERATURE
    assert WRITER_TEMPERATURE > 0.5


async def test_editor_requests_editor_schema_with_low_temperature():
    provider = RecordingProvider()
    result = await run_editor(
        make_service(provider),
        topic="AI",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        posts=make_posts(2),
        revision=0,
    )
    assert isinstance(result.value, EditorOutput)
    call = provider.captured[0]
    assert call["schema"] is EditorOutput
    assert call["temperature"] == EDITOR_TEMPERATURE
    assert EDITOR_TEMPERATURE < 0.5


async def test_writer_tolerates_post_count_mismatch(caplog):
    """A count slip is logged, not fatal: failing would spend the revision budget."""
    provider = RecordingProvider(post_count=2)
    result = await run_writer(
        make_service(provider),
        topic="AI",
        platform=Platform.LINKEDIN,
        tone=Tone.PROFESSIONAL,
        number_of_posts=5,
    )
    assert len(result.value.posts) == 2
    assert "unexpected post count" in caplog.text
