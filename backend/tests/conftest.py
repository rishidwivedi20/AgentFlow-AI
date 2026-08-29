"""Shared fixtures and stub provider.

``StubProvider`` implements the ``LLMProvider`` protocol by replaying a
scripted sequence of behaviours. That makes the Writer/Editor loop fully
deterministic: a test states which drafts get approved and which do not, and
the graph does the rest.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest

from app.core.config import Settings
from app.models.content import Post, WriterOutput
from app.models.enums import Platform, Tone
from app.models.metrics import TokenUsage
from app.models.review import EditorIssue, EditorOutput, IssueSeverity, ReviewDimension
from app.services.llm.base import LLMResult
from app.services.llm_service import LLMService

STUB_MODEL = "stub-model"


def make_posts(count: int = 2, *, tag: str = "d") -> list[Post]:
    return [
        Post(angle=f"angle-{tag}-{i}", content=f"content-{tag}-{i}", hashtags=["ai"])
        for i in range(count)
    ]


def make_verdict(*, approved: bool, score: float) -> EditorOutput:
    issues = (
        []
        if approved
        else [
            EditorIssue(
                dimension=ReviewDimension.CLARITY,
                severity=IssueSeverity.MEDIUM,
                description="The opening buries the point.",
            )
        ]
    )
    return EditorOutput(
        approved=approved,
        quality_score=score,
        issues=issues,
        feedback=[] if approved else ["Lead with the conclusion."],
    )


def make_usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=100, output_tokens=200, thought_tokens=50, total_tokens=350)


class StubProvider:
    """Replays scripted verdicts; raises on demand to simulate agent failure.

    ``verdicts`` drives Editor responses in order. Once exhausted the last
    verdict repeats, so a test need only script what it cares about.
    """

    def __init__(
        self,
        *,
        verdicts: Sequence[EditorOutput] | None = None,
        post_count: int = 2,
        fail_on: Sequence[str] = (),
    ) -> None:
        self._verdicts = list(verdicts or [make_verdict(approved=True, score=9.0)])
        self._post_count = post_count
        self._fail_on = set(fail_on)
        self.writer_calls = 0
        self.editor_calls = 0

    name = "stub"
    model = STUB_MODEL

    def _next_verdict(self) -> EditorOutput:
        index = min(self.editor_calls - 1, len(self._verdicts) - 1)
        return self._verdicts[index]

    async def generate_structured(self, *, schema, **_: object) -> LLMResult:
        if schema is WriterOutput:
            self.writer_calls += 1
            if "writer" in self._fail_on:
                raise RuntimeError("stub writer failure")
            return LLMResult(
                value=WriterOutput(posts=make_posts(self._post_count, tag=f"r{self.writer_calls}")),
                usage=make_usage(),
                model=STUB_MODEL,
            )

        self.editor_calls += 1
        if "editor" in self._fail_on:
            raise RuntimeError("stub editor failure")
        return LLMResult(value=self._next_verdict(), usage=make_usage(), model=STUB_MODEL)


def make_service(provider: StubProvider, *, timeout: float = 5.0, retries: int = 0) -> LLMService:
    return LLMService(provider, timeout_seconds=timeout, max_retries=retries)


@pytest.fixture
def settings() -> Settings:
    """Settings for a test run: a key is present so config checks pass."""
    return Settings(
        gemini_api_key="test-key",
        llm_provider="gemini",
        max_revisions=3,
        app_env="test",
    )


@pytest.fixture
def run_kwargs() -> dict[str, object]:
    return {
        "topic": "How AI is changing software engineering",
        "platform": Platform.LINKEDIN,
        "tone": Tone.PROFESSIONAL,
        "number_of_posts": 2,
    }


@pytest.fixture
def approving_provider() -> StubProvider:
    return StubProvider(verdicts=[make_verdict(approved=True, score=9.0)])


@pytest.fixture
def rejecting_provider() -> StubProvider:
    """Never approves, forcing the loop to the revision cap."""
    return StubProvider(verdicts=[make_verdict(approved=False, score=5.0)])


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Keep the cached settings singleton from leaking between tests."""
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
