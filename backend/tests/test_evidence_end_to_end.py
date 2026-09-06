"""Evidence through a whole run, asserted on the prompts each agent saw.

Runs the real graph to check which agents end up with untrusted page text in
their context. Rendering and repair are covered in isolation elsewhere.
"""

from __future__ import annotations

import pytest

from app.models.enums import WorkflowStatus
from tests.conftest import (
    StubEmbeddingProvider,
    StubSearchProvider,
    make_retrieval,
    make_sources,
    run_pipeline,
)
from tests.test_evidence_prompt import INJECTION

# The stub embedder scores on shared vocabulary, so "alpha" makes this page
# rank against the topic exactly as a genuinely relevant page would.
POISONED_PAGE = f"alpha alpha beta {INJECTION} alpha gamma"
CLEAN_PAGE = "alpha beta gamma delta alpha beta adoption rose across teams"


@pytest.fixture
def run_kwargs_alpha(run_kwargs):
    return {**run_kwargs, "topic": "alpha beta"}


async def test_untrusted_page_text_reaches_the_analysts_but_not_the_writer(
    approving_provider, settings, run_kwargs_alpha
):
    """A page carrying an injection is retrieved, ranked, and passed to the two
    agents that emit constrained structured output. The Writer gets the
    Researcher's extracted facts instead.
    """
    search = StubSearchProvider(
        sources=make_sources(2), page_text={0: POISONED_PAGE, 1: CLEAN_PAGE}
    )
    state = await run_pipeline(
        approving_provider,
        settings,
        run_kwargs_alpha,
        search=search,
        retrieval=make_retrieval(StubEmbeddingProvider()),
    )

    assert state["workflow_status"] is WorkflowStatus.COMPLETED
    assert state["evidence"] is not None
    assert state["evidence"].is_usable

    prompts = approving_provider.prompts
    assert INJECTION in prompts["researcher"][0]
    assert INJECTION in prompts["fact_checker"][0]

    for stage in ("writer", "planner", "editor"):
        for prompt in prompts.get(stage, []):
            assert INJECTION not in prompt, f"untrusted text reached the {stage}"


async def test_retrieved_text_is_fenced_and_labelled_in_every_prompt_that_gets_it(
    approving_provider, settings, run_kwargs_alpha
):
    search = StubSearchProvider(sources=make_sources(2), page_text={0: POISONED_PAGE})
    await run_pipeline(
        approving_provider,
        settings,
        run_kwargs_alpha,
        search=search,
        retrieval=make_retrieval(StubEmbeddingProvider()),
    )

    for stage in ("researcher", "fact_checker"):
        prompt = approving_provider.prompts[stage][0]
        assert "UNTRUSTED DATA" in prompt
        # One fence, closed once: the page could not have broken out of it.
        assert prompt.count("<<<BEGIN_SOURCE_EXCERPTS") == 1
        assert prompt.count("END_SOURCE_EXCERPTS>>>") == 1


async def test_a_run_without_retrieval_tells_the_agents_so(
    approving_provider, settings, run_kwargs_alpha
):
    """Otherwise the model must guess whether the sources were empty or the
    step never ran."""
    state = await run_pipeline(
        approving_provider,
        settings,
        run_kwargs_alpha,
        search=StubSearchProvider(sources=make_sources(2)),
        retrieval=None,
    )

    assert state["workflow_status"] is WorkflowStatus.COMPLETED
    assert "did not run" in approving_provider.prompts["researcher"][0]
    assert "<<<BEGIN_SOURCE_EXCERPTS" not in approving_provider.prompts["researcher"][0]
