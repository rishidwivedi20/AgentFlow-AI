"""The typed state carried through the workflow graph.

Why ``TypedDict`` and not a Pydantic ``BaseModel``
-------------------------------------------------
LangGraph nodes return *partial* updates which the framework merges into the
accumulated state, applying a reducer per key. A Pydantic model as the state
root fights that design: it validates whole objects, not fragments. So the
container is a ``TypedDict`` while every value inside it is a validated
Pydantic model - structure where it helps, no ceremony where it does not.

Reducer semantics
-----------------
Keys annotated ``Annotated[list[X], operator.add]`` accumulate: returning
``{"errors": [record]}`` from a node appends rather than replaces. Every other
key uses LangGraph's default, which is last-write-wins.

That distinction is load-bearing. ``revision_count`` must overwrite (it is a
counter, not a log), while ``agent_metrics`` must append (dropping earlier
executions would destroy the observability this project exists to demonstrate).
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.models.content import DraftRecord, Post, WriterOutput
from app.models.enums import Platform, Tone, WorkflowStatus
from app.models.errors import WorkflowErrorRecord
from app.models.factcheck import FactCheckReport
from app.models.metrics import AgentMetric
from app.models.plan import ContentPlan
from app.models.research import ResearchResult
from app.models.review import EditorOutput


class WorkflowState(TypedDict):
    """State threaded through every node of the Writer/Editor graph."""

    # --- Immutable inputs -------------------------------------------------
    # Set once by initial_state() and never written by a node.
    topic: str
    """Subject the content is about."""

    platform: Platform
    """Target platform; selects the prompt variant and formatting rules."""

    tone: Tone
    """Requested voice, passed to the Writer and checked by the Editor."""

    number_of_posts: int
    """How many posts the Writer must produce in each draft."""

    max_revisions: int
    """Hard cap on Writer revisions. Copied from settings so a run's limit is
    recorded in its own state rather than read from global config mid-flight."""

    # --- Research and planning -------------------------------------------
    research: ResearchResult | None
    """Facts and sources gathered by the Researcher. None before it runs, and
    when a search returned nothing."""

    research_context: str | None
    """The research rendered as grounding text for the Writer. Derived from
    ``research``; kept as its own field so the Writer stays decoupled from the
    research model's shape."""

    content_plan: ContentPlan | None
    """One angle per post, assigned by the Planner. None before it runs."""

    fact_check: FactCheckReport | None
    """Verdicts on the latest draft's factual claims. None until the Fact
    Checker has run at least once; replaced on each pass."""

    # --- Loop state (last-write-wins) -------------------------------------
    current_draft: WriterOutput | None
    """Most recent Writer output. None before the Writer has run, and the
    fallback for finalisation if the Editor never produced a verdict."""

    editor_feedback: EditorOutput | None
    """Most recent Editor verdict. Its presence is what tells the Writer node
    that this run is a revision rather than an initial draft."""

    revision_count: int
    """Number of times the Writer has *revised*. The initial draft is 0, so a
    cap of 3 permits at most 4 Writer runs. Incremented by the writer node
    only; routing functions stay free of side effects."""

    approved: bool
    """Whether the latest Editor verdict approved the draft."""

    # --- Accumulated history (append reducers) ----------------------------
    drafts: Annotated[list[DraftRecord], operator.add]
    """Every draft that reached a verdict, paired with its evaluation. Appended
    by the editor node. Source for best-draft selection at the revision cap."""

    agent_metrics: Annotated[list[AgentMetric], operator.add]
    """One entry per agent execution, appended by the metrics wrapper."""

    errors: Annotated[list[WorkflowErrorRecord], operator.add]
    """Client-safe failure records. A non-empty list does not by itself mean
    the run failed - a run can degrade and still return usable posts."""

    # --- Outputs ----------------------------------------------------------
    final_output: list[Post]
    """Posts chosen by the finalize node. Empty when no draft was produced."""

    workflow_status: WorkflowStatus
    """Terminal status, set by the finalize node."""


def initial_state(
    *,
    topic: str,
    platform: Platform,
    tone: Tone,
    number_of_posts: int,
    max_revisions: int,
    research_context: str | None = None,
) -> WorkflowState:
    """Build a complete starting state.

    Every key is populated explicitly. LangGraph does not insert defaults for
    missing keys, so a partially initialised state surfaces as a ``KeyError``
    inside a node - far harder to diagnose than a missing argument here.
    """
    return WorkflowState(
        topic=topic,
        platform=platform,
        tone=tone,
        number_of_posts=number_of_posts,
        max_revisions=max_revisions,
        research=None,
        research_context=research_context,
        content_plan=None,
        fact_check=None,
        current_draft=None,
        editor_feedback=None,
        revision_count=0,
        approved=False,
        drafts=[],
        agent_metrics=[],
        errors=[],
        final_output=[],
        workflow_status=WorkflowStatus.PENDING,
    )


def select_best_draft(state: WorkflowState) -> DraftRecord | None:
    """Return the highest-scoring evaluated draft, or None if there is none.

    Used when the revision cap is reached without approval: the last draft is
    not necessarily the best one, since a revision can score worse than what it
    replaced. Ties resolve to the earliest draft, which is the one the Editor
    scored without the benefit of later feedback.
    """
    if not state["drafts"]:
        return None
    return max(state["drafts"], key=lambda draft: draft.quality_score)
