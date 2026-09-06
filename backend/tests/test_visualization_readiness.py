"""Guards the contract a Phase 4 workflow visualization will depend on.

No SSE and no React Flow here. These tests only pin down the backend facts a
live graph view needs, so a later refactor cannot quietly break them.
"""

from __future__ import annotations

import pytest

from app.graph.workflow import build_workflow
from app.models.enums import AgentName, AgentStatus, WorkflowStatus
from tests.conftest import StubProvider, make_research, make_service, make_verdict, run_pipeline


def _graph():
    return build_workflow(make_service(StubProvider()), make_research()).get_graph()


def test_graph_node_ids_match_agent_names():
    """A client maps metric.agent -> node id directly, with no lookup table."""
    node_ids = {n for n in _graph().nodes if not n.startswith("__")}
    assert node_ids == {a.value for a in AgentName}


def test_graph_topology_is_self_describing():
    """LangGraph reports its own nodes and edges, so no hand-kept copy exists."""
    graph = _graph()
    edges = {(e.source, e.target) for e in graph.edges}

    assert ("__start__", "retriever") in edges
    assert ("retriever", "researcher") in edges
    assert ("researcher", "planner") in edges
    assert ("planner", "writer") in edges
    assert ("writer", "editor") in edges
    assert ("editor", "fact_checker") in edges
    assert ("fact_checker", "finalize") in edges
    assert ("finalize", "__end__") in edges


@pytest.mark.parametrize("source", ["editor", "fact_checker"])
def test_feedback_loop_edges_are_present_and_conditional(source):
    """Both critics can send work back; the UI draws each as a revision loop."""
    loop = [e for e in _graph().edges if (e.source, e.target) == (source, "writer")]
    assert loop, f"the {source} -> writer revision edge must exist"
    assert loop[0].conditional is True
    assert loop[0].data == "revise"


async def test_metrics_reconstruct_the_loop_in_order(settings, run_kwargs):
    """Each pass through the loop is individually identifiable."""
    provider = StubProvider(
        verdicts=[
            make_verdict(approved=False, score=5.0),
            make_verdict(approved=False, score=6.0),
            make_verdict(approved=True, score=9.0),
        ]
    )
    state = await run_workflow_helper(provider, settings, run_kwargs)

    metrics = state["agent_metrics"]
    assert [m.agent.value for m in metrics] == [
        "retriever",
        "researcher",
        "planner",
        "writer",
        "editor",
        "writer",
        "editor",
        "writer",
        "editor",
        "fact_checker",
        "finalize",
    ]
    # Sequence is dense, gapless and ordered: usable as a stream cursor.
    assert [m.sequence for m in metrics] == list(range(len(metrics)))


async def test_sequence_distinguishes_repeated_runs_of_the_same_node(settings, run_kwargs):
    """Without this, three Writer events are indistinguishable in a live stream."""
    provider = StubProvider(verdicts=[make_verdict(approved=False, score=5.0)])
    state = await run_workflow_helper(provider, settings, run_kwargs)

    writer_runs = [m for m in state["agent_metrics"] if m.agent is AgentName.WRITER]
    assert len(writer_runs) == 4
    assert len({m.sequence for m in writer_runs}) == 4, "sequences must be unique"


async def test_failed_runs_are_also_sequenced(settings, run_kwargs):
    provider = StubProvider(fail_on=["writer"])
    state = await run_workflow_helper(provider, settings, run_kwargs)

    metrics = state["agent_metrics"]
    assert [m.sequence for m in metrics] == list(range(len(metrics)))
    writer_metric = next(m for m in metrics if m.agent is AgentName.WRITER)
    assert writer_metric.status is AgentStatus.FAILED


def test_agent_status_covers_every_ui_node_state():
    """The four node states the visualization needs share one vocabulary."""
    assert {s.value for s in AgentStatus} == {"pending", "running", "completed", "failed"}


def test_only_terminal_statuses_are_recordable():
    assert AgentStatus.COMPLETED.is_terminal
    assert AgentStatus.FAILED.is_terminal
    assert not AgentStatus.PENDING.is_terminal
    assert not AgentStatus.RUNNING.is_terminal


def test_workflow_status_terminality_is_explicit():
    assert WorkflowStatus.COMPLETED.is_terminal
    assert WorkflowStatus.FAILED.is_terminal
    assert WorkflowStatus.MAX_REVISIONS_REACHED.is_terminal
    assert not WorkflowStatus.RUNNING.is_terminal
    assert not WorkflowStatus.PENDING.is_terminal


async def run_workflow_helper(provider, settings, run_kwargs):
    return await run_pipeline(provider, settings, run_kwargs)
