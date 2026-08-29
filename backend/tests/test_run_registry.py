"""Run registry: buffering, replay, subscriber fan-out and retention."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.models.enums import AgentName, AgentStatus
from app.models.events import EventType, WorkflowEvent
from app.services.run_registry import QUEUE_MAXSIZE, RunRegistry, RunState


def event(sequence: int, event_type: EventType = EventType.AGENT_STARTED) -> WorkflowEvent:
    return WorkflowEvent(
        sequence=sequence,
        type=event_type,
        timestamp=datetime.now(UTC),
        agent=AgentName.WRITER,
        status=AgentStatus.RUNNING,
    )


async def drain(registry: RunRegistry, run_id: str, *, cursor: int = -1) -> list[WorkflowEvent]:
    return [e async for e in registry.subscribe(run_id, cursor=cursor)]


# --- lifecycle --------------------------------------------------------------


def test_create_assigns_a_unique_id():
    registry = RunRegistry()
    first = registry.create(topic="a")
    second = registry.create(topic="b")
    assert first.run_id != second.run_id
    assert first.state is RunState.QUEUED
    assert len(registry) == 2


def test_get_returns_none_for_unknown_run():
    assert RunRegistry().get("nope") is None


def test_state_transitions():
    registry = RunRegistry()
    record = registry.create(topic="t")
    registry.mark_running(record.run_id)
    assert record.state is RunState.RUNNING
    assert record.finished_at is None

    registry.fail(record.run_id, error_type="rate_limited", message="slow down")
    assert record.state is RunState.FAILED
    assert record.finished_at is not None
    assert record.error_type == "rate_limited"


def test_run_state_finishedness():
    assert not RunState.QUEUED.is_finished
    assert not RunState.RUNNING.is_finished
    assert RunState.COMPLETED.is_finished
    assert RunState.FAILED.is_finished


# --- buffering and replay ---------------------------------------------------


async def test_events_are_buffered_with_no_subscribers():
    """Replay must work for a client that connects after the run started."""
    registry = RunRegistry()
    record = registry.create(topic="t")
    for i in range(3):
        await registry.publish(record.run_id, event(i))
    assert len(record.events) == 3


async def test_finished_run_replays_everything_then_ends():
    registry = RunRegistry()
    record = registry.create(topic="t")
    for i in range(3):
        await registry.publish(record.run_id, event(i))
    registry.fail(record.run_id, error_type="x", message="y")

    received = await drain(registry, record.run_id)
    assert [e.sequence for e in received] == [0, 1, 2]


async def test_cursor_replays_only_later_events():
    registry = RunRegistry()
    record = registry.create(topic="t")
    for i in range(5):
        await registry.publish(record.run_id, event(i))
    registry.fail(record.run_id, error_type="x", message="y")

    assert [e.sequence for e in await drain(registry, record.run_id, cursor=2)] == [3, 4]


async def test_cursor_beyond_the_end_yields_nothing():
    registry = RunRegistry()
    record = registry.create(topic="t")
    await registry.publish(record.run_id, event(0))
    registry.fail(record.run_id, error_type="x", message="y")

    assert await drain(registry, record.run_id, cursor=99) == []


async def test_subscribing_to_an_unknown_run_yields_nothing():
    assert await drain(RunRegistry(), "missing") == []


# --- live streaming ---------------------------------------------------------


async def test_live_events_reach_an_open_subscriber():
    registry = RunRegistry()
    record = registry.create(topic="t")
    registry.mark_running(record.run_id)

    received: list[WorkflowEvent] = []

    async def consume():
        async for e in registry.subscribe(record.run_id):
            received.append(e)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)  # let the subscriber attach

    for i in range(3):
        await registry.publish(record.run_id, event(i))
    registry.complete(record.run_id, None)
    await asyncio.wait_for(task, timeout=2)

    assert [e.sequence for e in received] == [0, 1, 2]


async def test_replay_then_live_has_no_gap_and_no_duplicate():
    """The buffer and the queue overlap by design; sequence suppresses it."""
    registry = RunRegistry()
    record = registry.create(topic="t")
    registry.mark_running(record.run_id)
    for i in range(2):
        await registry.publish(record.run_id, event(i))  # buffered before subscribing

    received: list[WorkflowEvent] = []

    async def consume():
        async for e in registry.subscribe(record.run_id):
            received.append(e)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    for i in range(2, 5):
        await registry.publish(record.run_id, event(i))
    registry.complete(record.run_id, None)
    await asyncio.wait_for(task, timeout=2)

    sequences = [e.sequence for e in received]
    assert sequences == [0, 1, 2, 3, 4], "ordered, gapless, no repeats"


async def test_two_subscribers_both_receive_every_event():
    registry = RunRegistry()
    record = registry.create(topic="t")
    registry.mark_running(record.run_id)

    a: list[int] = []
    b: list[int] = []

    async def consume(sink):
        async for e in registry.subscribe(record.run_id):
            sink.append(e.sequence)

    tasks = [asyncio.create_task(consume(a)), asyncio.create_task(consume(b))]
    await asyncio.sleep(0)

    for i in range(3):
        await registry.publish(record.run_id, event(i))
    registry.complete(record.run_id, None)
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)

    assert a == b == [0, 1, 2]


async def test_completing_a_run_releases_a_waiting_subscriber():
    """Without the close sentinel this would hang forever."""
    registry = RunRegistry()
    record = registry.create(topic="t")
    registry.mark_running(record.run_id)

    async def consume():
        return [e async for e in registry.subscribe(record.run_id)]

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    registry.complete(record.run_id, None)

    assert await asyncio.wait_for(task, timeout=2) == []


async def test_subscriber_is_removed_when_the_stream_closes():
    registry = RunRegistry()
    record = registry.create(topic="t")
    registry.mark_running(record.run_id)

    stream = registry.subscribe(record.run_id)
    await registry.publish(record.run_id, event(0))
    await stream.__anext__()
    assert len(record.subscribers) == 1

    await stream.aclose()
    assert record.subscribers == set(), "a closed stream must not leak a queue"


async def test_slow_subscriber_is_dropped_rather_than_stalling_the_run():
    registry = RunRegistry()
    record = registry.create(topic="t")
    registry.mark_running(record.run_id)

    stream = registry.subscribe(record.run_id)
    # A queue attached directly, simulating a consumer that never reads.
    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    record.subscribers.add(queue)

    for i in range(QUEUE_MAXSIZE + 5):
        await registry.publish(record.run_id, event(i))

    assert queue not in record.subscribers, "the stalled subscriber was dropped"
    assert len(record.events) == QUEUE_MAXSIZE + 5, "buffering continued regardless"
    await stream.aclose()


# --- retention --------------------------------------------------------------


def test_expired_finished_runs_are_evicted():
    registry = RunRegistry(ttl=timedelta(minutes=60))
    record = registry.create(topic="old")
    registry.complete(record.run_id, None)
    record.finished_at = datetime.now(UTC) - timedelta(minutes=61)

    assert registry.evict_expired() == 1
    assert registry.get(record.run_id) is None


def test_unexpired_runs_survive_eviction():
    registry = RunRegistry(ttl=timedelta(minutes=60))
    record = registry.create(topic="recent")
    registry.complete(record.run_id, None)
    record.finished_at = datetime.now(UTC) - timedelta(minutes=59)

    assert registry.evict_expired() == 0
    assert registry.get(record.run_id) is not None


def test_running_runs_are_never_evicted_by_ttl():
    """Evicting live work would orphan its task and 404 a streaming client."""
    registry = RunRegistry(ttl=timedelta(minutes=1))
    record = registry.create(topic="long")
    registry.mark_running(record.run_id)
    record.created_at = datetime.now(UTC) - timedelta(hours=5)

    assert registry.evict_expired() == 0
    assert registry.get(record.run_id) is not None


def test_count_limit_evicts_the_oldest_finished_run():
    registry = RunRegistry(max_runs=3)
    finished = []
    for i in range(3):
        record = registry.create(topic=f"t{i}")
        registry.complete(record.run_id, None)
        record.finished_at = datetime.now(UTC) - timedelta(minutes=10 - i)
        finished.append(record)

    registry.create(topic="newest")
    assert len(registry) == 3
    assert registry.get(finished[0].run_id) is None, "oldest finished run evicted"
    assert registry.get(finished[2].run_id) is not None


def test_count_limit_does_not_evict_running_runs():
    registry = RunRegistry(max_runs=2)
    running = [registry.create(topic=f"r{i}") for i in range(3)]
    for record in running:
        registry.mark_running(record.run_id)

    registry.create(topic="another")
    # Nothing finished, so nothing can be evicted; the cap is exceeded rather
    # than live work being destroyed.
    assert all(registry.get(r.run_id) is not None for r in running)


def test_creating_a_run_triggers_expiry_eviction():
    registry = RunRegistry(ttl=timedelta(minutes=60))
    stale = registry.create(topic="stale")
    registry.complete(stale.run_id, None)
    stale.finished_at = datetime.now(UTC) - timedelta(hours=2)

    registry.create(topic="fresh")
    assert registry.get(stale.run_id) is None


@pytest.mark.parametrize("state", [RunState.QUEUED, RunState.RUNNING])
def test_age_measures_from_creation_while_unfinished(state):
    registry = RunRegistry()
    record = registry.create(topic="t")
    record.state = state
    record.created_at = datetime.now(UTC) - timedelta(minutes=30)
    assert record.age >= timedelta(minutes=29)
