"""In-memory store of workflow runs and their event streams.

SINGLE WORKER ONLY
------------------
Runs, their buffered events and their subscriber queues live in this process's
memory. With more than one uvicorn worker a client can be routed to a process
that has never heard of its run, and will get a 404 for a run executing
elsewhere. Run the API with ``--workers 1``.

This is a deliberate MVP trade: no database, no Redis, no broker. The registry
is one class with a narrow surface, so swapping the storage later touches this
file alone.

Retention
---------
Bounded twice over, because either bound alone leaks. A count cap alone keeps
stale runs alive indefinitely on a quiet server; a time cap alone lets a burst
exhaust memory before anything expires. A run still executing is never evicted,
however old - losing its entry would orphan the task and 404 a live client.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from app.models.events import WorkflowEvent

if TYPE_CHECKING:
    from app.schemas.workflow import WorkflowRunResponse

logger = logging.getLogger(__name__)

MAX_RUNS: Final = 20
"""Most runs retained. The oldest finished run is evicted beyond this."""

RUN_TTL: Final = timedelta(minutes=60)
"""How long a finished run stays readable before eviction."""

QUEUE_MAXSIZE: Final = 256
"""Per-subscriber backlog. A client too slow to keep up is dropped rather than
allowed to grow this without bound or stall the run."""

# Pushed to a subscriber queue when the run ends, so a waiting stream wakes up
# instead of blocking forever on a queue nothing will write to again.
_CLOSE_SENTINEL: Final = None

QueueItem = WorkflowEvent | None


class RunState(StrEnum):
    """Lifecycle of a run, distinct from the workflow's own status.

    A run can finish as ``COMPLETED`` while its workflow reports failure: the
    execution completed, the content did not succeed. Conflating the two would
    make a crashed process indistinguishable from unusable output.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_finished(self) -> bool:
        return self in {RunState.COMPLETED, RunState.FAILED}


@dataclass
class RunRecord:
    """One workflow run: its state, its events, and who is watching."""

    run_id: str
    created_at: datetime
    state: RunState = RunState.QUEUED
    topic: str = ""
    events: list[WorkflowEvent] = field(default_factory=list)
    result: WorkflowRunResponse | None = None
    error_type: str | None = None
    error_message: str | None = None
    finished_at: datetime | None = None
    subscribers: set[asyncio.Queue[QueueItem]] = field(default_factory=set)
    task: asyncio.Task[Any] | None = None
    """Held so the background task is not garbage collected mid-run."""

    @property
    def age(self) -> timedelta:
        """Time since the run finished, or since it was created if still going."""
        return datetime.now(UTC) - (self.finished_at or self.created_at)


class RunRegistry:
    """Creates runs, buffers their events, and fans them out to subscribers."""

    def __init__(self, *, max_runs: int = MAX_RUNS, ttl: timedelta = RUN_TTL) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._max_runs = max_runs
        self._ttl = ttl

    # --- lifecycle --------------------------------------------------------

    def create(self, *, topic: str) -> RunRecord:
        """Register a new run, evicting expired ones first."""
        self.evict_expired()
        record = RunRecord(run_id=uuid.uuid4().hex, created_at=datetime.now(UTC), topic=topic)
        self._runs[record.run_id] = record
        self._enforce_max_runs()
        return record

    def get(self, run_id: str) -> RunRecord | None:
        """Return a run, or None if unknown or already evicted."""
        return self._runs.get(run_id)

    def mark_running(self, run_id: str) -> None:
        record = self._runs.get(run_id)
        if record is not None:
            record.state = RunState.RUNNING

    def complete(self, run_id: str, result: WorkflowRunResponse) -> None:
        """Record a finished execution and release every subscriber."""
        record = self._runs.get(run_id)
        if record is None:
            return
        record.state = RunState.COMPLETED
        record.result = result
        record.finished_at = datetime.now(UTC)
        self._release_subscribers(record)

    def fail(self, run_id: str, *, error_type: str, message: str) -> None:
        """Record a failed execution. ``message`` must already be client-safe."""
        record = self._runs.get(run_id)
        if record is None:
            return
        record.state = RunState.FAILED
        record.error_type = error_type
        record.error_message = message
        record.finished_at = datetime.now(UTC)
        self._release_subscribers(record)

    # --- events -----------------------------------------------------------

    async def publish(self, run_id: str, event: WorkflowEvent) -> None:
        """Buffer an event, then fan it out to live subscribers.

        Buffering happens first and unconditionally: replay must work even when
        nobody is listening, which is what lets a client open the stream after
        a run has already started and still see everything.
        """
        record = self._runs.get(run_id)
        if record is None:
            logger.warning("Event for unknown run discarded", extra={"run_id": run_id})
            return

        record.events.append(event)

        for queue in list(record.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A subscriber that cannot keep up is dropped rather than
                # allowed to stall the run. It can reconnect and replay.
                logger.warning("Dropping slow subscriber", extra={"run_id": run_id})
                record.subscribers.discard(queue)

    async def subscribe(self, run_id: str, *, cursor: int = -1) -> AsyncIterator[WorkflowEvent]:
        """Yield buffered events after ``cursor``, then live ones until the run ends.

        ``cursor`` is the last sequence the client already holds; -1 means send
        everything. The queue is attached *before* the buffer is replayed, so
        an event published during the handover lands in the queue rather than
        vanishing between the two. Sequence numbers then suppress the overlap.
        """
        record = self._runs.get(run_id)
        if record is None:
            return

        queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        already_finished = record.state.is_finished
        if not already_finished:
            record.subscribers.add(queue)

        try:
            delivered = cursor
            for event in list(record.events):
                if event.sequence > delivered:
                    delivered = event.sequence
                    yield event

            if already_finished:
                return

            while True:
                item = await queue.get()
                if item is _CLOSE_SENTINEL:
                    return
                if item.sequence <= delivered:
                    continue  # already sent during replay
                delivered = item.sequence
                yield item
        finally:
            record.subscribers.discard(queue)

    def _release_subscribers(self, record: RunRecord) -> None:
        """Wake every subscriber so its stream can close cleanly."""
        for queue in list(record.subscribers):
            try:
                queue.put_nowait(_CLOSE_SENTINEL)
            except asyncio.QueueFull:
                record.subscribers.discard(queue)

    # --- retention --------------------------------------------------------

    def evict_expired(self) -> int:
        """Drop finished runs older than the TTL. Returns how many were removed."""
        expired = [
            run_id
            for run_id, record in self._runs.items()
            if record.state.is_finished and record.age > self._ttl
        ]
        for run_id in expired:
            del self._runs[run_id]
        if expired:
            logger.info("Evicted expired runs", extra={"count": len(expired)})
        return len(expired)

    def _enforce_max_runs(self) -> None:
        """Drop the oldest finished runs until the count fits."""
        finished = sorted(
            (record for record in self._runs.values() if record.state.is_finished),
            key=lambda record: record.finished_at or record.created_at,
        )
        while len(self._runs) > self._max_runs and finished:
            oldest = finished.pop(0)
            del self._runs[oldest.run_id]
            logger.info("Evicted run over count limit", extra={"run_id": oldest.run_id})

    # --- introspection ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._runs)

    @property
    def run_ids(self) -> list[str]:
        return list(self._runs)


_registry: RunRegistry | None = None


def get_run_registry() -> RunRegistry:
    """Process-wide registry singleton. See the single-worker note above."""
    global _registry
    if _registry is None:
        _registry = RunRegistry()
    return _registry


def reset_run_registry() -> None:
    """Drop the singleton. For tests and shutdown."""
    global _registry
    _registry = None
