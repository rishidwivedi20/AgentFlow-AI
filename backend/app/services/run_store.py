"""Durable run history in SQLite.

The in-memory :mod:`app.services.run_registry` stays the live path: it owns
running tasks and fans out events. This store only records runs once they
finish, which is all that history and cross-run metrics need - and it keeps a
finished run readable, events included, after the registry has evicted it or
the process has restarted.

SQLite from the standard library, not a server. One file, no dependency, and
the volume is a handful of runs a day. Every method is synchronous and opens
its own connection; callers on the event loop go through ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Final

from pydantic import BaseModel, Field, TypeAdapter

from app.core.config import Settings
from app.models.enums import AgentName, Platform, Tone
from app.models.events import WorkflowEvent
from app.schemas.workflow import WorkflowRunResponse

logger = logging.getLogger(__name__)

SCHEMA_VERSION: Final = 1

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    finished_at      TEXT NOT NULL,
    topic            TEXT NOT NULL,
    platform         TEXT NOT NULL,
    tone             TEXT NOT NULL,
    number_of_posts  INTEGER NOT NULL,
    state            TEXT NOT NULL,
    workflow_status  TEXT,
    revision_count   INTEGER,
    approved         INTEGER,
    quality_score    REAL,
    total_tokens     INTEGER,
    duration_seconds REAL NOT NULL,
    cost_usd         REAL,
    provider         TEXT NOT NULL,
    model            TEXT NOT NULL,
    error_type       TEXT,
    error_message    TEXT,
    result_json      TEXT,
    events_json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_created_at ON runs (created_at DESC);

CREATE TABLE IF NOT EXISTS agent_executions (
    run_id          TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    sequence        INTEGER NOT NULL,
    agent           TEXT NOT NULL,
    status          TEXT NOT NULL,
    latency_seconds REAL NOT NULL,
    total_tokens    INTEGER,
    PRIMARY KEY (run_id, sequence)
);
"""

_EVENTS = TypeAdapter(list[WorkflowEvent])


# --- read models ------------------------------------------------------------


class RunSummary(BaseModel):
    """One row of run history: enough to list and filter, not the full result."""

    run_id: str
    topic: str
    platform: Platform
    tone: Tone
    number_of_posts: int
    state: str = Field(description="Run lifecycle state: completed or failed.")
    workflow_status: str | None = Field(
        default=None, description="The workflow's own outcome. None when the run crashed."
    )
    created_at: datetime
    finished_at: datetime
    duration_seconds: float
    revision_count: int | None = None
    approved: bool | None = None
    quality_score: float | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    provider: str
    model: str
    error_type: str | None = None


class StoredRun(BaseModel):
    """A finished run as persisted: summary, full result and its events."""

    summary: RunSummary
    result: WorkflowRunResponse | None
    error_message: str | None
    events: list[WorkflowEvent]


class AgentAggregate(BaseModel):
    """One agent's behaviour across every stored run."""

    agent: AgentName
    executions: int
    failures: int
    failure_rate: float
    mean_latency_seconds: float
    mean_tokens: float | None = Field(
        default=None, description="Over executions that reported usage. None for non-LLM nodes."
    )


class MetricsSummary(BaseModel):
    """Aggregates over stored run history.

    Averages over outcomes - quality, revisions, approval - only count runs
    whose workflow actually produced a verdict. A run that crashed has no
    quality score, and averaging it in as zero would describe the crash rate,
    not the writing.
    """

    total_runs: int
    runs_by_state: dict[str, int]
    runs_by_workflow_status: dict[str, int]
    approval_rate: float | None
    mean_revisions: float | None
    mean_quality: float | None
    mean_duration_seconds: float | None
    mean_tokens: float | None
    total_cost_usd: float | None
    mean_cost_usd: float | None
    first_run_at: datetime | None
    last_run_at: datetime | None
    agents: list[AgentAggregate]


# --- the store ----------------------------------------------------------------


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


class RunStore:
    """SQLite-backed history of finished runs."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @property
    def path(self) -> Path:
        return self._path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection that commits on success and always closes.

        ``sqlite3.Connection`` as a context manager commits but does not close,
        which on Windows leaves the file locked.
        """
        with closing(sqlite3.connect(self._path, timeout=10)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            with conn:
                yield conn

    # --- writes -------------------------------------------------------------

    def save(
        self,
        *,
        run_id: str,
        topic: str,
        platform: Platform,
        tone: Tone,
        number_of_posts: int,
        state: str,
        created_at: datetime,
        finished_at: datetime,
        provider: str,
        model: str,
        result: WorkflowRunResponse | None,
        events: list[WorkflowEvent],
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Insert or replace one finished run and its per-agent executions."""
        metrics = result.agent_metrics if result else []
        tokens = [m.usage.total_tokens for m in metrics if m.usage is not None]
        row = {
            "run_id": run_id,
            "created_at": created_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "topic": topic,
            "platform": platform.value,
            "tone": tone.value,
            "number_of_posts": number_of_posts,
            "state": state,
            "workflow_status": result.workflow_status.value if result else None,
            "revision_count": result.revision_count if result else None,
            "approved": int(result.approved) if result else None,
            "quality_score": result.quality_score if result else None,
            "total_tokens": sum(tokens) if tokens else None,
            "duration_seconds": round((finished_at - created_at).total_seconds(), 4),
            "cost_usd": result.cost.total_usd if result and result.cost else None,
            "provider": provider,
            "model": model,
            "error_type": error_type,
            "error_message": error_message,
            # Computed fields are left out: they are derived on the way back
            # out, and models that forbid extra keys would reject them.
            "result_json": (
                result.model_dump_json(exclude_computed_fields=True) if result else None
            ),
            "events_json": _EVENTS.dump_json(events).decode(),
        }
        columns = ", ".join(row)
        placeholders = ", ".join(f":{key}" for key in row)
        with self._connect() as conn:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.execute(f"INSERT INTO runs ({columns}) VALUES ({placeholders})", row)
            conn.executemany(
                "INSERT INTO agent_executions "
                "(run_id, sequence, agent, status, latency_seconds, total_tokens) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        m.sequence,
                        m.agent.value,
                        m.status.value,
                        m.latency_seconds,
                        m.usage.total_tokens if m.usage else None,
                    )
                    for m in metrics
                ],
            )

    # --- reads --------------------------------------------------------------

    @staticmethod
    def _summary(row: sqlite3.Row) -> RunSummary:
        # A Row's own `in` tests values, not column names.
        columns = set(row.keys())
        values = {key: row[key] for key in RunSummary.model_fields if key in columns}
        if values["approved"] is not None:
            values["approved"] = bool(values["approved"])
        return RunSummary.model_validate(values)

    def get(self, run_id: str) -> StoredRun | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return StoredRun(
            summary=self._summary(row),
            result=(
                WorkflowRunResponse.model_validate_json(row["result_json"])
                if row["result_json"]
                else None
            ),
            error_message=row["error_message"],
            events=_EVENTS.validate_json(row["events_json"]),
        )

    def list(self, *, limit: int = 50, offset: int = 0) -> list[RunSummary]:
        """Most recent first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._summary(row) for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])

    def summary(self) -> MetricsSummary:
        with self._connect() as conn:
            totals = conn.execute(
                """
                SELECT
                    COUNT(*)                                   AS total_runs,
                    AVG(approved)                              AS approval_rate,
                    AVG(revision_count)                        AS mean_revisions,
                    AVG(quality_score)                         AS mean_quality,
                    AVG(duration_seconds)                      AS mean_duration,
                    AVG(total_tokens)                          AS mean_tokens,
                    SUM(cost_usd)                              AS total_cost,
                    AVG(cost_usd)                              AS mean_cost,
                    MIN(created_at)                            AS first_run_at,
                    MAX(created_at)                            AS last_run_at
                FROM runs
                """
            ).fetchone()
            by_state = conn.execute(
                "SELECT state, COUNT(*) AS n FROM runs GROUP BY state"
            ).fetchall()
            by_status = conn.execute(
                "SELECT workflow_status, COUNT(*) AS n FROM runs "
                "WHERE workflow_status IS NOT NULL GROUP BY workflow_status"
            ).fetchall()
            agents = conn.execute(
                """
                SELECT
                    agent,
                    COUNT(*)                                        AS executions,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failures,
                    AVG(latency_seconds)                            AS mean_latency,
                    AVG(total_tokens)                               AS mean_tokens
                FROM agent_executions
                GROUP BY agent
                """
            ).fetchall()

        order = {agent.value: index for index, agent in enumerate(AgentName)}
        return MetricsSummary(
            total_runs=totals["total_runs"],
            runs_by_state={row["state"]: row["n"] for row in by_state},
            runs_by_workflow_status={row["workflow_status"]: row["n"] for row in by_status},
            approval_rate=_round(totals["approval_rate"]),
            mean_revisions=_round(totals["mean_revisions"]),
            mean_quality=_round(totals["mean_quality"]),
            mean_duration_seconds=_round(totals["mean_duration"]),
            mean_tokens=_round(totals["mean_tokens"], 1),
            total_cost_usd=_round(totals["total_cost"], 6),
            mean_cost_usd=_round(totals["mean_cost"], 6),
            first_run_at=totals["first_run_at"],
            last_run_at=totals["last_run_at"],
            agents=sorted(
                (
                    AgentAggregate(
                        agent=row["agent"],
                        executions=row["executions"],
                        failures=row["failures"],
                        failure_rate=round(row["failures"] / row["executions"], 4),
                        mean_latency_seconds=round(row["mean_latency"], 4),
                        mean_tokens=_round(row["mean_tokens"], 1),
                    )
                    for row in agents
                ),
                key=lambda aggregate: order.get(aggregate.agent.value, len(order)),
            ),
        )


# --- process-wide instance -----------------------------------------------------

_stores: dict[Path, RunStore] = {}
_lock = Lock()


def get_run_store(settings: Settings) -> RunStore | None:
    """The store for the configured path, or None when history is switched off.

    Never raises. History is an addition to a run, not a prerequisite: an
    unwritable path is logged and the API carries on without it.
    """
    if not settings.run_store_enabled:
        return None
    path = Path(settings.run_store_path).resolve()
    with _lock:
        store = _stores.get(path)
        if store is None:
            try:
                store = RunStore(path)
            except (OSError, sqlite3.Error):
                logger.exception("Run history unavailable", extra={"path": str(path)})
                return None
            _stores[path] = store
        return store


def reset_run_stores() -> None:
    """Forget cached stores. For tests."""
    with _lock:
        _stores.clear()
