"""Asynchronous run endpoints with Server-Sent Events.

    POST /api/workflows              queue a run, return its id immediately
    GET  /api/workflows/{id}         current state and, once finished, the result
    GET  /api/workflows/{id}/events  SSE stream of execution events

The existing synchronous ``POST /api/workflows/run`` is untouched. Streaming is
purely additive: a client that does not want it never encounters this module.

SINGLE WORKER ONLY - runs live in this process's memory. See
``app.services.run_registry``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.core.config import Settings, get_settings
from app.core.exceptions import OrchestrAIError
from app.graph.workflow import run_workflow
from app.models.events import WorkflowEvent
from app.schemas.run import RunAcceptedResponse, RunStatusResponse
from app.schemas.workflow import WorkflowRunRequest, build_workflow_response
from app.services.event_bus import RunEventEmitter
from app.services.llm_service import build_llm_service
from app.services.research_service import build_research_service
from app.services.run_registry import RunRecord, RunRegistry, get_run_registry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["runs"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
RegistryDep = Annotated[RunRegistry, Depends(get_run_registry)]

KEEPALIVE_SECONDS: Final = 15.0
"""How often to send an SSE comment when no event has occurred. Proxies and
browsers close idle connections; a run's Researcher step alone can exceed 45s."""


def _require_run(registry: RunRegistry, run_id: str) -> RunRecord:
    """Fetch a run or raise 404.

    A 404 here can mean 'never existed' or 'evicted after an hour'. The message
    says so, because the two are indistinguishable to the client and the second
    is the likelier one.
    """
    record = registry.get(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' was not found. Runs are retained for a limited time.",
        )
    return record


async def _execute(run_id: str, request: WorkflowRunRequest, settings: Settings) -> None:
    """Run the workflow in the background, recording the outcome.

    Never raises: this is a detached task, so an escaping exception would be
    logged by asyncio and lost to the client. Every failure becomes run state
    instead.
    """
    registry = get_run_registry()
    registry.mark_running(run_id)
    emitter = RunEventEmitter(run_id, registry)

    try:
        state = await run_workflow(
            build_llm_service(settings),
            build_research_service(settings),
            topic=request.topic,
            platform=request.platform,
            tone=request.tone,
            number_of_posts=request.number_of_posts,
            settings=settings,
            emitter=emitter,
        )
        registry.complete(run_id, build_workflow_response(state))
    except OrchestrAIError as exc:
        # Domain failures carry client-safe messages already.
        logger.warning("Run failed", extra={"run_id": run_id, "error_type": exc.error_type})
        registry.fail(run_id, error_type=exc.error_type, message=exc.message)
    except Exception:
        # Log the traceback; tell the client nothing about our internals.
        logger.exception("Run failed unexpectedly", extra={"run_id": run_id})
        registry.fail(
            run_id,
            error_type="internal_error",
            message="The run failed unexpectedly.",
        )


@router.post(
    "",
    response_model=RunAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a workflow run",
)
async def create_run(
    request: WorkflowRunRequest,
    settings: SettingsDep,
    registry: RegistryDep,
) -> RunAcceptedResponse:
    """Start a run and return immediately with its id.

    Configuration is validated here, before the run is created, so a missing
    key is a 503 on this request rather than a run that exists only to fail.
    """
    build_llm_service(settings)
    build_research_service(settings)

    record = registry.create(topic=request.topic)
    record.task = asyncio.create_task(_execute(record.run_id, request, settings))

    logger.info("Run queued", extra={"run_id": record.run_id, "topic": request.topic[:80]})
    return RunAcceptedResponse(
        run_id=record.run_id,
        state=record.state,
        topic=record.topic,
        created_at=record.created_at,
        events_url=f"/api/workflows/{record.run_id}/events",
        status_url=f"/api/workflows/{record.run_id}",
    )


@router.get("/{run_id}", response_model=RunStatusResponse, summary="Run state and result")
async def get_run(run_id: str, registry: RegistryDep) -> RunStatusResponse:
    """Return a run's current state, including its result once finished."""
    record = _require_run(registry, run_id)
    return RunStatusResponse(
        run_id=record.run_id,
        state=record.state,
        topic=record.topic,
        created_at=record.created_at,
        finished_at=record.finished_at,
        event_count=len(record.events),
        result=record.result,
        error_type=record.error_type,
        error_message=record.error_message,
    )


def _format_sse(event: WorkflowEvent) -> str:
    """Render one event in the SSE wire format.

    ``id:`` carries the sequence so a browser echoes it back as
    ``Last-Event-ID`` when EventSource reconnects, which is what makes replay
    automatic rather than something the client has to implement.
    """
    return (
        f"id: {event.sequence}\n"
        f"event: {event.type.value}\n"
        f"data: {event.model_dump_json()}\n\n"
    )


def _resolve_cursor(last_event_id: str | None, cursor: int | None) -> int:
    """Decide where to resume from.

    ``Last-Event-ID`` wins over the query parameter. On reconnect the browser
    reuses the original URL - including whatever cursor it first carried - so
    trusting the query string would replay from the start every time.
    """
    if last_event_id is not None:
        try:
            return int(last_event_id)
        except ValueError:
            logger.warning("Ignoring malformed Last-Event-ID", extra={"value": last_event_id[:32]})
    return cursor if cursor is not None else -1


async def _event_stream(
    request: Request, registry: RunRegistry, run_id: str, cursor: int
) -> AsyncIterator[str]:
    """Yield SSE frames until the run ends or the client goes away."""
    events = registry.subscribe(run_id, cursor=cursor).__aiter__()

    try:
        while True:
            if await request.is_disconnected():
                logger.debug("Client disconnected from stream", extra={"run_id": run_id})
                return
            try:
                event = await asyncio.wait_for(events.__anext__(), timeout=KEEPALIVE_SECONDS)
            except StopAsyncIteration:
                return
            except TimeoutError:
                # A comment frame: keeps the connection alive without being
                # delivered to the client's message handler.
                yield ": keepalive\n\n"
                continue
            yield _format_sse(event)
    finally:
        await events.aclose()


@router.get("/{run_id}/events", summary="Stream run events (SSE)")
async def stream_events(
    run_id: str,
    request: Request,
    registry: RegistryDep,
    cursor: Annotated[int | None, Query(ge=-1)] = None,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    """Stream this run's events, replaying anything after the cursor first.

    Works whether the run is queued, executing or long finished: buffered
    events are replayed from the cursor, then live ones follow.
    """
    _require_run(registry, run_id)
    resolved = _resolve_cursor(last_event_id, cursor)

    return StreamingResponse(
        _event_stream(request, registry, run_id, resolved),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Stops nginx and similar buffering the stream into uselessness.
            "X-Accel-Buffering": "no",
        },
    )
