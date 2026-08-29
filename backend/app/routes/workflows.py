"""Workflow execution endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.graph.workflow import run_workflow
from app.schemas.workflow import (
    WorkflowRunRequest,
    WorkflowRunResponse,
    build_workflow_response,
)
from app.services.llm_service import build_llm_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["workflows"])

SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.post(
    "/run",
    response_model=WorkflowRunResponse,
    summary="Run the Writer/Editor workflow",
)
async def run(request: WorkflowRunRequest, settings: SettingsDep) -> WorkflowRunResponse:
    """Execute one workflow run.

    Returns 200 with a failure ``workflow_status`` when the run executed but
    did not succeed. A missing API key raises ``ConfigurationError`` (503)
    before any work starts - there is no point spending a request to discover
    configuration is absent.
    """
    llm = build_llm_service(settings)

    state = await run_workflow(
        llm,
        topic=request.topic,
        platform=request.platform,
        tone=request.tone,
        number_of_posts=request.number_of_posts,
        settings=settings,
    )
    return build_workflow_response(state)
