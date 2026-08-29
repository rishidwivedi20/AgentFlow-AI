"""Application entry point.

Run from the ``backend/`` directory:

    uvicorn app.main:app --reload --port 8000

SINGLE WORKER ONLY. Live workflow runs, their buffered events and their SSE
subscriber queues are held in this process's memory by
``app.services.run_registry``. With ``--workers 2`` a client polling or
streaming a run can be routed to a process that has never heard of it and will
receive a 404 for work that is executing elsewhere.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.routes import api_router
from app.services.run_registry import reset_run_registry

logger = logging.getLogger(__name__)


def _log_startup_state(settings: Settings) -> None:
    """Surface configuration state once, loudly, at boot.

    A missing key must be obvious here - not a KeyError forty seconds into a
    workflow run.
    """
    logger.info(
        "Starting %s v%s",
        settings.app_name,
        settings.app_version,
        extra={
            "environment": settings.app_env,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.active_model,
            "llm_configured": settings.llm_configured,
            "max_revisions": settings.max_revisions,
            "cors_origins": settings.cors_origin_list,
        },
    )

    if not settings.llm_configured:
        message = (
            "No API key set for provider '%s'. This is expected in Phase 1; "
            "agent execution will be unavailable until %s_API_KEY is set in backend/.env."
        )
        logger.warning(message, settings.llm_provider, settings.llm_provider.upper())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate configuration on startup, log a clean shutdown."""
    settings: Settings = app.state.settings
    _log_startup_state(settings)
    yield
    # Drop in-flight runs and their subscriber queues so a reload does not
    # leave orphaned tasks holding memory.
    reset_run_registry()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.is_production)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Multi-agent AI content generation and evaluation platform.",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
