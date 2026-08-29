"""Health and readiness endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.health import HealthChecks, HealthResponse

router = APIRouter(tags=["health"])

SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get("/health", response_model=HealthResponse, summary="Service health")
async def health(settings: SettingsDep) -> HealthResponse:
    """Report real readiness rather than a constant 'ok'.

    A missing provider key is only ``degraded`` in production. In development
    it is the expected Phase 1 state and must not look like a failure.
    """
    api_key_present = settings.llm_configured
    degraded = settings.is_production and not api_key_present

    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=settings.app_version,
        environment=settings.app_env,
        llm_provider=settings.llm_provider,
        llm_model=settings.active_model,
        llm_configured=api_key_present,
        checks=HealthChecks(config_loaded=True, api_key_present=api_key_present),
    )
