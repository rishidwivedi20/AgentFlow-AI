"""Cross-run metrics computed from run history."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.services.pricing import ModelPricing, resolve_pricing
from app.services.run_store import MetricsSummary, get_run_store

router = APIRouter(tags=["metrics"])

SettingsDep = Annotated[Settings, Depends(get_settings)]


class MetricsSummaryResponse(BaseModel):
    """Aggregates over every stored run, plus the rates costs are estimated at."""

    history_enabled: bool
    summary: MetricsSummary | None = Field(
        default=None, description="None when run history is switched off."
    )
    pricing: ModelPricing | None = Field(
        default=None,
        description="Rates for the currently configured model. None when it is unpriced.",
    )


@router.get("/summary", response_model=MetricsSummaryResponse, summary="Metrics across runs")
async def metrics_summary(settings: SettingsDep) -> MetricsSummaryResponse:
    """Run outcomes, cost and per-agent behaviour aggregated over history."""
    store = get_run_store(settings)
    pricing = resolve_pricing(settings)
    if store is None:
        return MetricsSummaryResponse(history_enabled=False, pricing=pricing)
    summary = await asyncio.to_thread(store.summary)
    return MetricsSummaryResponse(history_enabled=True, summary=summary, pricing=pricing)
