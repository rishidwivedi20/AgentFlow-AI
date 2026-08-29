"""Response schemas for the health endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HealthStatus = Literal["ok", "degraded"]


class HealthChecks(BaseModel):
    """Individual readiness signals behind the overall status."""

    config_loaded: bool = Field(description="Settings were parsed and validated successfully.")
    api_key_present: bool = Field(
        description="The selected LLM provider has a non-empty API key configured."
    )


class HealthResponse(BaseModel):
    """Honest service health, not a hardcoded 'ok'."""

    status: HealthStatus = Field(description="'ok' when the service can serve its current phase.")
    version: str
    environment: str
    llm_provider: str = Field(description="Provider the LLM service layer resolves to.")
    llm_model: str = Field(description="Model name for the selected provider.")
    llm_configured: bool = Field(
        description=(
            "False is expected before Phase 2: no provider key is required until "
            "the agents exist."
        )
    )
    checks: HealthChecks
