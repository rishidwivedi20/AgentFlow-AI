"""Domain exceptions and the HTTP error envelope.

Every error the API returns - validation, domain, or unexpected - uses one
shape, so the frontend has exactly one error branch to write:

    {"error": {"type": "...", "message": "...", "details": {...}}}
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class OrchestrAIError(Exception):
    """Base class for every error this application raises deliberately.

    Carries the HTTP status and a machine-readable ``error_type`` so handlers
    never need to inspect the concrete class.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(OrchestrAIError):
    """Required configuration is missing or invalid (e.g. absent API key)."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_type = "configuration_error"


class LLMProviderError(OrchestrAIError):
    """The upstream LLM provider failed, refused, or returned an error."""

    status_code = status.HTTP_502_BAD_GATEWAY
    error_type = "llm_provider_error"


class LLMTimeoutError(OrchestrAIError):
    """The LLM provider did not respond within the configured timeout."""

    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    error_type = "llm_timeout"


class InvalidAgentOutputError(OrchestrAIError):
    """An agent returned output that failed schema validation or JSON parsing.

    Raised after retries are exhausted - a single malformed response is
    normally retried by the LLM service rather than surfaced.
    """

    status_code = status.HTTP_502_BAD_GATEWAY
    error_type = "invalid_agent_output"


class WorkflowError(OrchestrAIError):
    """The orchestration graph could not complete."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = "workflow_error"


class MaxRevisionsExceededError(WorkflowError):
    """The Writer/Editor loop hit its hard revision cap.

    Routing code normally terminates the loop gracefully; this exists as a
    safety net so a runaway loop fails loudly instead of burning tokens.
    """

    error_type = "max_revisions_exceeded"


def _envelope(error_type: str, message: str, details: dict[str, Any] | None = None) -> dict:
    return {"error": {"type": error_type, "message": message, "details": details or {}}}


async def orchestrai_error_handler(_: Request, exc: OrchestrAIError) -> JSONResponse:
    logger.warning(
        "Handled domain error",
        extra={"error_type": exc.error_type, "detail_keys": sorted(exc.details)},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(exc.error_type, exc.message, exc.details),
    )


async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Reshape FastAPI's 422 body into the standard envelope."""
    fields = [
        {
            "field": ".".join(str(part) for part in err["loc"][1:]) or str(err["loc"][0]),
            "message": err["msg"],
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_envelope("validation_error", "Request validation failed", {"fields": fields}),
    )


async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Keep framework-raised 404/405 responses in the same envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope("http_error", str(exc.detail)),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort. Logs the full traceback, returns no internals to the client."""
    logger.exception(
        "Unhandled exception",
        extra={"path": request.url.path, "method": request.method},
        exc_info=exc,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_envelope("internal_error", "An unexpected internal error occurred."),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire every handler onto the application."""
    app.add_exception_handler(OrchestrAIError, orchestrai_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
