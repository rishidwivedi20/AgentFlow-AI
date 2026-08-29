"""API router aggregation.

Every route module is mounted here; ``main`` includes only ``api_router``.
"""

from fastapi import APIRouter

from app.routes.health import router as health_router
from app.routes.workflows import router as workflows_router

api_router = APIRouter(prefix="/api")
api_router.include_router(health_router)
api_router.include_router(workflows_router, prefix="/workflows")

__all__ = ["api_router"]
