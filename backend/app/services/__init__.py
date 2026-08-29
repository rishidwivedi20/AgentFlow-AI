"""Service layer: LLM abstraction, metrics, run registry.

Agents depend on these services rather than on provider SDKs, so swapping a
provider touches one adapter instead of every agent.
"""

from app.services.llm_service import LLMService, build_llm_service, build_provider
from app.services.research_service import (
    ResearchService,
    build_research_provider,
    build_research_service,
)

__all__ = [
    "LLMService",
    "ResearchService",
    "build_llm_service",
    "build_provider",
    "build_research_provider",
    "build_research_service",
]
