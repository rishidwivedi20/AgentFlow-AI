"""Web search abstraction and adapters.

``base`` defines the contract, each other module is one provider adapter.
Import the contract from here; import an adapter only from the service factory.
"""

from app.services.research.base import RETRYABLE_DETAIL_KEY, ResearchProvider

__all__ = ["RETRYABLE_DETAIL_KEY", "ResearchProvider"]
