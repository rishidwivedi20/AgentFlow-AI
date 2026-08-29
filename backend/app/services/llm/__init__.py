"""Provider abstraction and adapters.

``base`` defines the contract, each other module is one vendor adapter. Import
the contract from here; import an adapter only from the service factory.
"""

from app.services.llm.base import RETRYABLE_DETAIL_KEY, LLMProvider, LLMResult, SchemaT

__all__ = ["RETRYABLE_DETAIL_KEY", "LLMProvider", "LLMResult", "SchemaT"]
