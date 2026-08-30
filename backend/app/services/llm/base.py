"""Provider-agnostic contract for structured LLM generation.

Agents depend on this module, never on a vendor SDK. Adding a provider means
writing one adapter that satisfies :class:`LLMProvider`; no agent changes.

The contract is deliberately narrow: one method, structured output only. Free
text generation is not exposed because every agent in this system returns a
validated schema, and a looser API would invite agents to bypass validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from app.models.metrics import TokenUsage

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Marker placed in AgentFlowError.details by adapters when a failure is worth
# retrying (transient server-side faults). Client errors - bad key, quota
# exhausted, malformed request - are never marked: retrying them wastes a
# free-tier request and cannot succeed.
RETRYABLE_DETAIL_KEY = "retryable"


@dataclass(frozen=True, slots=True)
class LLMResult(Generic[SchemaT]):
    """A validated model instance plus the metadata needed for observability."""

    value: SchemaT
    """The parsed, schema-valid response."""

    usage: TokenUsage | None
    """Real provider-reported token counts, or None when the provider
    reported none. Never synthesised - see TokenUsage's docstring."""

    model: str
    """Model that produced the result, recorded per call because the
    configured model can change between runs."""


class LLMProvider(Protocol):
    """What every provider adapter must offer."""

    @property
    def name(self) -> str:
        """Short provider identifier, e.g. 'gemini'."""
        ...

    @property
    def model(self) -> str:
        """Model identifier this adapter is configured to call."""
        ...

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        temperature: float,
    ) -> LLMResult[SchemaT]:
        """Generate a response validated against ``schema``.

        Implementations must raise:

        * ``InvalidAgentOutputError`` when the response cannot be parsed into
          ``schema`` - including refusals, truncation and safety blocks, since
          all three yield no usable object.
        * ``LLMProviderError`` when the provider itself fails. Set
          ``details[RETRYABLE_DETAIL_KEY] = True`` only for transient faults.
        """
        ...
