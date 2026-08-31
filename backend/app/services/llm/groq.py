"""Groq adapter.

The only module that imports ``groq``. Everything above it speaks
:class:`~app.services.llm.base.LLMProvider`, so adding this provider required
no change to any agent, prompt or graph node.

Chosen as a second provider because its free tier allows far more requests per
day than Gemini's, which is what makes repeated end-to-end runs practical.

Structured output
-----------------
Groq implements OpenAI-style ``response_format`` with a JSON schema. Its strict
mode is the exact inverse of Gemini's requirements: Gemini *rejects*
``additionalProperties`` while Groq strict mode *requires* it on every object.
Our ``extra="forbid"`` models already emit it, so the schema passes through
here rather than being stripped as it is for Gemini.

Strict mode does reject several validation keywords, which are removed below -
see :func:`_to_provider_schema`.
"""

from __future__ import annotations

import logging
from typing import Any, Final

from groq import APIConnectionError, APIStatusError, APITimeoutError, AsyncGroq
from pydantic import ValidationError

from app.core.exceptions import (
    AgentFlowError,
    ConfigurationError,
    InvalidAgentOutputError,
    LLMProviderError,
    LLMRateLimitError,
)
from app.models.metrics import TokenUsage
from app.services.llm.base import RETRYABLE_DETAIL_KEY, LLMResult, SchemaT

logger = logging.getLogger(__name__)

PROVIDER_NAME: Final = "groq"

# Strict mode accepts a subset of JSON Schema and errors on the rest. These are
# validation constraints, not structure: dropping them changes what the model is
# told, never what we accept. The Pydantic model still enforces every one of
# them when the response is parsed, so nothing is actually relaxed.
_UNSUPPORTED_KEYWORDS: Final = frozenset(
    {
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "default",
    }
)

_AUTH_STATUS_CODES: Final = frozenset({401, 403})

REASONING_EFFORT: Final = "low"
"""Every agent here follows an explicit rubric against a fixed schema, so open
reasoning buys little. Measured on the Researcher's real prompt: the default
effort spent 4,337 reasoning tokens for the same seven extracted facts that
"low" produced with 19 - and the difference is what fits a run inside the free
tier's per-minute budget."""


def _strip_unsupported(node: Any) -> Any:
    """Recursively drop schema keywords strict mode will not accept."""
    if isinstance(node, dict):
        return {
            key: _strip_unsupported(value)
            for key, value in node.items()
            if key not in _UNSUPPORTED_KEYWORDS
        }
    if isinstance(node, list):
        return [_strip_unsupported(item) for item in node]
    return node


def _to_provider_schema(schema: type[SchemaT]) -> dict[str, Any]:
    """Convert a Pydantic model into a schema Groq strict mode accepts.

    ``additionalProperties: false`` and a complete ``required`` list are both
    strict-mode requirements, and both fall out of our LLM-facing model rules -
    ``extra="forbid"``, no optional fields, no defaults - so neither is added
    here. Only the unsupported validation keywords are removed.
    """
    return _strip_unsupported(schema.model_json_schema())


def _extract_usage(response: Any) -> TokenUsage | None:
    """Map provider usage onto :class:`TokenUsage`, or None if unreported.

    Returns None rather than zeros when nothing is reported, so "not reported"
    is never recorded as "cost nothing".
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    # Reasoning models report thinking separately, in a nested field that not
    # every model populates.
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) or 0

    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        thought_tokens=reasoning,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


def _map_api_error(exc: APIStatusError) -> AgentFlowError:
    """Translate a provider API error into a meaningful domain category.

    Only the status code and the SDK's own message are used. The raw response
    body is never propagated - it can carry request echoes and provider
    internals that must not reach a client.
    """
    code = exc.status_code

    if code == 413:
        return LLMProviderError(
            "The request exceeded Groq's per-minute token budget. Reduce "
            "TAVILY_MAX_RESULTS or GROQ_MAX_COMPLETION_TOKENS.",
            details={"provider": PROVIDER_NAME, "status": code},
        )

    if code == 429:
        return LLMRateLimitError(
            "Groq rate limit reached. The free tier allows 30 requests per minute "
            "and 1,000 per day.",
            details={"provider": PROVIDER_NAME, "status": code},
        )

    if code in _AUTH_STATUS_CODES:
        # A rejected key is a configuration fault, not a transient one. Our own
        # wording replaces the provider's, which is the message most likely to
        # echo request material.
        return ConfigurationError(
            "Groq rejected the API key. Check GROQ_API_KEY in backend/.env.",
            details={"provider": PROVIDER_NAME, "status": code},
        )

    if code >= 500:
        return LLMProviderError(
            "Groq is temporarily unavailable.",
            details={RETRYABLE_DETAIL_KEY: True, "provider": PROVIDER_NAME, "status": code},
        )

    return LLMProviderError(
        f"Groq rejected the request: {exc.message}",
        details={"provider": PROVIDER_NAME, "status": code},
    )


def _parse_response(response: Any, schema: type[SchemaT]) -> SchemaT | None:
    """Validate the completion against ``schema``, or None if it cannot be."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None

    message = choices[0].message
    # OpenAI-compatible APIs surface a deliberate refusal separately from a
    # failed parse; it is not retryable in a useful way, but it is worth
    # reporting distinctly in the log.
    if getattr(message, "refusal", None):
        logger.warning(
            "Groq refused the request",
            extra={"schema": schema.__name__, "reason": str(message.refusal)[:200]},
        )
        return None

    content = getattr(message, "content", None)
    if not content:
        return None

    try:
        return schema.model_validate_json(content)
    except ValidationError:
        logger.warning(
            "Groq response failed schema validation", extra={"schema": schema.__name__}
        )
        return None


class GroqProvider:
    """Structured-output adapter over the Groq API."""

    def __init__(self, *, api_key: str, model: str, max_completion_tokens: int) -> None:
        # max_retries=0: LLMService owns the retry policy. Leaving the SDK's
        # own retries enabled would nest one bounded loop inside another and
        # multiply the worst-case request count.
        self._client = AsyncGroq(api_key=api_key, max_retries=0)
        self._model = model
        self._max_completion_tokens = max_completion_tokens

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def model(self) -> str:
        return self._model

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        temperature: float,
    ) -> LLMResult[SchemaT]:
        """Call Groq and return a validated ``schema`` instance."""
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": _to_provider_schema(schema),
                    },
                },
                temperature=temperature,
                max_completion_tokens=self._max_completion_tokens,
                reasoning_effort=REASONING_EFFORT,
            )
        except APITimeoutError as exc:
            raise LLMProviderError(
                "Groq did not respond in time.",
                details={RETRYABLE_DETAIL_KEY: True, "provider": PROVIDER_NAME},
            ) from exc
        except APIStatusError as exc:
            raise _map_api_error(exc) from exc
        except APIConnectionError as exc:
            raise LLMProviderError(
                "Could not reach Groq.",
                details={RETRYABLE_DETAIL_KEY: True, "provider": PROVIDER_NAME},
            ) from exc

        parsed = _parse_response(response, schema)
        if parsed is None:
            # Retryable: a second sample often succeeds where one refusal or a
            # truncated body did not.
            raise InvalidAgentOutputError(
                f"Groq did not return valid {schema.__name__} output.",
                details={RETRYABLE_DETAIL_KEY: True, "schema": schema.__name__},
            )

        return LLMResult(value=parsed, usage=_extract_usage(response), model=self._model)
