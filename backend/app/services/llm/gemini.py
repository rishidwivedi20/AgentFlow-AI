"""Google Gemini adapter.

The only module in the project that imports ``google.genai``. Everything above
it speaks :class:`~app.services.llm.base.LLMProvider`, so replacing or adding a
provider never reaches into an agent.
"""

from __future__ import annotations

import logging
from typing import Final

from google import genai
from google.genai import _transformers as genai_transformers
from google.genai import errors, types
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

PROVIDER_NAME: Final = "gemini"

# Finish reasons that explain an absent parsed result in terms a caller can act
# on. Anything else falls through to a generic message.
_FINISH_REASON_HINTS: Final[dict[types.FinishReason, str]] = {
    types.FinishReason.MAX_TOKENS: (
        "the response was truncated before the JSON was complete; "
        "reduce the requested number of posts or raise max_output_tokens"
    ),
    types.FinishReason.SAFETY: "the response was blocked by safety filters",
    types.FinishReason.PROHIBITED_CONTENT: "the response was blocked as prohibited content",
    types.FinishReason.RECITATION: "the response was blocked as recitation",
}


def _to_provider_schema(schema: type[SchemaT]) -> types.Schema:
    """Convert a Pydantic model into a schema the Gemini API will accept.

    Our LLM-facing models set ``extra="forbid"`` so that a drifting response
    fails validation loudly. Pydantic renders that as ``additionalProperties:
    false``, which the SDK forwards as ``additional_properties`` - a field the
    generateContent endpoint rejects outright with "Cannot find field".

    Stripping it here keeps provider quirks inside the adapter. Strictness is
    not lost: the constraint still applies when the response is validated
    against the Pydantic model on the way back.
    """
    converted = genai_transformers.t_schema(None, schema)
    _strip_additional_properties(converted)
    return converted


def _strip_additional_properties(schema: types.Schema) -> None:
    """Recursively clear ``additional_properties`` in place."""
    schema.additional_properties = None
    for prop in (schema.properties or {}).values():
        _strip_additional_properties(prop)
    if schema.items is not None:
        _strip_additional_properties(schema.items)


def _parse_response(
    response: types.GenerateContentResponse, schema: type[SchemaT]
) -> SchemaT | None:
    """Validate the response body against ``schema``, or None if it cannot be.

    ``response.parsed`` only returns a model instance when the Pydantic class
    itself was passed as the response schema. We pass an explicit
    ``types.Schema`` (see :func:`_to_provider_schema`), so validation happens
    here - which is also where ``extra="forbid"`` is enforced.
    """
    parsed = response.parsed
    if isinstance(parsed, schema):
        return parsed

    text = response.text
    if not text:
        return None
    try:
        return schema.model_validate_json(text)
    except ValidationError:
        logger.warning(
            "Gemini response failed schema validation",
            extra={"schema": schema.__name__},
        )
        return None


def _extract_usage(response: types.GenerateContentResponse) -> TokenUsage | None:
    """Map provider usage onto :class:`TokenUsage`, or None if unreported.

    Returns None rather than zeros when the provider says nothing, so that
    "not reported" is never silently recorded as "cost nothing".
    """
    meta = response.usage_metadata
    if meta is None:
        return None
    return TokenUsage(
        prompt_tokens=meta.prompt_token_count or 0,
        output_tokens=meta.candidates_token_count or 0,
        # Gemini 2.5+ bills internal reasoning separately; excluding it would
        # understate real cost in the Phase 6 comparison.
        thought_tokens=meta.thoughts_token_count or 0,
        total_tokens=meta.total_token_count or 0,
    )


def _describe_empty_response(response: types.GenerateContentResponse) -> str:
    """Explain why a response yielded no parsed object."""
    candidates = response.candidates or []
    if not candidates:
        feedback = response.prompt_feedback
        if feedback is not None and feedback.block_reason is not None:
            return f"the prompt was blocked ({feedback.block_reason.name})"
        return "the provider returned no candidates"

    reason = candidates[0].finish_reason
    if reason is None:
        return "the provider returned no parsable content"
    hint = _FINISH_REASON_HINTS.get(reason)
    return hint if hint else f"generation stopped with finish_reason={reason.name}"


_AUTH_STATUS_CODES: Final = frozenset({401, 403})
_RATE_LIMIT_STATUS_CODES: Final = frozenset({429})


def _map_api_error(exc: errors.APIError) -> AgentFlowError:
    """Translate a provider API error into a meaningful domain category.

    Only ``exc.code`` and ``exc.message`` are used. ``exc.details`` holds the
    raw provider response body and is deliberately never propagated - it can
    carry request echoes and provider internals that must not reach a client.
    """
    code = exc.code

    if code in _RATE_LIMIT_STATUS_CODES:
        return LLMRateLimitError(
            "Gemini rate limit or quota exceeded. Wait before retrying, or reduce "
            "the number of posts per run.",
            details={"provider": PROVIDER_NAME, "status": code},
        )

    if code in _AUTH_STATUS_CODES:
        # A rejected key is a configuration fault, not a transient provider
        # fault. Our own wording is used rather than the provider's, which is
        # the message most likely to echo request material.
        return ConfigurationError(
            "Gemini rejected the API key. Check GEMINI_API_KEY in backend/.env.",
            details={"provider": PROVIDER_NAME, "status": code},
        )

    if code is not None and code >= 500:
        return LLMProviderError(
            f"Gemini is temporarily unavailable: {exc.message}",
            details={RETRYABLE_DETAIL_KEY: True, "provider": PROVIDER_NAME, "status": code},
        )

    return LLMProviderError(
        f"Gemini rejected the request: {exc.message}",
        details={"provider": PROVIDER_NAME, "status": code},
    )


class GeminiProvider:
    """Structured-output adapter over the Google Gen AI SDK.

    Uses the stable ``generateContent`` API with a Pydantic ``response_schema``
    rather than the newer Interactions API, which Google still marks Beta and
    advises against for production.
    """

    def __init__(self, *, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

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
        """Call Gemini and return a validated ``schema`` instance."""
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=_to_provider_schema(schema),
            temperature=temperature,
            # No agent uses tools. Left enabled, the SDK logs an automatic
            # function calling warning on every call.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=config,
            )
        except errors.APIError as exc:
            # ServerError and ClientError both subclass APIError; the status
            # code decides the category. Only 5xx is marked retryable.
            raise _map_api_error(exc) from exc

        parsed = _parse_response(response, schema)
        if parsed is None:
            # No usable object: refusal, truncation, safety block or drift.
            # Retryable - a second sample often succeeds.
            reason = _describe_empty_response(response)
            logger.warning(
                "Gemini returned no valid structured output",
                extra={"schema": schema.__name__, "reason": reason, "model": self._model},
            )
            raise InvalidAgentOutputError(
                f"Gemini did not return valid {schema.__name__} output: {reason}.",
                details={RETRYABLE_DETAIL_KEY: True, "schema": schema.__name__},
            )

        return LLMResult(value=parsed, usage=_extract_usage(response), model=self._model)
