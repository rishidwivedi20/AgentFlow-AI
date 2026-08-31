"""Groq adapter: schema conversion, error mapping and response parsing.

No network. The provider's own HTTP layer is never exercised here - what is
tested is everything we put around it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from groq import APIStatusError

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, LLMProviderError, LLMRateLimitError
from app.models.content import Post, WriterOutput
from app.models.factcheck import FactCheckClaim, FactCheckReport
from app.models.plan import ContentAngle, ContentPlan
from app.models.research import ResearchFact, ResearchOutput
from app.models.review import EditorIssue, EditorOutput
from app.services.llm.base import RETRYABLE_DETAIL_KEY
from app.services.llm.groq import (
    PROVIDER_NAME,
    GroqProvider,
    _extract_usage,
    _map_api_error,
    _parse_response,
    _to_provider_schema,
)
from app.services.llm_service import build_provider

LLM_FACING_MODELS = [
    WriterOutput,
    EditorOutput,
    Post,
    EditorIssue,
    ResearchOutput,
    ResearchFact,
    ContentPlan,
    ContentAngle,
    FactCheckReport,
    FactCheckClaim,
]

UNSUPPORTED_KEYWORDS = [
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "default",
]


def status_error(code: int, message: str = "boom") -> APIStatusError:
    """Build a real SDK error the way the client does."""
    response = httpx.Response(
        code, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    )
    return APIStatusError(message, response=response, body=None)


def collect_keys(node, found=None):
    """Every dict key appearing anywhere in the schema tree."""
    found = set() if found is None else found
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            collect_keys(value, found)
    elif isinstance(node, list):
        for item in node:
            collect_keys(item, found)
    return found


# --- schema conversion ------------------------------------------------------


@pytest.mark.parametrize("model", LLM_FACING_MODELS, ids=lambda m: m.__name__)
def test_unsupported_keywords_are_stripped(model):
    """Strict mode errors on these; our Pydantic model still enforces them."""
    keys = collect_keys(_to_provider_schema(model))
    offenders = [kw for kw in UNSUPPORTED_KEYWORDS if kw in keys]
    assert offenders == [], f"strict mode would reject: {offenders}"


@pytest.mark.parametrize("model", LLM_FACING_MODELS, ids=lambda m: m.__name__)
def test_additional_properties_is_kept(model):
    """The exact inverse of Gemini: strict mode REQUIRES this to be present."""
    assert "additionalProperties" in collect_keys(_to_provider_schema(model))


@pytest.mark.parametrize("model", LLM_FACING_MODELS, ids=lambda m: m.__name__)
def test_every_field_is_required(model):
    """Strict mode rejects optional properties; our models have none."""
    schema = _to_provider_schema(model)
    assert set(schema.get("required", [])) == set(model.model_fields)


def test_structure_and_defs_survive():
    schema = _to_provider_schema(EditorOutput)
    assert set(schema["properties"]) == set(EditorOutput.model_fields)
    # Enums are referenced, not inlined; strict mode supports $defs/$ref.
    assert "EditorIssue" in schema["$defs"]
    assert "ReviewDimension" in schema["$defs"]


def test_enum_values_are_preserved():
    """Dropping enum members would let the model return an unparsable verdict."""
    schema = _to_provider_schema(EditorOutput)
    assert set(schema["$defs"]["IssueSeverity"]["enum"]) == {"low", "medium", "high"}


def test_conversion_does_not_mutate_the_model():
    """model_json_schema() is cached by Pydantic; stripping must not corrupt it."""
    before = json.dumps(EditorOutput.model_json_schema(), sort_keys=True)
    _to_provider_schema(EditorOutput)
    assert json.dumps(EditorOutput.model_json_schema(), sort_keys=True) == before


# --- error mapping ----------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected", "error_type"),
    [
        (429, LLMRateLimitError, "rate_limited"),
        (401, ConfigurationError, "configuration_error"),
        (403, ConfigurationError, "configuration_error"),
        (500, LLMProviderError, "llm_provider_error"),
        (503, LLMProviderError, "llm_provider_error"),
        (400, LLMProviderError, "llm_provider_error"),
        (404, LLMProviderError, "llm_provider_error"),
    ],
)
def test_status_codes_map_to_meaningful_categories(code, expected, error_type):
    mapped = _map_api_error(status_error(code))
    assert isinstance(mapped, expected)
    assert mapped.error_type == error_type
    assert mapped.details["provider"] == PROVIDER_NAME


def test_only_server_errors_are_retryable():
    """Retrying a bad key or an exhausted quota cannot succeed."""
    assert _map_api_error(status_error(500)).details.get(RETRYABLE_DETAIL_KEY) is True
    for code in (400, 401, 403, 429):
        assert _map_api_error(status_error(code)).details.get(RETRYABLE_DETAIL_KEY) is not True


def test_auth_errors_use_our_wording_not_the_providers():
    """Auth messages are the likeliest to echo request material."""
    secret = "gsk_TOTALLY-SECRET-KEY"
    mapped = _map_api_error(status_error(401, f"Invalid API Key {secret}"))
    assert secret not in mapped.message
    assert "GROQ_API_KEY" in mapped.message


def test_details_never_carry_the_response_body():
    for code in (400, 401, 429, 500):
        assert set(_map_api_error(status_error(code)).details) <= {
            RETRYABLE_DETAIL_KEY,
            "provider",
            "status",
        }


def test_rate_limit_message_states_the_actual_limits():
    message = _map_api_error(status_error(429)).message
    assert "30 requests per minute" in message
    assert "1,000 per day" in message


# --- usage extraction -------------------------------------------------------


def test_usage_is_mapped_including_reasoning_tokens():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=340,
            total_tokens=548,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=88),
        )
    )
    usage = _extract_usage(response)
    assert (usage.prompt_tokens, usage.output_tokens, usage.thought_tokens) == (120, 340, 88)
    assert usage.total_tokens == 548


def test_missing_reasoning_details_is_zero_not_an_error():
    """Not every model reports a reasoning breakdown."""
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    )
    assert _extract_usage(response).thought_tokens == 0


def test_absent_usage_is_none_not_zeros():
    """'Not reported' must never be recorded as 'cost nothing'."""
    assert _extract_usage(SimpleNamespace(usage=None)) is None
    assert _extract_usage(SimpleNamespace()) is None


# --- response parsing -------------------------------------------------------


def completion(content=None, refusal=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal=refusal))]
    )


def test_valid_content_parses():
    payload = json.dumps(
        {"approved": True, "quality_score": 8.5, "issues": [], "feedback": []}
    )
    parsed = _parse_response(completion(payload), EditorOutput)
    assert parsed is not None
    assert parsed.quality_score == 8.5


def test_schema_violations_are_rejected_after_the_provider_call():
    """Stripping constraints from the wire schema must not relax what we accept."""
    over_range = json.dumps(
        {"approved": True, "quality_score": 99.0, "issues": [], "feedback": []}
    )
    assert _parse_response(completion(over_range), EditorOutput) is None


def test_extra_fields_are_still_rejected():
    drifted = json.dumps(
        {"approved": True, "quality_score": 8.0, "issues": [], "feedback": [], "rewrite": "x"}
    )
    assert _parse_response(completion(drifted), EditorOutput) is None


@pytest.mark.parametrize(
    "response",
    [
        completion(None),
        completion(""),
        completion("not json"),
        completion(refusal="I cannot help with that"),
        SimpleNamespace(choices=[]),
        SimpleNamespace(),
    ],
)
def test_unusable_responses_yield_none(response):
    assert _parse_response(response, EditorOutput) is None


# --- factory ----------------------------------------------------------------


def test_factory_builds_the_groq_provider():
    provider = build_provider(
        Settings(
            llm_provider="groq",
            groq_api_key="gsk_test",
            groq_model="openai/gpt-oss-20b",
            app_env="test",
        )
    )
    assert isinstance(provider, GroqProvider)
    assert provider.name == "groq"
    assert provider.model == "openai/gpt-oss-20b"


def test_missing_groq_key_raises_configuration_error():
    with pytest.raises(ConfigurationError) as exc:
        build_provider(Settings(llm_provider="groq", groq_api_key="", app_env="test"))
    assert "GROQ_API_KEY" in exc.value.message


def test_default_groq_model_supports_strict_schema_mode():
    """Llama models on Groq cannot enforce a schema, which every agent needs."""
    assert Settings(app_env="test").groq_model.startswith("openai/gpt-oss")


def test_both_providers_satisfy_the_same_contract():
    """The point of the abstraction: agents cannot tell them apart."""
    from app.services.llm.gemini import GeminiProvider

    for provider_type in (GeminiProvider, GroqProvider):
        assert hasattr(provider_type, "generate_structured")
        assert hasattr(provider_type, "name")
        assert hasattr(provider_type, "model")
