"""Provider errors must keep meaningful categories, not collapse into 502."""

from __future__ import annotations

import pytest
from google.genai import errors

from app.core.exceptions import ConfigurationError, LLMProviderError, LLMRateLimitError
from app.services.llm.base import RETRYABLE_DETAIL_KEY
from app.services.llm.gemini import _map_api_error


def api_error(code: int, message: str = "boom") -> errors.APIError:
    """Build a real APIError the way the SDK does, from a response body."""
    return errors.APIError(code, {"error": {"message": message, "status": "ERROR"}})


def test_rate_limit_maps_to_rate_limited_category():
    mapped = _map_api_error(api_error(429, "Quota exceeded"))
    assert isinstance(mapped, LLMRateLimitError)
    assert mapped.error_type == "rate_limited"
    assert mapped.status_code == 429


def test_rate_limit_is_not_retryable():
    """A spent free-tier quota cannot be fixed by trying again immediately."""
    mapped = _map_api_error(api_error(429))
    assert mapped.details.get(RETRYABLE_DETAIL_KEY) is not True


@pytest.mark.parametrize("code", [401, 403])
def test_auth_failures_map_to_configuration_error(code):
    mapped = _map_api_error(api_error(code, "API key not valid"))
    assert isinstance(mapped, ConfigurationError)
    assert mapped.error_type == "configuration_error"
    assert mapped.status_code == 503
    assert "GEMINI_API_KEY" in mapped.message


@pytest.mark.parametrize("code", [500, 503])
def test_server_errors_are_provider_unavailable_and_retryable(code):
    mapped = _map_api_error(api_error(code))
    assert isinstance(mapped, LLMProviderError)
    assert mapped.error_type == "llm_provider_error"
    assert mapped.details[RETRYABLE_DETAIL_KEY] is True
    assert "temporarily unavailable" in mapped.message


@pytest.mark.parametrize("code", [400, 404])
def test_other_client_errors_stay_non_retryable_provider_errors(code):
    mapped = _map_api_error(api_error(code))
    assert isinstance(mapped, LLMProviderError)
    assert mapped.details.get(RETRYABLE_DETAIL_KEY) is not True


def test_rate_limit_is_still_caught_as_a_provider_error():
    """Subclassing keeps the service's existing except clause working."""
    assert issubclass(LLMRateLimitError, LLMProviderError)


@pytest.mark.parametrize("code", [400, 401, 429, 500])
def test_raw_provider_response_body_never_leaks(code):
    """`APIError.details` holds the raw response and must not be propagated."""
    secret = "AIzaSyTOTALLY-SECRET-KEY"
    exc = errors.APIError(code, {"error": {"message": f"key {secret} rejected", "apiKey": secret}})
    mapped = _map_api_error(exc)

    assert secret not in str(mapped.details), "provider response body leaked into details"
    assert set(mapped.details) <= {RETRYABLE_DETAIL_KEY, "provider", "status"}
    if code in (401, 403):
        # Auth messages are the likeliest to echo credentials, so our own
        # wording replaces the provider's entirely.
        assert secret not in mapped.message


def test_status_code_is_recorded_for_diagnostics():
    assert _map_api_error(api_error(429)).details["status"] == 429
