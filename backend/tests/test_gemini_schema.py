"""Regression tests for provider schema compatibility.

The Gemini generateContent endpoint rejects `additional_properties`, which
Pydantic emits for every model configured with `extra="forbid"`. Local schema
conversion succeeds regardless - only the live API refuses it - so these tests
pin the adapter's sanitisation in place.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.content import Post, WriterOutput
from app.models.factcheck import FactCheckClaim, FactCheckReport
from app.models.plan import ContentAngle, ContentPlan
from app.models.research import ResearchFact, ResearchOutput
from app.models.review import EditorIssue, EditorOutput
from app.services.llm.gemini import _to_provider_schema

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


def collect_additional_properties(schema, path="root"):
    """Every additional_properties value in the converted schema tree."""
    found = [(path, schema.additional_properties)]
    for name, prop in (schema.properties or {}).items():
        found += collect_additional_properties(prop, f"{path}.{name}")
    if schema.items is not None:
        found += collect_additional_properties(schema.items, f"{path}[]")
    return found


@pytest.mark.parametrize("model", LLM_FACING_MODELS, ids=lambda m: m.__name__)
def test_additional_properties_is_stripped_everywhere(model):
    """Including nested objects inside arrays, where the second occurrence hid."""
    offenders = [
        (path, value)
        for path, value in collect_additional_properties(_to_provider_schema(model))
        if value is not None
    ]
    assert offenders == [], f"additional_properties would be sent to the API: {offenders}"


@pytest.mark.parametrize("model", LLM_FACING_MODELS, ids=lambda m: m.__name__)
def test_schema_still_describes_the_model(model):
    """Sanitisation must not strip the parts the provider needs."""
    schema = _to_provider_schema(model)
    assert set(schema.properties or {}) == set(model.model_fields)
    assert set(schema.required or []) == set(model.model_fields)


def test_strictness_is_preserved_on_the_way_back():
    """extra='forbid' still rejects drift; only the wire schema is relaxed."""
    with pytest.raises(ValidationError):
        EditorOutput.model_validate(
            {
                "approved": True,
                "quality_score": 9.0,
                "issues": [],
                "feedback": [],
                "unexpected": "field",
            }
        )
