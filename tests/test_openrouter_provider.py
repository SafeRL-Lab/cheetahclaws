"""Tests for the OpenRouter provider entry + multi-level model routing.

OpenRouter serves 400+ models from many vendors behind one OpenAI-compatible
endpoint. Model IDs keep the upstream <vendor>/<model> path, so calls use the
double-prefixed form `openrouter/<vendor>/<model>`, e.g.
`openrouter/deepseek/deepseek-v4-flash` — the first segment is the provider
and everything after it is passed through verbatim to the API.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cheetahclaws.providers import (
    PROVIDERS, AssistantTurn, TextChunk,
    bare_model, detect_provider, stream,
)


# ── Provider registration ────────────────────────────────────────────────


def test_openrouter_provider_entry_present():
    assert "openrouter" in PROVIDERS
    e = PROVIDERS["openrouter"]
    assert e["type"] == "openai"
    assert e["base_url"] == "https://openrouter.ai/api/v1"
    assert e["api_key_env"] == "OPENROUTER_API_KEY"
    assert len(e["models"]) >= 5, "expect a curated model list for the /model picker"


@pytest.mark.parametrize("model_id,expected_bare", [
    ("openrouter/deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash"),
    ("openrouter/deepseek/deepseek-v4-pro",   "deepseek/deepseek-v4-pro"),
    ("openrouter/anthropic/claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"),
])
def test_openrouter_routing_strips_only_first_segment(model_id, expected_bare):
    """`openrouter/<vendor>/<model>` must route to openrouter and keep the
    vendor/model bare — that's exactly the ID OpenRouter's API expects."""
    assert detect_provider(model_id) == "openrouter"
    assert bare_model(model_id) == expected_bare


def test_stream_dispatches_to_openrouter_endpoint(monkeypatch):
    """`stream()` must resolve openrouter/... to the OpenRouter base_url and
    pass the full vendor/model ID through, using the OPENROUTER_API_KEY."""
    captured = {}

    def fake_stream(api_key, base_url, model, system, messages, tool_schemas, config):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        captured["model"] = model
        yield TextChunk("hi")
        yield AssistantTurn("hi", [], in_tokens=1, out_tokens=1)

    monkeypatch.setattr("cheetahclaws.providers.stream_openai_compat", fake_stream)

    cfg = {"openrouter_api_key": "sk-test-123"}
    events = list(stream(
        "openrouter/deepseek/deepseek-v4-flash",
        "sys", [], [], cfg,
    ))

    assert captured["api_key"] == "sk-test-123"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["model"] == "deepseek/deepseek-v4-flash"
    assert any(isinstance(ev, AssistantTurn) for ev in events)
