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
    bare_model, detect_provider, parse_openrouter_routing,
    stream, stream_openai_compat,
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


# ── Provider / quantization routing (@suffix) ────────────────────────────


@pytest.mark.parametrize("model_id,expected_model,expected_body", [
    # plain passthrough — no routing
    ("deepseek/deepseek-v4-flash",
     "deepseek/deepseek-v4-flash", None),
    # pin a secondary provider
    ("deepseek/deepseek-v4-flash@gmicloud",
     "deepseek/deepseek-v4-flash",
     {"order": ["gmicloud"], "allow_fallbacks": False}),
    # pin provider + quantization — the user-reported case
    ("deepseek/deepseek-v4-flash@gmicloud/fp8",
     "deepseek/deepseek-v4-flash",
     {"order": ["gmicloud"], "allow_fallbacks": False,
      "quantizations": ["fp8"]}),
    # quantization only (no provider pin)
    ("deepseek/deepseek-v4-flash@fp8",
     "deepseek/deepseek-v4-flash",
     {"quantizations": ["fp8"]}),
    # multiple quantizations
    ("deepseek/deepseek-v4-flash@fp4/int8",
     "deepseek/deepseek-v4-flash",
     {"quantizations": ["fp4", "int8"]}),
])
def test_parse_openrouter_routing(model_id, expected_model, expected_body):
    """`@<provider>[/<quant>]` must be split off the model ID into a provider
    routing body; the model field keeps the real vendor/model ID."""
    assert parse_openrouter_routing(model_id) == (expected_model, expected_body)


def test_stream_dispatches_to_openrouter_endpoint(monkeypatch):
    """`stream()` must resolve openrouter/... to the OpenRouter base_url and
    pass the full vendor/model ID through, using the OPENROUTER_API_KEY."""
    captured = {}

    def fake_stream(api_key, base_url, model, system, messages, tool_schemas, config):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        captured["model"] = model
        captured["config"] = config
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


def test_stream_splits_routing_suffix_off_model(monkeypatch):
    """`openrouter/<vendor>/<model>@gmicloud/fp8` must send the real model ID
    in the model field and the provider/quantization routing via config (which
    stream_openai_compat turns into the `provider` request body)."""
    captured = {}

    def fake_stream(api_key, base_url, model, system, messages, tool_schemas, config):
        captured["model"] = model
        captured["provider_body"] = config.get("_openrouter_provider")
        yield TextChunk("hi")
        yield AssistantTurn("hi", [], in_tokens=1, out_tokens=1)

    monkeypatch.setattr("cheetahclaws.providers.stream_openai_compat", fake_stream)

    cfg = {"openrouter_api_key": "sk-test-123"}
    events = list(stream(
        "openrouter/deepseek/deepseek-v4-flash@gmicloud/fp8",
        "sys", [], [], cfg,
    ))

    assert captured["model"] == "deepseek/deepseek-v4-flash"
    assert captured["provider_body"] == {
        "order": ["gmicloud"],
        "allow_fallbacks": False,
        "quantizations": ["fp8"],
    }
    assert any(isinstance(ev, AssistantTurn) for ev in events)


def test_openai_compat_sends_provider_as_request_body(monkeypatch):
    """`stream_openai_compat` must forward the parsed routing as the `provider`
    request-body element while the `model` field keeps the real model ID."""
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return []  # no chunks → function yields a clean AssistantTurn

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = FakeChat

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    routing = {"order": ["gmicloud"], "allow_fallbacks": False,
               "quantizations": ["fp8"]}
    cfg = {"_openrouter_provider": routing}
    events = list(stream_openai_compat(
        "sk-x", "https://openrouter.ai/api/v1", "deepseek/deepseek-v4-flash",
        "sys", [], [], cfg,
    ))

    assert captured["kwargs"]["model"] == "deepseek/deepseek-v4-flash"
    assert captured["kwargs"]["extra_body"]["provider"] == routing
    assert any(isinstance(ev, AssistantTurn) for ev in events)
