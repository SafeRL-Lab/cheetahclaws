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
    bare_model, calc_cost, detect_provider, lookup_model_key,
    parse_openrouter_routing, stream, stream_openai_compat,
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


# ── Provider identity must survive the prefix strip ──────────────────────
#
# `stream()` hands `stream_openai_compat` a model string with the provider
# prefix already removed, so an OpenRouter route arrives as a plain upstream
# path ("deepseek/deepseek-v4-flash").  Re-deriving the provider from that
# string reads it as the *DeepSeek* provider — the tests below pin the
# behaviour that must not regress.


def _capture_request(monkeypatch):
    """Patch openai.OpenAI and return the dict that receives create()'s kwargs."""
    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return []

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = FakeChat

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    return captured


def test_openrouter_deepseek_route_omits_deepseek_only_fields(monkeypatch):
    """`extra_body.thinking` / `reasoning_effort` are DeepSeek-API fields.
    An openrouter/deepseek/... route must not pick them up just because the
    upstream vendor segment reads "deepseek"."""
    captured = _capture_request(monkeypatch)

    list(stream(
        "openrouter/deepseek/deepseek-v4-flash", "sys", [], [],
        {"openrouter_api_key": "sk-x", "thinking": False,
         "reasoning_effort": "high"},
    ))

    kwargs = captured["kwargs"]
    assert "thinking" not in (kwargs.get("extra_body") or {})
    assert "reasoning_effort" not in kwargs


def test_deepseek_provider_still_gets_its_thinking_toggle(monkeypatch):
    """Guard the other direction: the real DeepSeek provider keeps the field."""
    captured = _capture_request(monkeypatch)

    list(stream("deepseek/deepseek-v4-flash", "sys", [], [],
                {"deepseek_api_key": "sk-x", "thinking": False}))

    assert captured["kwargs"]["extra_body"]["thinking"] == {"type": "disabled"}


def test_openrouter_uses_max_tokens_and_its_own_cap(monkeypatch):
    """OpenRouter's documented output field is `max_tokens`.  A route whose
    vendor segment is "openai" must not switch to the OpenAI-only
    `max_completion_tokens`, and the cap must come from the openrouter entry."""
    captured = _capture_request(monkeypatch)

    list(stream("openrouter/openai/gpt-5", "sys", [], [],
                {"openrouter_api_key": "sk-x", "max_tokens": 64000}))

    kwargs = captured["kwargs"]
    assert "max_completion_tokens" not in kwargs
    assert kwargs["max_tokens"] <= PROVIDERS["openrouter"]["max_completion_tokens"]


# ── Per-model registry lookups (cost, context window) ────────────────────


@pytest.mark.parametrize("model_id,expected_key", [
    ("openrouter/deepseek/deepseek-v4-flash",             "deepseek-v4-flash"),
    ("openrouter/deepseek/deepseek-v4-flash@gmicloud/fp8", "deepseek-v4-flash"),
    ("openrouter/anthropic/claude-sonnet-4-6",            "claude-sonnet-4-6"),
    ("gpt-4o",                                            "gpt-4o"),
])
def test_lookup_model_key_strips_prefixes_and_routing(model_id, expected_key):
    assert lookup_model_key(model_id) == expected_key


def test_openrouter_usage_is_priced():
    """OpenRouter bills real money — a $0.00 estimate would let a session sail
    past the cost budget in quota.record_usage."""
    direct  = calc_cost("deepseek-v4-flash", 1_000_000, 1_000_000)
    gateway = calc_cost("openrouter/deepseek/deepseek-v4-flash", 1_000_000, 1_000_000)
    assert direct > 0
    assert gateway == direct
    # The routing suffix must not knock the lookup out either.
    assert calc_cost("openrouter/deepseek/deepseek-v4-flash@gmicloud/fp8",
                     1_000_000, 1_000_000) == direct


def test_openrouter_context_window_resolves_per_model():
    """A gateway route must resolve the real model's window, not the
    provider-level default."""
    from cheetahclaws.compaction import get_context_limit
    model = "openrouter/meta-llama/llama-3.3-70b-instruct"
    assert (get_context_limit(model, {"model": model})
            == get_context_limit("llama-3.3-70b-instruct", {}))


def test_routing_suffix_keeps_model_family_overlay():
    """The prompt overlay routes on the model-name tail; the `@provider/quant`
    suffix must not make "claude-sonnet-4-6@gmicloud/fp8" tail to "fp8"."""
    from cheetahclaws.prompts.select import _family_overlay_for_model
    assert (_family_overlay_for_model(
        "openrouter/anthropic/claude-sonnet-4-6@gmicloud/fp8") == "claude.md")


def test_openrouter_context_window_falls_back_to_vendor_provider():
    """When the per-model registry has no entry, a gateway route should still
    beat the gateway's generic default by reading the vendor's own window
    (openrouter/anthropic/… → Anthropic's 200k, not OpenRouter's 128k)."""
    from cheetahclaws.compaction import get_context_limit
    model = "openrouter/anthropic/claude-sonnet-4-6"
    assert (get_context_limit(model, {"model": model})
            == PROVIDERS["anthropic"]["context_limit"])
