"""Tests for the NVIDIA NIM provider entry + 429 cascade fallback.

NIM is registered as a free-tier OpenAI-compat provider. The agent loop
swaps to the next model in the curated chain on a rate-limit error,
capped at ``_NIM_FALLBACK_LIMIT`` swaps per turn so a fully-throttled
catalog can't busy-loop.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cheetahclaws import agent
from cheetahclaws.agent import AgentState, run
from cheetahclaws.providers import (
    PROVIDERS, COSTS, AssistantTurn, TextChunk,
    bare_model, detect_provider, nim_next_model,
)


# ── Provider registration ────────────────────────────────────────────────


def test_nim_provider_entry_present():
    assert "nim" in PROVIDERS
    e = PROVIDERS["nim"]
    assert e["type"] == "openai"
    assert e["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert e["api_key_env"] == "NVIDIA_API_KEY"
    assert len(e["models"]) >= 5, "expect a non-trivial curated chain"


def test_nim_costs_are_free():
    """Every model in the curated chain must show $0 — NIM is free-tier."""
    for m in PROVIDERS["nim"]["models"]:
        assert COSTS.get(m) == (0.0, 0.0), (
            f"{m} missing from COSTS or not zero — UI would show 'unknown'."
        )


@pytest.mark.parametrize("model_id,expected_bare", [
    ("nim/meta/llama-3.3-70b-instruct",     "meta/llama-3.3-70b-instruct"),
    ("nim/deepseek-ai/deepseek-r1",         "deepseek-ai/deepseek-r1"),
    ("nim/qwen/qwen2.5-coder-32b-instruct", "qwen/qwen2.5-coder-32b-instruct"),
])
def test_nim_routing_strips_only_first_segment(model_id, expected_bare):
    """`nim/<vendor>/<model>` must route to nim and keep the vendor/model bare."""
    assert detect_provider(model_id) == "nim"
    assert bare_model(model_id) == expected_bare


# ── nim_next_model chain cycling ─────────────────────────────────────────


def test_nim_next_model_cycles_through_chain():
    chain = PROVIDERS["nim"]["models"]
    cur = chain[0]
    seen = []
    for _ in range(len(chain) + 1):  # +1 to verify wrap-around
        seen.append(cur)
        cur = nim_next_model(cur)
    # First (len(chain)) entries cover every model exactly once.
    assert set(seen[:len(chain)]) == set(chain)
    # Then it wraps back to the first model.
    assert seen[-1] == chain[0]


def test_nim_next_model_preserves_prefix():
    cur = "nim/" + PROVIDERS["nim"]["models"][0]
    nxt = nim_next_model(cur)
    assert nxt.startswith("nim/"), "prefix must be preserved when input had it"


def test_nim_next_model_unknown_starts_at_head():
    """Unknown bare model → return the chain head, so an off-list user
    invocation still cycles into known-good territory on the first 429."""
    nxt = nim_next_model("nim/private-org/private-finetune")
    assert nxt == "nim/" + PROVIDERS["nim"]["models"][0]


# ── Agent loop 429 cascade ───────────────────────────────────────────────


def _fake_turn(text="ok"):
    t = AssistantTurn.__new__(AssistantTurn)
    t.text = text
    t.tool_calls = []
    t.in_tokens = 1
    t.out_tokens = 1
    t.cache_read_tokens = 0
    t.cache_write_tokens = 0
    return t


class _FakeRateLimit(Exception):
    """OpenAI-style 429 — error_classifier matches the substring."""
    def __init__(self):
        super().__init__("rate_limit_exceeded: too many requests (429)")


def _baseline_config():
    return {
        "model":          "nim/meta/llama-3.3-70b-instruct",
        "permission_mode": "auto",
        "no_tools":       False,
        "_session_id":    "test-nim",
        "nim_auto_fallback": True,
    }


def test_nim_429_swaps_to_next_model_then_succeeds(monkeypatch):
    """First call raises 429 → loop swaps to next NIM model → second call succeeds."""
    call_log = []
    state_box = {"first": True}

    def fake_stream(model, **_):
        call_log.append(model)
        if state_box["first"]:
            state_box["first"] = False
            raise _FakeRateLimit()
        yield _fake_turn("done")

    monkeypatch.setattr(agent, "stream", fake_stream)
    state = AgentState()
    list(run("hi", state, _baseline_config(), system_prompt="sys"))

    assert len(call_log) == 2, f"expected 2 stream calls, got {len(call_log)}"
    assert call_log[0] == "nim/meta/llama-3.3-70b-instruct"
    assert call_log[1] == nim_next_model("nim/meta/llama-3.3-70b-instruct")


def test_nim_fallback_capped_to_limit(monkeypatch):
    """Continuous 429 must stop after _NIM_FALLBACK_LIMIT (3) swaps so
    a fully-throttled catalog can't busy-loop. After the cap, the loop
    falls through to the regular retry/backoff path and ultimately fails."""
    call_log = []

    def always_429(model, **_):
        call_log.append(model)
        raise _FakeRateLimit()
        yield  # unreachable, makes this a generator

    monkeypatch.setattr(agent, "stream", always_429)
    # Speed up the test: skip the fallthrough backoff sleep.
    monkeypatch.setattr(agent.time, "sleep", lambda *_: None)

    state = AgentState()
    list(run("hi", state, _baseline_config(), system_prompt="sys"))

    # Expected: 1 initial + 3 fallback swaps = 4 swaps, then fall through to
    # the regular retry path (max_retries=3) for a total of 4 + 3 = 7-ish.
    # We just assert the lower bound: at least the cap was honored, and the
    # loop didn't hang.
    assert 4 <= len(call_log) <= 8, (
        f"unexpected call count {len(call_log)}; "
        f"chain must terminate after fallback cap"
    )


def test_nim_fallback_disabled_via_config(monkeypatch):
    """When `nim_auto_fallback=False`, 429 falls through to the regular
    retry path immediately — no model swap."""
    call_log = []

    def always_429(model, **_):
        call_log.append(model)
        raise _FakeRateLimit()
        yield

    monkeypatch.setattr(agent, "stream", always_429)
    monkeypatch.setattr(agent.time, "sleep", lambda *_: None)

    cfg = {**_baseline_config(), "nim_auto_fallback": False}
    state = AgentState()
    list(run("hi", state, cfg, system_prompt="sys"))

    # Every call must use the original model — no swaps happened.
    assert all(m == "nim/meta/llama-3.3-70b-instruct" for m in call_log), (
        f"fallback should be disabled, but model changed: {call_log}"
    )


def test_nim_fallback_does_not_apply_to_other_providers(monkeypatch):
    """A 429 from openai or anthropic must NOT trigger a NIM swap."""
    call_log = []

    def always_429(model, **_):
        call_log.append(model)
        raise _FakeRateLimit()
        yield

    monkeypatch.setattr(agent, "stream", always_429)
    monkeypatch.setattr(agent.time, "sleep", lambda *_: None)

    cfg = {**_baseline_config(), "model": "claude-opus-4-7"}
    state = AgentState()
    list(run("hi", state, cfg, system_prompt="sys"))

    # Model never changes — no NIM swap leaked into the openai/anthropic path.
    assert all(m == "claude-opus-4-7" for m in call_log)
