"""Prompt-cache injection + SDK client-reuse coverage for the Anthropic path.

Layers covered:
1. cache_control breakpoints (last tool schema, system block, last content
   block of the final message) present by default, absent with
   prompt_cache=False, and never more than the API's 4-breakpoint limit
   (including with extended thinking enabled).
2. Copy-on-write: the shared tool-schema registry dicts and the neutral
   session messages are never mutated by annotation.
3. Legacy shape: prompt_cache=False produces exactly today's kwargs
   (plain-string system, zero cache_control keys anywhere).
4. Proxy fallback: a 400 naming cache_control triggers one retry without
   breakpoints and disables caching for that endpoint; transient errors
   that merely mention the field do not.
5. Quota projection reserves the 1.25x cache-write premium exactly when
   the next request will actually carry cache_control.
6. Cross-provider isolation: the OpenAI-compat request payload never
   contains cache_control.
7. calc_cost prices cache tokens at 0.1x read / 1.25x write.

(SDK client reuse is covered separately in tests/test_client_reuse.py.)
All tests fake the SDKs via sys.modules — no network, no API key.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest


# ── Fake Anthropic SDK ─────────────────────────────────────────────────────

_CAPTURED: dict = {}


class _FakeUsage:
    input_tokens = 10
    output_tokens = 2
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class _FakeFinal:
    content: list = []
    usage = _FakeUsage()


class _FakeStreamCtx:
    def __init__(self, kwargs: dict):
        _CAPTURED.clear()
        _CAPTURED.update(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return _FakeFinal()


class _FakeMessages:
    def stream(self, **kwargs):
        return _FakeStreamCtx(kwargs)


class _FakeAnthropicClient:
    instantiations = 0

    def __init__(self, api_key: str = "", base_url: str = ""):
        type(self).instantiations += 1
        self.api_key = api_key
        self.base_url = base_url
        self.closed = False
        self.messages = _FakeMessages()

    def close(self):
        self.closed = True


@pytest.fixture()
def fake_anthropic(monkeypatch):
    """Install a fake `anthropic` module and reset provider-level caches."""
    from cheetahclaws import providers
    _CAPTURED.clear()
    _FakeAnthropicClient.instantiations = 0
    fake_mod = SimpleNamespace(Anthropic=_FakeAnthropicClient)
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
    providers._client_cache_clear()
    providers._cache_control_disabled.clear()
    yield _CAPTURED
    providers._client_cache_clear()
    providers._cache_control_disabled.clear()


def _call_stream(config: dict, messages: list, tools: list, system="SYSTEM PROMPT"):
    from cheetahclaws.providers import stream_anthropic
    return list(stream_anthropic(
        api_key="k1",
        model="claude-sonnet-4-5",
        system=system,
        messages=messages,
        tool_schemas=tools,
        config=config,
    ))


def _make_tools():
    return [
        {"name": "Read", "description": "read", "input_schema": {"type": "object"}},
        {"name": "Bash", "description": "run", "input_schema": {"type": "object"}},
    ]


def _make_msgs():
    return [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok", "tool_calls": [
            {"id": "t1", "name": "Read", "input": {"file_path": "/x"}},
        ]},
        {"role": "tool", "tool_call_id": "t1", "name": "Read", "content": "data"},
    ]


def _count_cache_controls(kwargs: dict) -> int:
    n = 0
    for t in kwargs.get("tools", []) or []:
        n += "cache_control" in t
    system = kwargs.get("system")
    if isinstance(system, list):
        n += sum("cache_control" in b for b in system if isinstance(b, dict))
    for m in kwargs.get("messages", []) or []:
        c = m.get("content")
        if isinstance(c, list):
            n += sum("cache_control" in b for b in c if isinstance(b, dict))
    return n


# ── 1: breakpoint placement ────────────────────────────────────────────────

def test_cache_breakpoints_present_by_default(fake_anthropic):
    _call_stream({"model": "claude-sonnet-4-5"}, _make_msgs(), _make_tools())
    kw = fake_anthropic

    assert isinstance(kw["system"], list)
    assert kw["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert kw["system"][-1]["text"] == "SYSTEM PROMPT"

    assert "cache_control" not in kw["tools"][0]
    assert kw["tools"][-1]["cache_control"] == {"type": "ephemeral"}

    # Final converted message is the collected tool_result block list.
    last_content = kw["messages"][-1]["content"]
    assert isinstance(last_content, list)
    assert last_content[-1]["cache_control"] == {"type": "ephemeral"}
    assert last_content[-1]["type"] == "tool_result"


def test_exactly_three_breakpoints_within_api_limit(fake_anthropic):
    _call_stream({"model": "claude-sonnet-4-5"}, _make_msgs(), _make_tools())
    assert _count_cache_controls(fake_anthropic) == 3


def test_thinking_enabled_keeps_breakpoint_count(fake_anthropic):
    _call_stream({"model": "claude-sonnet-4-5", "thinking": True},
                 _make_msgs(), _make_tools())
    assert fake_anthropic["thinking"]["type"] == "enabled"
    n = _count_cache_controls(fake_anthropic)
    assert n == 3
    assert n <= 4


def test_string_user_final_message_wrapped_in_text_block(fake_anthropic):
    _call_stream({"model": "claude-sonnet-4-5"},
                 [{"role": "user", "content": "plain"}], _make_tools())
    last = fake_anthropic["messages"][-1]["content"]
    assert last == [{"type": "text", "text": "plain",
                     "cache_control": {"type": "ephemeral"}}]


def test_no_tools_no_system_still_safe(fake_anthropic):
    _call_stream({"model": "claude-sonnet-4-5"},
                 [{"role": "user", "content": "q"}], [], system="")
    kw = fake_anthropic
    assert kw["tools"] == []
    assert kw["system"] == ""          # empty system left untouched
    assert _count_cache_controls(kw) == 1   # only the message breakpoint


def test_volatile_env_block_excluded_from_cached_system_span(fake_anthropic):
    """The per-run '# Environment' block (git status / date) mutates turn to
    turn. The cache breakpoint must land on the STABLE span before it — so a
    changed git status can't invalidate the large static base prompt across
    turns — while the volatile tail carries no breakpoint of its own. The two
    system blocks must still concatenate to the original string byte-for-byte
    (the model sees identical content)."""
    base = "You are a helpful agent.\nFollow the rules."
    env = ("# Environment\n- Current date: 2026-07-11 Saturday\n"
           "- Git status:\n  M providers.py\n")
    system = base + "\n\n" + env          # exactly how build_system_prompt joins
    _call_stream({"model": "claude-sonnet-4-5"},
                 [{"role": "user", "content": "q"}], _make_tools(), system=system)
    sys_blocks = fake_anthropic["system"]
    assert isinstance(sys_blocks, list) and len(sys_blocks) == 2
    stable, volatile = sys_blocks
    # Breakpoint on the stable prefix only; volatile tail uncached.
    assert stable["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in volatile
    # The volatile git status lives entirely in the uncached tail.
    assert "Git status" in volatile["text"]
    assert "Git status" not in stable["text"]
    # Byte-for-byte identical content — caching boundary only, not a prompt change.
    assert stable["text"] + volatile["text"] == system
    # Still within the API's 4-breakpoint budget: tools + stable system + msg.
    assert _count_cache_controls(fake_anthropic) == 3


def test_marker_free_system_still_cached_whole(fake_anthropic):
    """A custom system prompt without the environment marker (e.g. the
    autonomous agent-runner) is cached as a single block, exactly as before."""
    _call_stream({"model": "claude-sonnet-4-5"},
                 [{"role": "user", "content": "q"}], _make_tools(),
                 system="CUSTOM AUTONOMOUS PROMPT")
    sys_blocks = fake_anthropic["system"]
    assert sys_blocks == [{"type": "text", "text": "CUSTOM AUTONOMOUS PROMPT",
                           "cache_control": {"type": "ephemeral"}}]


# ── 2: copy-on-write ───────────────────────────────────────────────────────

def test_registry_tool_schemas_not_mutated(fake_anthropic):
    tools = _make_tools()
    snapshot = json.dumps(tools, sort_keys=True)
    _call_stream({"model": "claude-sonnet-4-5"}, _make_msgs(), tools)
    assert json.dumps(tools, sort_keys=True) == snapshot


def test_neutral_messages_not_mutated(fake_anthropic):
    msgs = _make_msgs()
    snapshot = json.dumps(msgs, sort_keys=True)
    _call_stream({"model": "claude-sonnet-4-5"}, msgs, _make_tools())
    assert json.dumps(msgs, sort_keys=True) == snapshot
    assert "cache_control" not in json.dumps(msgs)


# ── 3: legacy shape with the flag off ──────────────────────────────────────

def test_prompt_cache_false_restores_legacy_shape(fake_anthropic):
    _call_stream({"model": "claude-sonnet-4-5", "prompt_cache": False},
                 _make_msgs(), _make_tools())
    kw = fake_anthropic
    assert kw["system"] == "SYSTEM PROMPT"
    assert "cache_control" not in json.dumps(kw["tools"])
    assert "cache_control" not in json.dumps(kw["messages"])


# ── 4: proxy 400 fallback ──────────────────────────────────────────────────

def test_cache_control_rejection_retries_without_and_disables(monkeypatch):
    from cheetahclaws import providers
    providers._client_cache_clear()
    providers._cache_control_disabled.clear()

    calls: list[dict] = []

    class _RejectingStreamCtx(_FakeStreamCtx):
        def __init__(self, kwargs):
            calls.append(kwargs)
            if "cache_control" in json.dumps(kwargs):
                raise RuntimeError(
                    "400 invalid_request_error: unexpected field cache_control")
            super().__init__(kwargs)

    class _RejectingMessages:
        def stream(self, **kwargs):
            return _RejectingStreamCtx(kwargs)

    class _RejectingClient(_FakeAnthropicClient):
        def __init__(self, api_key="", base_url=""):
            super().__init__(api_key=api_key, base_url=base_url)
            self.messages = _RejectingMessages()

    fake_mod = SimpleNamespace(Anthropic=_RejectingClient)
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    cfg = {"model": "claude-sonnet-4-5",
           "anthropic_endpoint": "https://proxy.local"}
    events = _call_stream(cfg, [{"role": "user", "content": "q"}], _make_tools())

    # First attempt carried breakpoints and was rejected; retry succeeded bare.
    assert len(calls) == 2
    assert "cache_control" in json.dumps(calls[0])
    assert "cache_control" not in json.dumps(calls[1])
    assert events  # AssistantTurn still produced

    # Endpoint is disabled for the rest of the process: next call goes bare.
    calls.clear()
    _call_stream(cfg, [{"role": "user", "content": "q2"}], _make_tools())
    assert len(calls) == 1
    assert "cache_control" not in json.dumps(calls[0])

    providers._client_cache_clear()
    providers._cache_control_disabled.clear()


# ── rejection-detection narrowing ──────────────────────────────────────────

def test_rejection_detector_requires_schema_signal():
    from cheetahclaws.providers import _is_cache_control_rejection

    assert _is_cache_control_rejection(
        RuntimeError("400 invalid_request_error: unexpected field cache_control"))
    err = RuntimeError("cache_control not supported")
    err.status_code = 400
    assert _is_cache_control_rejection(err)

    # Transient errors that merely mention the field must NOT disable caching.
    assert not _is_cache_control_rejection(
        RuntimeError("connection reset while sending cache_control block"))
    assert not _is_cache_control_rejection(RuntimeError("400 bad request"))


def test_transient_error_mentioning_cache_control_does_not_disable(monkeypatch):
    from cheetahclaws import providers
    providers._client_cache_clear()
    providers._cache_control_disabled.clear()

    calls: list[dict] = []

    class _FlakyStreamCtx(_FakeStreamCtx):
        def __init__(self, kwargs):
            calls.append(kwargs)
            raise RuntimeError("connection reset while sending cache_control block")

    class _FlakyMessages:
        def stream(self, **kwargs):
            return _FlakyStreamCtx(kwargs)

    class _FlakyClient(_FakeAnthropicClient):
        def __init__(self, api_key="", base_url=""):
            super().__init__(api_key=api_key, base_url=base_url)
            self.messages = _FlakyMessages()

    monkeypatch.setitem(sys.modules, "anthropic",
                        SimpleNamespace(Anthropic=_FlakyClient))

    with pytest.raises(RuntimeError):
        _call_stream({"model": "claude-sonnet-4-5"},
                     [{"role": "user", "content": "q"}], _make_tools())
    # No blind fallback retry, and the endpoint stays cache-enabled.
    assert len(calls) == 1
    assert not providers._cache_control_disabled

    providers._client_cache_clear()


# ── quota projection reserves the cache-write premium ──────────────────────

def test_quota_projection_reserves_cache_write_premium(monkeypatch, tmp_path):
    """A cost budget between the raw input estimate and its 1.25x cache-write
    ceiling must pause BEFORE the call when prompt caching is on, and proceed
    when it is off — the reviewer scenario of a full-miss overshooting an
    approved budget."""
    from cheetahclaws import tools as _tools_init  # noqa: F401 - register tools
    from cheetahclaws import quota
    from cheetahclaws.agent import AgentState, run, QuotaPause
    from cheetahclaws.providers import AssistantTurn, calc_cost

    monkeypatch.setattr(quota, "_quota_dir", lambda: tmp_path)

    def fake_stream(**_kwargs):
        yield AssistantTurn("done", [], 0, 0)

    monkeypatch.setattr("cheetahclaws.agent.stream", fake_stream)

    model = "claude-sonnet-4-6"
    prompt = "x" * 28_000          # ≈11k estimated input tokens
    from cheetahclaws.compaction import estimate_tokens
    proj = (estimate_tokens([{"role": "user", "content": prompt}])
            + estimate_tokens([{"role": "system", "content": "sys"}]))
    raw_cost = calc_cost(model, proj, 0)
    budget = raw_cost * 1.1        # raw fits, 1.25x-reserved does not

    def _run(cfg_extra, sid):
        quota._sess_tokens.pop(sid, None)
        quota._sess_cost.pop(sid, None)
        state = AgentState()
        cfg = {"model": model, "permission_mode": "accept-all",
               "_session_id": sid, "disabled_tools": ["Agent"],
               "session_cost_budget": budget, **cfg_extra}
        return list(run(prompt, state, cfg, "sys"))

    events_on = _run({}, "quota_cache_on")
    assert any(isinstance(e, QuotaPause) for e in events_on)

    events_off = _run({"prompt_cache": False}, "quota_cache_off")
    assert not any(isinstance(e, QuotaPause) for e in events_off)

    # Reviewer scenario: after a proxy rejection disabled the endpoint, the
    # real request goes out raw — the projection must NOT reserve the 1.25x
    # premium anymore, or a budget the raw request fits within would wrongly
    # pause. is_prompt_cache_active() keeps both sides consistent.
    from cheetahclaws import providers
    providers._cache_control_disabled.add("https://api.anthropic.com")
    try:
        events_disabled = _run({}, "quota_cache_disabled_ep")
        assert not any(isinstance(e, QuotaPause) for e in events_disabled)
    finally:
        providers._cache_control_disabled.clear()


# ── 6: cross-provider isolation ────────────────────────────────────────────

def test_openai_compat_payload_never_contains_cache_control(monkeypatch):
    captured: dict = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return iter(())

    class _FakeOpenAIClient:
        def __init__(self, api_key="", base_url=""):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    fake_mod = SimpleNamespace(OpenAI=_FakeOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", fake_mod)

    from cheetahclaws.providers import stream_openai_compat
    list(stream_openai_compat(
        api_key="k", base_url="https://api.openai.com/v1",
        model="gpt-4o", system="SYS",
        messages=_make_msgs(), tool_schemas=_make_tools(),
        config={"model": "gpt-4o", "prompt_cache": True},
    ))
    assert captured
    assert "cache_control" not in json.dumps(captured, default=str)


# ── 7: cache-aware pricing ─────────────────────────────────────────────────

def test_calc_cost_prices_cache_tokens():
    from cheetahclaws.providers import calc_cost, COSTS, bare_model
    model = "claude-sonnet-4-6"
    ic, oc = COSTS.get(bare_model(model), (0.0, 0.0))
    assert ic > 0, "test model must have a price entry"

    base = calc_cost(model, 1000, 0)
    with_cache = calc_cost(model, 1000, 0,
                           cache_read_tok=10_000, cache_write_tok=1000)
    expected_extra = (10_000 * 0.1 + 1000 * 1.25) * ic / 1_000_000
    assert with_cache == pytest.approx(base + expected_extra)


def test_calc_cost_backward_compatible():
    from cheetahclaws.providers import calc_cost
    assert calc_cost("claude-sonnet-4-5", 100, 50) == calc_cost(
        "claude-sonnet-4-5", 100, 50, 0, 0)
