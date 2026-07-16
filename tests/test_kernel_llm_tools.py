"""Tests for LLM tool calling integration (RFC 0022)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from cheetahclaws.kernel import (
    Kernel,
    LedgerStore,
    SandboxPolicy,
    ToolRegistry,
    register_builtin_tools,
)
from cheetahclaws.kernel.runner.llm import (
    LlmRequest,
    LlmResponse,
    MockProvider,
    ProviderInvalidRequest,
    ProviderUnavailable,
    ScriptedMockProvider,
)


pytestmark_subprocess = pytest.mark.skipif(
    os.name != "posix",
    reason="end-to-end LLM runner tests spawn POSIX subprocesses",
)


# ── LlmRequest.tools ──────────────────────────────────────────────────


def test_request_tools_default_empty():
    r = LlmRequest(model="m", user="hi")
    assert r.tools == ()


def test_request_tools_round_trip():
    r = LlmRequest(
        model="m", user="hi",
        tools=({"name": "X", "description": "x"},),
    )
    d = r.to_dict()
    assert d["tools"] == [{"name": "X", "description": "x"}]
    r2 = LlmRequest.from_dict(d)
    assert r2.tools == r.tools


def test_request_messages_with_list_content_accepted():
    """Multi-content shape (text + tool_use blocks) must be valid."""
    r = LlmRequest(
        model="m",
        messages=(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "calling tool"},
                {"type": "tool_use", "id": "t1", "name": "X", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            ]},
        ),
    )
    assert len(r.messages) == 3


def test_request_messages_content_must_be_str_or_list():
    with pytest.raises(ProviderInvalidRequest):
        LlmRequest(model="m", messages=(
            {"role": "user", "content": 42},
        ))


def test_request_messages_list_blocks_must_be_dicts():
    with pytest.raises(ProviderInvalidRequest):
        LlmRequest(model="m", messages=(
            {"role": "user", "content": ["just a string"]},
        ))


# ── LlmResponse.tool_calls ───────────────────────────────────────────


def test_response_tool_calls_default_empty():
    r = LlmResponse(text="hi", tokens_input=1, tokens_output=1,
                     cost_micro=10, model="m")
    assert r.tool_calls == ()
    assert r.is_tool_use is False


def test_response_tool_calls_field():
    r = LlmResponse(
        text="", tokens_input=1, tokens_output=1, cost_micro=10,
        model="m", finish_reason="tool_use",
        tool_calls=({"id": "t1", "name": "X", "input": {"k": "v"}},),
    )
    assert r.is_tool_use is True
    assert r.tool_calls[0]["name"] == "X"


def test_response_tool_calls_validate_id():
    with pytest.raises(ProviderInvalidRequest):
        LlmResponse(text="", tokens_input=0, tokens_output=0,
                     cost_micro=0, model="m",
                     tool_calls=({"id": "", "name": "X", "input": {}},))


def test_response_tool_calls_validate_name():
    with pytest.raises(ProviderInvalidRequest):
        LlmResponse(text="", tokens_input=0, tokens_output=0,
                     cost_micro=0, model="m",
                     tool_calls=({"id": "i", "name": "", "input": {}},))


def test_response_tool_calls_round_trip():
    r = LlmResponse(
        text="", tokens_input=5, tokens_output=2, cost_micro=100,
        model="m",
        tool_calls=({"id": "t1", "name": "Echo", "input": {"text": "hi"}},),
    )
    d = r.to_dict()
    r2 = LlmResponse.from_dict(d)
    assert r2.tool_calls == r.tool_calls
    assert r2.is_tool_use is True


# ── ScriptedMockProvider ────────────────────────────────────────────


def _r(text="ok", **kw):
    defaults = dict(text=text, tokens_input=1, tokens_output=1,
                     cost_micro=10, model="m")
    defaults.update(kw)
    return LlmResponse(**defaults)


def test_scripted_returns_in_order():
    p = ScriptedMockProvider([_r("first"), _r("second"), _r("third")])
    req = LlmRequest(model="m", user="x")
    assert p(req).text == "first"
    assert p(req).text == "second"
    assert p(req).text == "third"


def test_scripted_records_calls():
    p = ScriptedMockProvider([_r(), _r()])
    p(LlmRequest(model="m", user="a"))
    p(LlmRequest(model="m", user="b"))
    assert [c.user for c in p.calls] == ["a", "b"]


def test_scripted_exhaustion_raises():
    p = ScriptedMockProvider([_r()])
    p(LlmRequest(model="m", user="x"))
    with pytest.raises(ProviderUnavailable):
        p(LlmRequest(model="m", user="x"))


def test_scripted_remaining():
    p = ScriptedMockProvider([_r(), _r(), _r()])
    assert p.remaining == 3
    p(LlmRequest(model="m", user="x"))
    assert p.remaining == 2


def test_scripted_empty_rejected():
    with pytest.raises(ProviderInvalidRequest):
        ScriptedMockProvider([])


def test_scripted_non_response_entries_rejected():
    with pytest.raises(ProviderInvalidRequest):
        ScriptedMockProvider([{"text": "not an LlmResponse"}])  # type: ignore[list-item]


def test_scripted_from_env(monkeypatch):
    payload = [
        {"text": "first", "tokens_input": 1, "tokens_output": 1,
         "cost_micro": 10, "model": "m"},
        {"text": "second", "tokens_input": 1, "tokens_output": 1,
         "cost_micro": 10, "model": "m"},
    ]
    monkeypatch.setenv(ScriptedMockProvider.ENV_RESPONSES, json.dumps(payload))
    p = ScriptedMockProvider.from_env()
    assert p(LlmRequest(model="m", user="x")).text == "first"
    assert p(LlmRequest(model="m", user="x")).text == "second"


def test_scripted_from_env_unset_raises(monkeypatch):
    monkeypatch.delenv(ScriptedMockProvider.ENV_RESPONSES, raising=False)
    with pytest.raises(ProviderUnavailable):
        ScriptedMockProvider.from_env()


def test_mock_provider_scripted_factory():
    """MockProvider.scripted() is a convenience constructor."""
    p = MockProvider.scripted([_r("x"), _r("y")])
    assert isinstance(p, ScriptedMockProvider)
    assert p(LlmRequest(model="m", user="q")).text == "x"


# ── End-to-end via supervisor ─────────────────────────────────────────


def _scripted_env(responses: list[dict]) -> dict:
    return {
        **os.environ,
        "CC_LLM_PROVIDER":              "scripted",
        "CC_LLM_SCRIPTED_RESPONSES_JSON": json.dumps(responses),
    }


@pytestmark_subprocess
def test_runner_tool_use_then_final(tmp_path):
    """Iter 1 returns tool_use(Echo); iter 2 returns final text."""
    with Kernel.open(tmp_path / "kernel.db") as k:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        a = k.create_agent(name="x", template="t")
        k.cap.create(pid=a.pid, tool_grants=["Echo"])
        sup = k.make_supervisor(tool_registry=registry)
        responses = [
            LlmResponse(
                text="", tokens_input=20, tokens_output=10, cost_micro=300,
                model="m", finish_reason="tool_use",
                tool_calls=({"id": "t1", "name": "Echo",
                             "input": {"text": "world"}},),
            ).to_dict(),
            LlmResponse(
                text="echo returned 'world'", tokens_input=30,
                tokens_output=15, cost_micro=600, model="m",
                finish_reason="stop",
            ).to_dict(),
        ]
        sup.spawn(
            pid=a.pid,
            argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.llm"],
            policy=SandboxPolicy(wall_seconds=15),
            init_payload={
                "model": "m",
                "user":  "Use Echo with text=world",
                "tools": [{"name": "Echo", "description": "x",
                           "input_schema": {"type": "object"}}],
            },
            env=_scripted_env(responses),
        )
        info = sup.wait(a.pid, timeout=20)
        assert info.exit_kind == "completed", info.stderr_tail
        assert info.text == "echo returned 'world'"
        assert info.metadata["iterations"] == 2
        assert info.metadata["tokens_total"] == 75   # 30+15+20+10
        assert info.metadata["cost_micro"] == 900   # 300+600
        # Audit: one tool dispatch event.
        events = k.process.events_tail(pid=a.pid,
                                         kind="tool.call.dispatched")
        assert len(events) == 1
        assert events[0].payload["tool"] == "Echo"


@pytestmark_subprocess
def test_runner_tool_use_no_tools_field_falls_back_to_text(tmp_path):
    """Even with no tools=[] in init payload, the runner handles
    plain text responses correctly (RFC 0019 path)."""
    with Kernel.open(tmp_path / "kernel.db") as k:
        a = k.create_agent(name="x", template="t")
        sup = k.make_supervisor()
        sup.spawn(
            pid=a.pid,
            argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.llm"],
            policy=SandboxPolicy(wall_seconds=15),
            init_payload={"model": "m", "user": "hi"},
            env=_scripted_env([
                LlmResponse(text="hello", tokens_input=5, tokens_output=2,
                             cost_micro=50, model="m").to_dict(),
            ]),
        )
        info = sup.wait(a.pid, timeout=20)
        assert info.exit_kind == "completed"
        assert info.text == "hello"
        assert info.metadata["iterations"] == 1


@pytestmark_subprocess
def test_runner_max_iterations_cap(tmp_path):
    """Provider that always returns tool_use → runner hits cap."""
    with Kernel.open(tmp_path / "kernel.db") as k:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        a = k.create_agent(name="x", template="t")
        k.cap.create(pid=a.pid, tool_grants=["Echo"])
        sup = k.make_supervisor(tool_registry=registry)
        # 5 responses all returning tool_use; max_iterations=3 caps.
        responses = [
            LlmResponse(
                text="", tokens_input=5, tokens_output=2, cost_micro=50,
                model="m", finish_reason="tool_use",
                tool_calls=({"id": f"t{i}", "name": "Echo",
                             "input": {"text": f"call-{i}"}},),
            ).to_dict()
            for i in range(5)
        ]
        sup.spawn(
            pid=a.pid,
            argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.llm"],
            policy=SandboxPolicy(wall_seconds=20),
            init_payload={
                "model": "m", "user": "x",
                "max_iterations": 3,
                "tools": [{"name": "Echo", "description": "x",
                           "input_schema": {"type": "object"}}],
            },
            env=_scripted_env(responses),
        )
        info = sup.wait(a.pid, timeout=30)
        assert info.exit_kind == "failed"
        assert info.metadata["error"] == "max_iterations"
        assert info.metadata["iterations"] == 3
        # Three tool dispatches (one per iteration).
        events = k.process.events_tail(pid=a.pid,
                                         kind="tool.call.dispatched")
        assert len(events) == 3


@pytestmark_subprocess
def test_runner_tool_denied_continues_loop(tmp_path):
    """When a tool call is denied, the runner still appends the
    error result to messages and continues; the LLM can then
    surface the error in its final answer."""
    with Kernel.open(tmp_path / "kernel.db") as k:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        a = k.create_agent(name="x", template="t")
        # Cap doesn't include Echo.
        k.cap.create(pid=a.pid, tool_grants=["OtherTool"])
        sup = k.make_supervisor(tool_registry=registry)
        responses = [
            LlmResponse(
                text="", tokens_input=10, tokens_output=5, cost_micro=100,
                model="m", finish_reason="tool_use",
                tool_calls=({"id": "t1", "name": "Echo",
                             "input": {"text": "x"}},),
            ).to_dict(),
            LlmResponse(
                text="I couldn't run Echo — permission denied.",
                tokens_input=20, tokens_output=10, cost_micro=200,
                model="m", finish_reason="stop",
            ).to_dict(),
        ]
        sup.spawn(
            pid=a.pid,
            argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.llm"],
            policy=SandboxPolicy(wall_seconds=15),
            init_payload={
                "model": "m", "user": "do echo",
                "tools": [{"name": "Echo", "description": "x",
                           "input_schema": {"type": "object"}}],
            },
            env=_scripted_env(responses),
        )
        info = sup.wait(a.pid, timeout=20)
        assert info.exit_kind == "completed"
        assert "permission denied" in info.text
        # The denied event was recorded.
        denied = k.process.events_tail(pid=a.pid,
                                         kind="tool.call.denied")
        assert len(denied) == 1


@pytestmark_subprocess
def test_runner_charges_accumulate_across_iterations(tmp_path):
    """ledger.tokens reflects all iterations, not just the last."""
    with Kernel.open(tmp_path / "kernel.db") as k:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        a = k.create_agent(name="x", template="t")
        k.cap.create(pid=a.pid, tool_grants=["Echo"])
        k.ledger.create(pid=a.pid, grants={
            "tokens": 1_000_000, "cost_micro": 1_000_000,
        })
        sup = k.make_supervisor(tool_registry=registry)
        responses = [
            LlmResponse(text="", tokens_input=10, tokens_output=5,
                         cost_micro=100, model="m",
                         finish_reason="tool_use",
                         tool_calls=({"id": "t1", "name": "Echo",
                                      "input": {"text": "x"}},)).to_dict(),
            LlmResponse(text="done", tokens_input=20, tokens_output=10,
                         cost_micro=200, model="m",
                         finish_reason="stop").to_dict(),
        ]
        sup.spawn(
            pid=a.pid,
            argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.llm"],
            policy=SandboxPolicy(wall_seconds=15),
            init_payload={"model": "m", "user": "go",
                           "tools": [{"name": "Echo", "description": "x",
                                      "input_schema": {"type": "object"}}]},
            env=_scripted_env(responses),
        )
        info = sup.wait(a.pid, timeout=20)
        assert info.exit_kind == "completed"
        # Ledger: 15 + 30 = 45 tokens; 100 + 200 = 300 cost_micro.
        led = k.ledger.get(a.pid)
        used = {e.dim: e.used for e in led.entries}
        assert used["tokens"]     == 45
        assert used["cost_micro"] == 300


# ── Anthropic adapter compatibility (lazy import path) ───────────────


def test_anthropic_provider_messages_with_tools_doesnt_import_sdk():
    """We can construct LlmRequest with tools without anthropic
    SDK installed — the import is lazy on __call__."""
    from cheetahclaws.kernel.runner.llm.anthropic_provider import AnthropicProvider
    p = AnthropicProvider(api_key="dummy")
    # Just construct a request — provider call would fail without SDK,
    # but we don't call it.
    req = LlmRequest(
        model="claude-x",
        user="hi",
        tools=({"name": "Echo", "description": "x",
                "input_schema": {"type": "object"}},),
    )
    assert req.tools[0]["name"] == "Echo"
