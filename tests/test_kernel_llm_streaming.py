"""Tests for LLM streaming integration (RFC 0027)."""
from __future__ import annotations

import json
import os
import sys

import pytest

from cheetahclaws.kernel import (
    Kernel,
    SandboxPolicy,
    ToolRegistry,
    register_builtin_tools,
)
from cheetahclaws.kernel.runner.llm import (
    LlmRequest,
    LlmResponse,
    MockProvider,
    ProviderInvalidRequest,
    ScriptedMockProvider,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="streaming tests spawn POSIX subprocesses",
)


# ── LlmRequest.stream ─────────────────────────────────────────────────


def test_request_stream_default_false():
    r = LlmRequest(model="m", user="hi")
    assert r.stream is False


def test_request_stream_round_trip():
    r = LlmRequest(model="m", user="hi", stream=True)
    d = r.to_dict()
    assert d["stream"] is True
    r2 = LlmRequest.from_dict(d)
    assert r2.stream is True


def test_request_stream_must_be_bool():
    with pytest.raises(ProviderInvalidRequest):
        LlmRequest(model="m", user="hi",
                    stream="yes")  # type: ignore[arg-type]


# ── ScriptedMockProvider.stream ───────────────────────────────────────


def _resp(text="ok", **kw):
    defaults = dict(text=text, tokens_input=1, tokens_output=1,
                     cost_micro=10, model="m")
    defaults.update(kw)
    return LlmResponse(**defaults)


def test_scripted_stream_emits_each_char():
    p = ScriptedMockProvider([_resp("hello")])
    received: list = []
    response = p.stream(LlmRequest(model="m", user="hi", stream=True),
                         on_delta=lambda d: received.append(d))
    assert received == ["h", "e", "l", "l", "o"]
    assert response.text == "hello"


def test_scripted_stream_advances_cursor():
    p = ScriptedMockProvider([_resp("a"), _resp("b")])
    p.stream(LlmRequest(model="m", user="x", stream=True),
              on_delta=lambda d: None)
    assert p.remaining == 1
    p.stream(LlmRequest(model="m", user="x", stream=True),
              on_delta=lambda d: None)
    assert p.remaining == 0


def test_scripted_stream_rejects_non_callable():
    p = ScriptedMockProvider([_resp("x")])
    with pytest.raises(ProviderInvalidRequest):
        p.stream(LlmRequest(model="m", user="x"),
                  on_delta="not callable")  # type: ignore[arg-type]


def test_scripted_stream_empty_text_zero_deltas():
    """Tool-use responses (text='') emit zero deltas but still
    return the response."""
    resp = LlmResponse(
        text="", tokens_input=10, tokens_output=2, cost_micro=50,
        model="m", finish_reason="tool_use",
        tool_calls=({"id": "t1", "name": "X", "input": {}},),
    )
    p = ScriptedMockProvider([resp])
    received: list = []
    out = p.stream(LlmRequest(model="m", user="x"),
                    on_delta=lambda d: received.append(d))
    assert received == []
    assert out.is_tool_use is True


# ── Subprocess: streaming end-to-end ─────────────────────────────────


def _scripted_env(responses: list[dict]) -> dict:
    return {
        **os.environ,
        "CC_LLM_PROVIDER":              "scripted",
        "CC_LLM_SCRIPTED_RESPONSES_JSON": json.dumps(responses),
    }


@pytest.fixture
def kernel(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        yield k


def test_subprocess_streams_each_char_to_chunks(kernel):
    a = kernel.create_agent(name="x", template="t")
    sup = kernel.make_supervisor()
    responses = [
        _resp("hi", tokens_input=5, tokens_output=2,
               cost_micro=50).to_dict(),
    ]
    received: list = []
    sup.spawn(
        pid=a.pid,
        argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.llm"],
        policy=SandboxPolicy(wall_seconds=15),
        init_payload={"model": "m", "user": "hello",
                       "stream": True},
        env=_scripted_env(responses),
    )
    info = sup.wait(a.pid, timeout=20,
                     on_chunk=lambda c: received.append(c["content"]))
    assert info.exit_kind == "completed"
    assert info.text == "hi"
    assert received == ["h", "i"]
    # info.chunks also contains the delta entries.
    assert len(info.chunks) == 2
    for c in info.chunks:
        assert c["op"]   == "chunk"
        assert c["kind"] == "text"
        assert "iter" in c["metadata"]


def test_subprocess_stream_false_no_chunks(kernel):
    """stream=False (default) → no chunk messages emitted."""
    a = kernel.create_agent(name="x", template="t")
    sup = kernel.make_supervisor()
    responses = [_resp("hi").to_dict()]
    received: list = []
    sup.spawn(
        pid=a.pid,
        argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.llm"],
        policy=SandboxPolicy(wall_seconds=15),
        init_payload={"model": "m", "user": "x"},     # stream omitted
        env=_scripted_env(responses),
    )
    info = sup.wait(a.pid, timeout=20,
                     on_chunk=lambda c: received.append(c["content"]))
    assert info.exit_kind == "completed"
    assert received == []        # no streaming
    assert info.chunks == ()     # back-compat


def test_subprocess_stream_with_long_text(kernel):
    a = kernel.create_agent(name="x", template="t")
    sup = kernel.make_supervisor()
    text = "the quick brown fox"
    responses = [_resp(text, tokens_input=20, tokens_output=10,
                       cost_micro=200).to_dict()]
    received: list = []
    sup.spawn(
        pid=a.pid,
        argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.llm"],
        policy=SandboxPolicy(wall_seconds=15),
        init_payload={"model": "m", "user": "x", "stream": True},
        env=_scripted_env(responses),
    )
    info = sup.wait(a.pid, timeout=20,
                     on_chunk=lambda c: received.append(c["content"]))
    assert info.exit_kind == "completed"
    assert "".join(received) == text
    assert len(info.chunks) == len(text)


def test_subprocess_stream_with_unicode(kernel):
    a = kernel.create_agent(name="x", template="t")
    sup = kernel.make_supervisor()
    text = "你好世界"        # 4 chinese chars
    responses = [_resp(text).to_dict()]
    received: list = []
    sup.spawn(
        pid=a.pid,
        argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.llm"],
        policy=SandboxPolicy(wall_seconds=15),
        init_payload={"model": "m", "user": "x", "stream": True},
        env=_scripted_env(responses),
    )
    info = sup.wait(a.pid, timeout=20,
                     on_chunk=lambda c: received.append(c["content"]))
    assert info.exit_kind == "completed"
    assert "".join(received) == text


# ── Multi-iteration: tool_use iter no streaming, text iter streams ───


def test_subprocess_streaming_multi_iteration_tool_call(kernel):
    """Iter 1 returns tool_use (no text → no chunks expected);
    Iter 2 returns final text (streamed char-by-char)."""
    registry = ToolRegistry()
    register_builtin_tools(registry)
    a = kernel.create_agent(name="x", template="t")
    kernel.cap.create(pid=a.pid, tool_grants=["Echo"])
    sup = kernel.make_supervisor(tool_registry=registry)
    responses = [
        # Iter 1 — tool_use (no text).
        LlmResponse(
            text="", tokens_input=10, tokens_output=5, cost_micro=100,
            model="m", finish_reason="tool_use",
            tool_calls=({"id": "t1", "name": "Echo",
                         "input": {"text": "x"}},),
        ).to_dict(),
        # Iter 2 — text response.
        LlmResponse(
            text="done",
            tokens_input=20, tokens_output=10, cost_micro=200,
            model="m", finish_reason="stop",
        ).to_dict(),
    ]
    received: list = []
    sup.spawn(
        pid=a.pid,
        argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.llm"],
        policy=SandboxPolicy(wall_seconds=15),
        init_payload={
            "model": "m", "user": "go",
            "stream": True,
            "tools": [{"name": "Echo", "description": "x",
                       "input_schema": {"type": "object"}}],
        },
        env=_scripted_env(responses),
    )
    info = sup.wait(a.pid, timeout=25,
                     on_chunk=lambda c: received.append(c))
    assert info.exit_kind == "completed"
    assert info.text == "done"
    # Only iter 2 contributed deltas (iter 1 was tool_use, no text).
    contents = [c["content"] for c in received]
    assert "".join(contents) == "done"
    # All chunks tagged with their iter (from the runner's metadata).
    iters = [c["metadata"]["iter"] for c in received]
    assert all(i == 2 for i in iters), iters


# ── Provider without stream() falls back gracefully ──────────────────


def test_subprocess_stream_with_mock_no_stream_method(kernel):
    """MockProvider doesn't define .stream(), so the runner's
    ``provider_supports_stream`` check falls back to non-streaming
    even when stream=True."""
    a = kernel.create_agent(name="x", template="t")
    sup = kernel.make_supervisor()
    received: list = []
    response_json = json.dumps({
        "text": "hi", "tokens_input": 5, "tokens_output": 2,
        "cost_micro": 50, "model": "m",
    })
    env = {**os.environ, "CC_LLM_PROVIDER": "mock",
           "CC_LLM_MOCK_RESPONSE_JSON": response_json}
    sup.spawn(
        pid=a.pid,
        argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.llm"],
        policy=SandboxPolicy(wall_seconds=15),
        init_payload={"model": "m", "user": "x", "stream": True},
        env=env,
    )
    info = sup.wait(a.pid, timeout=20,
                     on_chunk=lambda c: received.append(c["content"]))
    assert info.exit_kind == "completed"
    assert info.text == "hi"
    # MockProvider has no stream() — runner falls back to non-stream.
    assert received == []


# ── Anthropic adapter lazy stream import ─────────────────────────────


def test_anthropic_provider_has_stream_method():
    """The class should expose `stream` as an attribute even
    without anthropic SDK installed."""
    from cheetahclaws.kernel.runner.llm.anthropic_provider import AnthropicProvider
    p = AnthropicProvider(api_key="dummy")
    assert hasattr(p, "stream")
    assert callable(p.stream)
