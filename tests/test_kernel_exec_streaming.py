"""Tests for Exec streaming (RFC 0028)."""
from __future__ import annotations

import os
import sys
import time

import pytest

from cheetahclaws.kernel import (
    Kernel,
    SandboxPolicy,
    ToolRegistry,
    register_builtin_tools,
)
from cheetahclaws.kernel.tools.exec_tool import exec_handler, register_exec_tool
from cheetahclaws.kernel.tools.registry import (
    ToolContext,
    ToolInvalidArgs,
    dispatch_tool_call,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="Exec streaming uses POSIX subprocess primitives",
)


# ── ToolContext.on_chunk plumbing ─────────────────────────────────────


def test_tool_context_on_chunk_default_none():
    ctx = ToolContext(pid=1, kernel=None)
    assert ctx.on_chunk is None


def test_tool_context_on_chunk_round_trip():
    cb = lambda x: None       # noqa: E731
    ctx = ToolContext(pid=1, kernel=None, on_chunk=cb)
    assert ctx.on_chunk is cb


def test_dispatch_passes_on_chunk_into_ctx():
    """The dispatch wrapper must thread on_chunk into the
    ToolContext seen by the handler."""
    captured: dict = {}

    def _spy(args, ctx):
        captured["on_chunk"] = ctx.on_chunk
        return {"ok": True}

    from cheetahclaws.kernel.tools.registry import Tool, ToolRegistry

    reg = ToolRegistry()
    reg.register(Tool(
        name="spy", description="x", handler=_spy,
        requires_capability=False,
    ))
    cb = lambda x: None       # noqa: E731
    resp = dispatch_tool_call(
        msg={"tool": "spy", "tool_call_id": "t1", "args": {}},
        pid=1, registry=reg, kernel=None, on_chunk=cb,
    )
    assert resp["ok"] is True
    assert captured["on_chunk"] is cb


# ── Exec stream arg validation ────────────────────────────────────────


def test_exec_stream_arg_must_be_bool():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        exec_handler({
            "argv": ["/bin/echo", "hi"],
            "stream": "yes",
        }, ctx)


def test_exec_stream_default_false_no_chunks():
    """stream omitted (default False) → zero chunks even if
    on_chunk is supplied."""
    received: list = []
    ctx = ToolContext(pid=1, kernel=None,
                       on_chunk=lambda x: received.append(x))
    result = exec_handler({
        "argv": ["/bin/echo", "hello"],
        "timeout_s": 5,
    }, ctx)
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]
    assert received == []        # no streaming


def test_exec_stream_true_but_no_callback_falls_back():
    """stream=True but ctx.on_chunk is None → buffered path."""
    ctx = ToolContext(pid=1, kernel=None)         # no on_chunk
    result = exec_handler({
        "argv": ["/bin/echo", "hi"],
        "stream": True,
        "timeout_s": 5,
    }, ctx)
    assert result["exit_code"] == 0


# ── Streaming path: chunk emission ────────────────────────────────────


def _run_streaming(argv: list, **kw) -> tuple[dict, list]:
    received: list = []
    ctx = ToolContext(
        pid=1, kernel=None,
        on_chunk=lambda x: received.append(x),
    )
    result = exec_handler({
        "argv": argv, "stream": True, "timeout_s": 10, **kw,
    }, ctx)
    return result, received


def test_exec_streams_each_stdout_line():
    result, chunks = _run_streaming(
        ["/bin/sh", "-c", "echo a; echo b; echo c"],
    )
    assert result["exit_code"] == 0
    out_chunks = [c for c in chunks if c["kind"] == "stdout"]
    assert [c["content"] for c in out_chunks] == ["a\n", "b\n", "c\n"]
    # All chunks are well-shaped.
    for c in out_chunks:
        assert c["op"] == "chunk"
        assert c["metadata"] == {"tool": "Exec"}


def test_exec_streams_stderr_separately():
    result, chunks = _run_streaming(
        ["/bin/sh", "-c", "echo out1; echo err1 1>&2; echo out2"],
    )
    assert result["exit_code"] == 0
    kinds = [c["kind"] for c in chunks]
    assert kinds.count("stdout") == 2
    assert kinds.count("stderr") == 1
    err = next(c for c in chunks if c["kind"] == "stderr")
    assert err["content"] == "err1\n"


def test_exec_streamed_content_matches_final_stdout():
    """Concatenating streamed stdout chunks reproduces the final
    result.stdout string."""
    result, chunks = _run_streaming(
        ["/bin/sh", "-c", "for i in 1 2 3 4 5; do echo line$i; done"],
    )
    assert result["exit_code"] == 0
    stdout_streamed = "".join(
        c["content"] for c in chunks if c["kind"] == "stdout"
    )
    assert stdout_streamed == result["stdout"]


def test_exec_streaming_real_time_arrival():
    """At least one chunk must arrive *before* the process exits.
    We launch a script that emits one line, sleeps, then exits.
    The wait()-side callback should see the line during the
    sleep, not after the process is reaped."""
    received_at: list = []
    start = time.monotonic()

    def cb(payload):
        received_at.append(time.monotonic() - start)

    ctx = ToolContext(pid=1, kernel=None, on_chunk=cb)
    result = exec_handler({
        "argv": ["/bin/sh", "-c",
                 "echo first; sleep 0.5; echo second"],
        "stream": True, "timeout_s": 5,
    }, ctx)
    total = time.monotonic() - start
    assert result["exit_code"] == 0
    # First chunk should arrive well before total runtime.
    assert received_at, "expected at least one chunk"
    assert received_at[0] < total - 0.2, (
        f"first chunk arrived at {received_at[0]:.2f}s, "
        f"total runtime {total:.2f}s — looks buffered, not streamed"
    )


# ── Wall-clock timeout under streaming ───────────────────────────────


def test_exec_streaming_respects_wall_timeout():
    """A long-running infinite producer should be killed by
    wall_seconds; chunks already streamed are preserved."""
    received: list = []
    ctx = ToolContext(pid=1, kernel=None,
                       on_chunk=lambda x: received.append(x))
    result = exec_handler({
        "argv": ["/bin/sh", "-c",
                 "while :; do echo tick; sleep 0.05; done"],
        "stream": True, "timeout_s": 1,
    }, ctx)
    assert result["timed_out"] is True
    # We should have streamed at least a few ticks before the kill.
    tick_chunks = [c for c in received if "tick" in c["content"]]
    assert len(tick_chunks) >= 2


# ── Bad callback can't crash the tool ────────────────────────────────


def test_exec_streaming_swallows_bad_callback():
    def bad_cb(payload):
        raise RuntimeError("user callback boom")
    ctx = ToolContext(pid=1, kernel=None, on_chunk=bad_cb)
    result = exec_handler({
        "argv": ["/bin/echo", "hi"],
        "stream": True, "timeout_s": 5,
    }, ctx)
    assert result["exit_code"] == 0
    assert "hi" in result["stdout"]


# ── End-to-end via supervisor.wait(on_chunk=...) ────────────────────


@pytest.fixture
def kernel(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        yield k


def test_exec_streaming_end_to_end(kernel):
    """A subprocess agent that calls Exec(stream=True) — the
    streamed stdout chunks must reach the supervisor's
    on_chunk callback in real time."""
    registry = ToolRegistry()
    register_builtin_tools(registry)
    register_exec_tool(registry)
    a = kernel.create_agent(name="x", template="t")
    kernel.cap.create(
        pid=a.pid,
        tool_grants=["Exec"],
        fs_grants=[
            {"mode": "r", "prefix": "/bin/sh"},
            {"mode": "r", "prefix": "/bin/echo"},
        ],
    )
    sup = kernel.make_supervisor(tool_registry=registry,
                                  tool_kernel=kernel)

    # Inline driver script: send a tool_call, await response,
    # then send exit. Lives in tests/ as a string and is exec'd
    # via -c so we don't need a fixture file.
    driver = (
        "import sys\n"
        "from cheetahclaws.kernel.runner.ipc import JsonLineChannel\n"
        "ch = JsonLineChannel(sys.stdin.buffer, sys.stdout.buffer)\n"
        "init = ch.recv(timeout=10)\n"
        "ch.send({'op':'ready','pid': init['pid']})\n"
        "ch.send({'op':'tool_call','tool_call_id':'t1',\n"
        "         'tool':'Exec',\n"
        "         'args':{'argv':['/bin/sh','-c',\n"
        "                          'echo a; echo b'],\n"
        "                  'stream': True,\n"
        "                  'timeout_s': 5}})\n"
        "resp = ch.recv(timeout=10)\n"
        "ch.send({'op':'exit','exit_kind':'completed',\n"
        "         'summary':'ok',\n"
        "         'text': resp.get('result',{}).get('stdout','')})\n"
    )
    received: list = []
    sup.spawn(
        pid=a.pid,
        argv=[sys.executable, "-c", driver],
        policy=SandboxPolicy(wall_seconds=15),
        init_payload={},
    )
    info = sup.wait(a.pid, timeout=20,
                     on_chunk=lambda c: received.append(c))
    assert info.exit_kind == "completed"
    # Streamed chunks reached the wait()-time callback.
    exec_chunks = [c for c in received
                   if c.get("metadata", {}).get("tool") == "Exec"]
    assert len(exec_chunks) >= 2
    streamed = "".join(
        c["content"] for c in exec_chunks if c["kind"] == "stdout"
    )
    assert streamed == "a\nb\n"
