"""Tests for IPC streaming chunks (RFC 0026)."""
from __future__ import annotations

import os
import sys

import pytest

from cheetahclaws.kernel import (
    Kernel,
    RunnerExitInfo,
    SandboxPolicy,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="streaming tests spawn POSIX subprocesses",
)


RUNNER_ARGV = [sys.executable, "-m", "cheetahclaws.kernel.runner.runner_main"]


@pytest.fixture
def stack(tmp_path):
    k = Kernel.open(tmp_path / "kernel.db")
    sup = k.make_supervisor()
    yield {"kernel": k, "sup": sup}
    for h in sup.list():
        try:
            sup.stop(h.pid)
        except Exception:
            pass
    k.close()


def _spawn_chunks(stack, n: int):
    a = stack["kernel"].create_agent(name=f"x{n}", template="t")
    stack["sup"].spawn(
        pid=a.pid,
        argv=RUNNER_ARGV,
        policy=SandboxPolicy(wall_seconds=10),
        env={**os.environ, "CC_RUNNER_BEHAVIOR": f"chunks={n}"},
    )
    return a.pid


# ── Default: no chunks ──────────────────────────────────────────────────


def test_no_chunks_when_runner_doesnt_emit(stack):
    """Echo runner doesn't emit chunks → info.chunks == ()."""
    a = stack["kernel"].create_agent(name="x", template="t")
    stack["sup"].spawn(
        pid=a.pid, argv=RUNNER_ARGV,
        policy=SandboxPolicy(wall_seconds=10),
    )
    info = stack["sup"].wait(a.pid, timeout=15)
    assert info.exit_kind == "completed"
    assert info.chunks == ()


def test_runner_exit_info_chunks_default():
    """RunnerExitInfo.chunks defaults to empty tuple — backward
    compatible default."""
    info = RunnerExitInfo(
        pid=1, exit_kind="completed", exit_code=0,
        stdout_tail=b"", stderr_tail=b"",
        duration_s=0.0, ledger_charged={},
    )
    assert info.chunks == ()


# ── 5-chunk emission ────────────────────────────────────────────────────


def test_chunks_appear_in_info(stack):
    pid = _spawn_chunks(stack, 5)
    info = stack["sup"].wait(pid, timeout=15)
    assert info.exit_kind == "completed"
    assert len(info.chunks) == 5
    contents = [c["content"] for c in info.chunks]
    assert contents == ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"]


def test_chunks_carry_kind_and_metadata(stack):
    pid = _spawn_chunks(stack, 2)
    info = stack["sup"].wait(pid, timeout=15)
    for i, c in enumerate(info.chunks, 1):
        assert c["op"]   == "chunk"
        assert c["kind"] == "text"
        assert c["metadata"]["i"]  == i
        assert c["metadata"]["of"] == 2


# ── on_chunk callback ──────────────────────────────────────────────────


def test_on_chunk_fires_per_chunk_in_order(stack):
    pid = _spawn_chunks(stack, 5)
    received: list = []
    info = stack["sup"].wait(pid, timeout=15,
                              on_chunk=lambda c: received.append(c["content"]))
    assert info.exit_kind == "completed"
    assert received == ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"]
    # And info.chunks has the same.
    assert [c["content"] for c in info.chunks] == received


def test_on_chunk_not_called_when_no_chunks(stack):
    a = stack["kernel"].create_agent(name="echo", template="t")
    stack["sup"].spawn(
        pid=a.pid, argv=RUNNER_ARGV,
        policy=SandboxPolicy(wall_seconds=10),
    )
    received: list = []
    info = stack["sup"].wait(a.pid, timeout=15,
                              on_chunk=lambda c: received.append(c))
    assert info.exit_kind == "completed"
    assert received == []


def test_on_chunk_callback_exception_doesnt_break_loop(stack):
    """A bad callback shouldn't break the wait loop or lose
    subsequent chunks."""
    pid = _spawn_chunks(stack, 3)
    received: list = []

    def bad_then_ok(c):
        received.append(c["content"])
        if c["metadata"]["i"] == 1:
            raise RuntimeError("boom")

    info = stack["sup"].wait(pid, timeout=15, on_chunk=bad_then_ok)
    assert info.exit_kind == "completed"
    # Despite the first callback raising, all 3 chunks were still
    # processed by the loop and stored in info.chunks.
    assert len(info.chunks) == 3
    # The second + third callbacks fired despite the first's
    # exception (chunks 2, 3 in the received list).
    assert "chunk-2" in received
    assert "chunk-3" in received


def test_on_chunk_default_none_works(stack):
    """on_chunk=None (default) shouldn't crash — same path as no
    callback."""
    pid = _spawn_chunks(stack, 3)
    info = stack["sup"].wait(pid, timeout=15)
    assert len(info.chunks) == 3


# ── Single chunk ────────────────────────────────────────────────────────


def test_single_chunk(stack):
    pid = _spawn_chunks(stack, 1)
    info = stack["sup"].wait(pid, timeout=15)
    assert len(info.chunks) == 1
    assert info.chunks[0]["content"] == "chunk-1"


# ── Many chunks ─────────────────────────────────────────────────────────


def test_many_chunks(stack):
    """Stress: 50 chunks → all delivered in order."""
    pid = _spawn_chunks(stack, 50)
    info = stack["sup"].wait(pid, timeout=20)
    assert info.exit_kind == "completed"
    assert len(info.chunks) == 50
    contents = [c["content"] for c in info.chunks]
    assert contents == [f"chunk-{i}" for i in range(1, 51)]


# ── Mixed with regular IPC traffic ─────────────────────────────────────


def test_chunks_dont_interfere_with_other_messages(stack):
    """Echo runner emits iteration_start, log, iteration_done, exit
    — chunks ALSO emitted between them shouldn't disrupt that flow."""
    pid = _spawn_chunks(stack, 2)
    info = stack["sup"].wait(pid, timeout=15)
    # Normal exit_kind path.
    assert info.exit_kind == "completed"
    # Chunks captured.
    assert len(info.chunks) == 2
    # exit summary still computed (even if empty for echo).
    assert info.text == "" or info.text is not None
