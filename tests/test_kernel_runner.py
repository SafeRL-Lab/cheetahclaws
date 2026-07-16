"""Tests for kernel.runner (RFC 0016) — subprocess agent runner.

Spawns real subprocesses via runner_main. POSIX-only (sandbox is
POSIX); skipped on Windows.
"""
from __future__ import annotations

import io
import json
import os
import signal
import sys
import threading
import time

import pytest

from cheetahclaws.kernel import (
    AgentState,
    JsonLineChannel,
    KernelStore,
    LedgerStore,
    RunnerExitInfo,
    RunnerHandle,
    RunnerHandshakeFailed,
    RunnerIllegalState,
    RunnerSupervisor,
    SandboxPolicy,
    UnknownPid,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="runner spawns POSIX-style subprocesses",
)


RUNNER_ARGV = [sys.executable, "-m", "cheetahclaws.kernel.runner.runner_main"]


@pytest.fixture
def stores(tmp_path):
    ks  = KernelStore.open(tmp_path / "kernel.db")
    led = LedgerStore(ks.connection, write_lock=ks.write_lock)
    sup = RunnerSupervisor(ks, ledger_store=led)
    yield {"ks": ks, "led": led, "sup": sup}
    # Clean up any leaked subprocesses.
    for h in sup.list():
        try:
            sup.stop(h.pid)
        except Exception:
            pass
    ks.close()


def _make_agent(ks: KernelStore, name: str = "alice"):
    return ks.create(name=name, template="t")


# ── JsonLineChannel unit tests ──────────────────────────────────────────


def test_jsonline_send_recv_roundtrip():
    """Use BytesIO pairs to verify the protocol without a subprocess."""
    inbound  = io.BytesIO()
    outbound = io.BytesIO()
    chan = JsonLineChannel(inbound, outbound)
    chan.send({"op": "init", "pid": 1, "payload": {"x": 1}})
    # Find what was written to outbound.
    written = outbound.getvalue()
    assert written.endswith(b"\n")
    parsed = json.loads(written)
    assert parsed["op"] == "init"
    assert parsed["pid"] == 1


def test_jsonline_recv_parses_dict():
    inbound = io.BytesIO(b'{"op":"ready","pid":42}\n')
    outbound = io.BytesIO()
    chan = JsonLineChannel(inbound, outbound)
    msg = chan.recv()
    assert msg == {"op": "ready", "pid": 42}


def test_jsonline_recv_eof_raises():
    inbound = io.BytesIO(b"")
    outbound = io.BytesIO()
    chan = JsonLineChannel(inbound, outbound)
    with pytest.raises(EOFError):
        chan.recv()


def test_jsonline_recv_invalid_json_raises():
    inbound = io.BytesIO(b"not-json\n")
    outbound = io.BytesIO()
    chan = JsonLineChannel(inbound, outbound)
    with pytest.raises(ValueError):
        chan.recv()


def test_jsonline_recv_non_dict_raises():
    inbound = io.BytesIO(b"42\n")
    outbound = io.BytesIO()
    chan = JsonLineChannel(inbound, outbound)
    with pytest.raises(ValueError):
        chan.recv()


def test_jsonline_send_rejects_non_dict():
    chan = JsonLineChannel(io.BytesIO(), io.BytesIO())
    with pytest.raises(TypeError):
        chan.send("not a dict")  # type: ignore[arg-type]


# ── Spawn → wait happy path ─────────────────────────────────────────────


def test_spawn_transitions_ready_to_running(stores):
    sup, ks = stores["sup"], stores["ks"]
    a = _make_agent(ks)
    handle = sup.spawn(
        pid=a.pid, argv=RUNNER_ARGV,
        policy=SandboxPolicy(wall_seconds=10),
    )
    # State is RUNNING after spawn.
    assert ks.get(a.pid).state == AgentState.RUNNING
    # Cleanup.
    sup.wait(a.pid, timeout=10)
    assert ks.get(a.pid).state == AgentState.DEAD


def test_spawn_returns_handle(stores):
    sup, ks = stores["sup"], stores["ks"]
    a = _make_agent(ks)
    handle = sup.spawn(
        pid=a.pid, argv=RUNNER_ARGV,
        policy=SandboxPolicy(wall_seconds=10),
    )
    assert isinstance(handle, RunnerHandle)
    assert handle.pid == a.pid
    assert handle.os_pid > 0
    assert handle.is_alive() is True
    sup.wait(a.pid, timeout=10)


def test_clean_runner_exits_completed(stores):
    sup, ks = stores["sup"], stores["ks"]
    a = _make_agent(ks)
    sup.spawn(pid=a.pid, argv=RUNNER_ARGV,
              policy=SandboxPolicy(wall_seconds=10))
    info = sup.wait(a.pid, timeout=10)
    assert isinstance(info, RunnerExitInfo)
    assert info.exit_kind == "completed"
    assert info.exit_code == 0
    assert info.pid == a.pid
    assert info.duration_s >= 0
    # Agent is DEAD with right exit_kind.
    agent = ks.get(a.pid)
    assert agent.state == AgentState.DEAD
    assert agent.exit_kind == "completed"


# ── Illegal state ───────────────────────────────────────────────────────


def test_spawn_rejects_non_ready_agent(stores):
    sup, ks = stores["sup"], stores["ks"]
    a = _make_agent(ks)
    ks.transition(a.pid, AgentState.RUNNING)
    with pytest.raises(RunnerIllegalState):
        sup.spawn(pid=a.pid, argv=RUNNER_ARGV,
                  policy=SandboxPolicy(wall_seconds=10))


def test_spawn_rejects_dead_agent(stores):
    sup, ks = stores["sup"], stores["ks"]
    a = _make_agent(ks)
    ks.terminate(a.pid, exit_kind="completed")
    with pytest.raises(RunnerIllegalState):
        sup.spawn(pid=a.pid, argv=RUNNER_ARGV)


def test_spawn_rejects_double_spawn(stores):
    sup, ks = stores["sup"], stores["ks"]
    a = _make_agent(ks)
    sup.spawn(pid=a.pid, argv=RUNNER_ARGV,
              policy=SandboxPolicy(wall_seconds=10))
    with pytest.raises(RunnerIllegalState):
        sup.spawn(pid=a.pid, argv=RUNNER_ARGV)
    sup.wait(a.pid, timeout=10)


def test_spawn_unknown_pid_raises(stores):
    sup = stores["sup"]
    with pytest.raises(UnknownPid):
        sup.spawn(pid=9999, argv=RUNNER_ARGV)


# ── Crash isolation ─────────────────────────────────────────────────────


def test_kill_9_does_not_kill_supervisor(stores):
    """Hard-kill the runner; supervisor + daemon must survive."""
    sup, ks = stores["sup"], stores["ks"]
    a = _make_agent(ks, name="loop")
    handle = sup.spawn(
        pid=a.pid, argv=RUNNER_ARGV,
        env={**os.environ, "CC_RUNNER_BEHAVIOR": "loop"},
        policy=SandboxPolicy(wall_seconds=10),
    )
    # External hard kill.
    os.kill(handle.os_pid, signal.SIGKILL)
    info = sup.wait(a.pid, timeout=10)
    assert info.exit_kind == "crashed"
    assert info.exit_code != 0
    # Supervisor still works for new spawns.
    b = _make_agent(ks, name="next")
    sup.spawn(pid=b.pid, argv=RUNNER_ARGV,
              policy=SandboxPolicy(wall_seconds=10))
    sup.wait(b.pid, timeout=10)


def test_runner_crash_exits_non_zero(stores):
    sup, ks = stores["sup"], stores["ks"]
    a = _make_agent(ks)
    sup.spawn(
        pid=a.pid, argv=RUNNER_ARGV,
        env={**os.environ, "CC_RUNNER_BEHAVIOR": "crash"},
        policy=SandboxPolicy(wall_seconds=10),
    )
    info = sup.wait(a.pid, timeout=10)
    # CC_RUNNER_BEHAVIOR=crash exits 1 without sending 'exit'.
    assert info.exit_code != 0
    assert info.exit_kind in ("crashed", "failed")


# ── Sandbox / wall-clock kill ───────────────────────────────────────────


def test_wall_seconds_kills_runaway(stores):
    """A loop runner must be killed by wall-clock enforcement."""
    sup, ks = stores["sup"], stores["ks"]
    a = _make_agent(ks)
    start = time.monotonic()
    sup.spawn(
        pid=a.pid, argv=RUNNER_ARGV,
        env={**os.environ, "CC_RUNNER_BEHAVIOR": "loop"},
        policy=SandboxPolicy(wall_seconds=2.0, cpu_seconds=2),
    )
    info = sup.wait(a.pid, timeout=15)
    elapsed = time.monotonic() - start
    assert elapsed < 12, f"runner ran for {elapsed}s"
    assert info.exit_code != 0


# ── stop() ───────────────────────────────────────────────────────────────


def test_stop_kills_running_runner(stores):
    sup, ks = stores["sup"], stores["ks"]
    a = _make_agent(ks)
    sup.spawn(
        pid=a.pid, argv=RUNNER_ARGV,
        env={**os.environ, "CC_RUNNER_BEHAVIOR": "loop"},
        policy=SandboxPolicy(wall_seconds=30),
    )
    start = time.monotonic()
    info = sup.stop(a.pid)
    elapsed = time.monotonic() - start
    # Should escalate to SIGKILL within ipc_timeout_s + grace.
    assert elapsed < 10, f"stop took {elapsed}s"
    assert ks.get(a.pid).state == AgentState.DEAD


def test_stop_unknown_pid_raises(stores):
    sup = stores["sup"]
    with pytest.raises(Exception):  # RunnerUnknownPid
        sup.stop(9999)


# ── Ledger integration ──────────────────────────────────────────────────


def test_wall_s_charged_when_dim_exists(stores):
    sup, ks, led = stores["sup"], stores["ks"], stores["led"]
    a = _make_agent(ks)
    led.create(pid=a.pid, grants={"wall_s": 1000})
    sup.spawn(
        pid=a.pid, argv=RUNNER_ARGV,
        env={**os.environ, "CC_RUNNER_BEHAVIOR": "slow=1.5"},
        policy=SandboxPolicy(wall_seconds=10),
    )
    info = sup.wait(a.pid, timeout=15)
    assert info.exit_kind == "completed"
    # Charged: at least 1 second of wall time.
    assert info.ledger_charged.get("wall_s", 0) >= 1
    led_obj = led.get(a.pid)
    assert led_obj.entries[0].used >= 1


def test_wall_s_silent_when_dim_missing(stores):
    """Without a wall_s ledger row, the supervisor doesn't charge."""
    sup, ks = stores["sup"], stores["ks"]
    a = _make_agent(ks)
    sup.spawn(pid=a.pid, argv=RUNNER_ARGV,
              policy=SandboxPolicy(wall_seconds=10))
    info = sup.wait(a.pid, timeout=10)
    assert info.exit_kind == "completed"
    # No ledger; nothing charged.
    assert info.ledger_charged.get("wall_s", 0) == 0


def test_runner_charge_message_translates_to_ledger(stores):
    """A runner that emits {op:'charge', dim, amount} must update the
    ledger."""
    sup, ks, led = stores["sup"], stores["ks"], stores["led"]
    a = _make_agent(ks)
    led.create(pid=a.pid, grants={"tool_calls": 100})
    sup.spawn(
        pid=a.pid, argv=RUNNER_ARGV,
        env={**os.environ, "CC_RUNNER_BEHAVIOR": "charge=tool_calls:7"},
        policy=SandboxPolicy(wall_seconds=10),
    )
    info = sup.wait(a.pid, timeout=10)
    assert info.exit_kind == "completed"
    assert info.ledger_charged.get("tool_calls", 0) == 7
    assert led.get(a.pid).entries[0].used == 7


def test_first_breach_records_event(stores):
    sup, ks, led = stores["sup"], stores["ks"], stores["led"]
    a = _make_agent(ks)
    led.create(pid=a.pid, grants={"tool_calls": 5})
    sup.spawn(
        pid=a.pid, argv=RUNNER_ARGV,
        env={**os.environ, "CC_RUNNER_BEHAVIOR": "charge=tool_calls:50"},
        policy=SandboxPolicy(wall_seconds=10),
    )
    info = sup.wait(a.pid, timeout=10)
    assert info.exit_kind == "completed"
    # The breach event must have been recorded.
    events = ks.events_tail(pid=a.pid,
                             kind="runner.first_breach")
    assert len(events) >= 1
    assert events[0].payload["dim"] == "tool_calls"


# ── list / cleanup ──────────────────────────────────────────────────────


def test_list_reports_active_handles(stores):
    sup, ks = stores["sup"], stores["ks"]
    a = _make_agent(ks, name="a")
    b = _make_agent(ks, name="b")
    sup.spawn(pid=a.pid, argv=RUNNER_ARGV,
              env={**os.environ, "CC_RUNNER_BEHAVIOR": "slow=2"},
              policy=SandboxPolicy(wall_seconds=10))
    sup.spawn(pid=b.pid, argv=RUNNER_ARGV,
              env={**os.environ, "CC_RUNNER_BEHAVIOR": "slow=2"},
              policy=SandboxPolicy(wall_seconds=10))
    handles = sup.list()
    pids = sorted(h.pid for h in handles)
    assert pids == sorted([a.pid, b.pid])
    sup.wait(a.pid, timeout=10)
    sup.wait(b.pid, timeout=10)
    assert sup.list() == []


# ── Concurrent spawns ───────────────────────────────────────────────────


def test_concurrent_spawn_distinct_agents(stores):
    """Two independent threads spawn + wait. No collision."""
    sup, ks = stores["sup"], stores["ks"]
    agents = [_make_agent(ks, name=f"a{i}") for i in range(4)]
    errors: list = []

    def worker(agent):
        try:
            sup.spawn(pid=agent.pid, argv=RUNNER_ARGV,
                      env={**os.environ,
                           "CC_RUNNER_BEHAVIOR": "slow=0.3"},
                      policy=SandboxPolicy(wall_seconds=10))
            info = sup.wait(agent.pid, timeout=15)
            assert info.exit_kind == "completed"
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(a,))
               for a in agents]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == []
    for a in agents:
        assert ks.get(a.pid).state == AgentState.DEAD


# ── Handshake failure ───────────────────────────────────────────────────


def test_handshake_timeout_raises(stores, tmp_path):
    """Spawn a script that doesn't honour the protocol — handshake
    times out."""
    sup, ks = stores["sup"], stores["ks"]
    a = _make_agent(ks)
    # A script that exits immediately without the protocol.
    bad_runner = tmp_path / "bad_runner.py"
    bad_runner.write_text("import sys; sys.exit(0)\n")
    with pytest.raises((RunnerHandshakeFailed, Exception)):
        sup.spawn(pid=a.pid,
                  argv=[sys.executable, str(bad_runner)],
                  policy=SandboxPolicy(wall_seconds=10))
    # Agent stayed in READY (we never transitioned it).
    assert ks.get(a.pid).state == AgentState.READY
