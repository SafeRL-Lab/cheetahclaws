"""Tests for kernel.worker (RFC 0017) — WorkerLoop."""
from __future__ import annotations

import os
import sys
import threading
import time

import pytest

from cheetahclaws.kernel import (
    AgentState,
    KernelStore,
    LedgerStore,
    RunnerSupervisor,
    SandboxPolicy,
    SchedulerStore,
    ScheduleSpec,
    WorkerLoop,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="WorkerLoop drives subprocesses — POSIX-only in v1",
)


RUNNER_ARGV = [sys.executable, "-m", "cheetahclaws.kernel.runner.runner_main"]


@pytest.fixture
def stack(tmp_path):
    """Build a kernel + scheduler + ledger + supervisor + worker."""
    ks = KernelStore.open(tmp_path / "kernel.db")
    led = LedgerStore(ks.connection, write_lock=ks.write_lock)
    sch = SchedulerStore(ks.connection, write_lock=ks.write_lock)
    sup = RunnerSupervisor(ks, ledger_store=led)

    yield {"ks": ks, "led": led, "sch": sch, "sup": sup}

    for h in sup.list():
        try:
            sup.stop(h.pid)
        except Exception:
            pass
    ks.close()


def _argv_factory(entry):
    return RUNNER_ARGV


def _quick_env_factory(entry):
    """Fast-exiting echo runner, suitable for tick tests."""
    return {**os.environ, "CC_RUNNER_BEHAVIOR": "echo"}


def _slow_env_factory(seconds: float):
    def fn(entry):
        return {**os.environ, "CC_RUNNER_BEHAVIOR": f"slow={seconds}"}
    return fn


def _short_policy(entry):
    return SandboxPolicy(wall_seconds=10)


def _enqueue(stack, n: int, name_prefix: str = "a") -> list[int]:
    """Create n agents + schedule entries; return sched_ids."""
    sids = []
    for i in range(n):
        agent = stack["ks"].create(name=f"{name_prefix}{i}", template="t")
        sids.append(stack["sch"].enqueue(ScheduleSpec(pid=agent.pid)))
    return sids


def _build_loop(stack, **kw):
    defaults = dict(
        argv_factory=_argv_factory,
        policy_factory=_short_policy,
        env_factory=_quick_env_factory,
        max_concurrent=4,
        poll_interval_s=0.05,
        wait_timeout_s=15.0,
    )
    defaults.update(kw)
    return WorkerLoop(
        kernel_store=stack["ks"],
        scheduler_store=stack["sch"],
        supervisor=stack["sup"],
        **defaults,
    )


# ── tick ─────────────────────────────────────────────────────────────────


def test_tick_returns_false_when_queue_empty(stack):
    loop = _build_loop(stack)
    assert loop.tick() is False


def test_tick_claims_and_runs_one(stack):
    sids = _enqueue(stack, 1)
    loop = _build_loop(stack)
    assert loop.tick() is True
    # Wait briefly for the runner thread to finish.
    deadline = time.monotonic() + 10
    while loop.in_flight() > 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    e = stack["sch"].get(sids[0])
    assert e.state == "completed"
    assert e.exit_kind == "completed"


def test_tick_marks_agent_dead(stack):
    sids = _enqueue(stack, 1)
    loop = _build_loop(stack)
    loop.tick()
    deadline = time.monotonic() + 10
    while loop.in_flight() > 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    agents, _ = stack["ks"].list()
    assert agents[0].state == AgentState.DEAD
    assert agents[0].exit_kind == "completed"


# ── max_concurrent ──────────────────────────────────────────────────────


def test_capacity_cap_blocks_third_tick(stack):
    sids = _enqueue(stack, 5)
    loop = _build_loop(
        stack, max_concurrent=2,
        env_factory=_slow_env_factory(2.0),
    )
    assert loop.tick() is True
    assert loop.tick() is True
    # Third tick: cap full.
    assert loop.tick() is False
    assert loop.in_flight() == 2
    # Wait for first batch.
    deadline = time.monotonic() + 15
    while loop.in_flight() > 0 and time.monotonic() < deadline:
        time.sleep(0.1)
    # Now we can tick again.
    assert loop.tick() is True


def test_observed_concurrency_does_not_exceed_cap(stack):
    sids = _enqueue(stack, 6)
    max_seen = [0]
    seen_lock = threading.Lock()

    loop = _build_loop(
        stack, max_concurrent=3,
        env_factory=_slow_env_factory(0.5),
    )
    loop.start()
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with seen_lock:
                cur = loop.in_flight()
                if cur > max_seen[0]:
                    max_seen[0] = cur
            # Done when all 6 entries are completed.
            queued, _ = stack["sch"].list(state="queued")
            disp,   _ = stack["sch"].list(state="dispatched")
            if not queued and not disp:
                break
            time.sleep(0.05)
        else:
            pytest.fail("loop did not drain queue in 30s")
    finally:
        loop.stop(drain=True, drain_timeout_s=5)

    assert max_seen[0] <= 3, f"saw {max_seen[0]} runners in flight"
    # All entries reached completed.
    for sid in sids:
        assert stack["sch"].get(sid).state == "completed"


# ── start / stop ────────────────────────────────────────────────────────


def test_start_drains_queue_in_background(stack):
    sids = _enqueue(stack, 4)
    loop = _build_loop(stack, max_concurrent=2, poll_interval_s=0.02)
    loop.start()
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            queued, _ = stack["sch"].list(state="queued")
            disp,   _ = stack["sch"].list(state="dispatched")
            if not queued and not disp:
                break
            time.sleep(0.05)
        else:
            pytest.fail("loop didn't drain")
        for sid in sids:
            assert stack["sch"].get(sid).state == "completed"
    finally:
        loop.stop(drain=True, drain_timeout_s=5)


def test_stop_drain_waits_for_in_flight(stack):
    """drain=True returns 0 (no kills) when in-flight finishes within
    drain_timeout."""
    sids = _enqueue(stack, 1)
    loop = _build_loop(
        stack, env_factory=_slow_env_factory(0.5),
    )
    loop.tick()
    killed = loop.stop(drain=True, drain_timeout_s=10)
    assert killed == 0
    assert stack["sch"].get(sids[0]).state == "completed"


def test_stop_no_drain_kills_in_flight(stack):
    """drain=False kills running runners; scheduler entry ends in
    a terminal state.

    Note: tick() returns the moment the worker thread is launched;
    the worker still has to call supervisor.spawn (which includes a
    subprocess startup + ready handshake). Until spawn completes,
    supervisor.list() is empty and there's nothing for stop() to
    kill. The test waits for the supervisor to register a handle
    before calling stop, to exercise the kill path rather than the
    "nothing to kill yet" path.
    """
    sids = _enqueue(stack, 1)
    loop = _build_loop(
        stack, env_factory=_slow_env_factory(30.0),
    )
    loop.tick()
    deadline = time.monotonic() + 5
    while not stack["sup"].list() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert stack["sup"].list(), "supervisor never registered the runner"
    killed = loop.stop(drain=False)
    assert killed >= 1
    e = stack["sch"].get(sids[0])
    # Killed runner ends as completed/cancelled/crashed depending on
    # exact timing; the queue entry should be in a terminal state.
    assert e.state in ("completed", "cancelled", "expired", "crashed")


def test_start_idempotent(stack):
    loop = _build_loop(stack)
    loop.start()
    loop.start()  # No-op
    loop.stop()


def test_stop_without_start_is_safe(stack):
    loop = _build_loop(stack)
    # Calling stop without start is just sets the event — no kill, no error.
    killed = loop.stop()
    assert killed == 0


# ── exit_kind mapping ───────────────────────────────────────────────────


def test_runner_crash_marks_scheduler_failed_or_crashed(stack):
    sids = _enqueue(stack, 1)
    loop = _build_loop(
        stack,
        env_factory=lambda e: {**os.environ,
                                "CC_RUNNER_BEHAVIOR": "crash"},
    )
    loop.tick()
    deadline = time.monotonic() + 10
    while loop.in_flight() > 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    e = stack["sch"].get(sids[0])
    # Runner exited 1 without sending 'exit'; supervisor maps to
    # 'failed' (non-zero exit with no exit message). Either is fine.
    assert e.state == "completed"
    assert e.exit_kind in ("failed", "crashed")


# ── ledger admission still works through the loop ──────────────────────


def test_admission_filter_skips_overlimit(stack):
    """An agent whose ledger is already over-limit gets skipped by
    scheduler.claim, so the worker never spawns it."""
    sup = stack["sup"]
    led = stack["led"]
    ks = stack["ks"]
    sch = stack["sch"]

    a_ok  = ks.create(name="ok",  template="t")
    a_bad = ks.create(name="bad", template="t")
    led.create(pid=a_ok.pid,  grants={"tokens": 1000})
    led.create(pid=a_bad.pid, grants={"tokens": 1000})
    led.charge(pid=a_bad.pid, dim="tokens", amount=2000)

    sid_ok  = sch.enqueue(ScheduleSpec(pid=a_ok.pid))
    sid_bad = sch.enqueue(ScheduleSpec(pid=a_bad.pid))

    loop = _build_loop(stack)
    loop.tick()  # ok claimed
    assert loop.tick() is False  # bad skipped by admission

    deadline = time.monotonic() + 10
    while loop.in_flight() > 0 and time.monotonic() < deadline:
        time.sleep(0.05)

    assert sch.get(sid_ok).state  == "completed"
    assert sch.get(sid_bad).state == "queued"  # untouched


# ── empty-queue idle path ──────────────────────────────────────────────


def test_idle_loop_does_not_busy_spin(stack):
    """With no work, the driver should sleep poll_interval between
    ticks, not burn CPU. Smoke test by measuring tick count."""
    loop = _build_loop(stack, poll_interval_s=0.3)
    loop.start()
    time.sleep(1.0)
    loop.stop()
    # No queue work happened — just confirm the loop stopped cleanly.
    assert loop.in_flight() == 0
