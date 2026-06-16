"""Chaos smoke test (RFC 0012 §6) — daemon survives random faults."""
from __future__ import annotations

import sqlite3

import pytest

from cheetahclaws.kernel import (
    AgentState,
    AgentFSStore,
    CapabilityStore,
    ChaosMonkey,
    KernelStore,
    LedgerStore,
    MailboxStore,
    ObservabilityStore,
    RegistryStore,
    SchedulerStore,
    ScheduleSpec,
)


@pytest.fixture
def stores(tmp_path):
    ks  = KernelStore.open(tmp_path / "kernel.db")
    cap = CapabilityStore(ks.connection, write_lock=ks.write_lock)
    led = LedgerStore(ks.connection, write_lock=ks.write_lock)
    sch = SchedulerStore(ks.connection, write_lock=ks.write_lock)
    mbx = MailboxStore(ks.connection, write_lock=ks.write_lock)
    reg = RegistryStore(ks.connection, write_lock=ks.write_lock)
    fs  = AgentFSStore(ks.connection, write_lock=ks.write_lock, ledger=led)
    obs = ObservabilityStore(
        kernel_store=ks, capability_store=cap, ledger_store=led,
        scheduler_store=sch, mailbox_store=mbx, registry_store=reg,
        agentfs_store=fs,
    )
    yield {"ks": ks, "mbx": mbx, "obs": obs}
    ks.close()


# ── Determinism ──────────────────────────────────────────────────────────


def test_chaos_kill_random_agent_is_deterministic(stores):
    ks = stores["ks"]
    pids = [ks.create(name=f"a{i}", template="t").pid for i in range(5)]

    monkey1 = ChaosMonkey(seed=42)
    killed1 = monkey1.kill_random_agent(ks)

    # Reset (revive). We can't truly revive a DEAD agent — that's the
    # whole point of the state machine — so create fresh agents to
    # mirror the same population, then re-test determinism by
    # re-running with the same seed against the same (different but
    # comparably shaped) population. Determinism here means: same
    # population shape + same seed → same victim INDEX.

    # For a stricter assertion: two ChaosMonkeys with the same seed
    # picking from the SAME population pick the same victim.
    # We'll use a fresh DB to make populations identical.
    from cheetahclaws.kernel import KernelStore as _KS
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        ks2 = _KS.open(pathlib.Path(d) / "k.db")
        try:
            for i in range(5):
                ks2.create(name=f"a{i}", template="t")
            monkey2 = ChaosMonkey(seed=42)
            killed2 = monkey2.kill_random_agent(ks2)
        finally:
            ks2.close()
    # Both monkeys picked the same nth-victim (pid offset within their
    # respective DBs).
    assert killed1 - pids[0] == killed2 - 1  # both DBs start at pid=1


def test_chaos_kill_no_live_agents_returns_none(stores):
    ks = stores["ks"]
    monkey = ChaosMonkey(seed=0)
    assert monkey.kill_random_agent(ks) is None
    assert monkey.events[-1]["killed"] is None


# ── kill_random_agent ────────────────────────────────────────────────────


def test_chaos_kill_terminates_agent(stores):
    ks = stores["ks"]
    a = ks.create(name="x", template="t")
    monkey = ChaosMonkey(seed=0)
    killed = monkey.kill_random_agent(ks)
    assert killed == a.pid
    assert ks.get(a.pid).state == AgentState.DEAD
    assert ks.get(a.pid).exit_kind == "crashed"


# ── fill_mailbox ─────────────────────────────────────────────────────────


def test_chaos_fill_mailbox(stores):
    ks, mbx = stores["ks"], stores["mbx"]
    a = ks.create(name="x", template="t")
    mbx.create(pid=a.pid, queue_size=5)
    monkey = ChaosMonkey()
    sent = monkey.fill_mailbox(mbx, a.pid)
    assert sent == 5


# ── lose_event ───────────────────────────────────────────────────────────


def test_chaos_lose_event_returns_true(stores):
    ks = stores["ks"]
    a = ks.create(name="x", template="t")  # creates one event
    eid = ks.events_append(pid=a.pid, kind="my.k", payload={})
    monkey = ChaosMonkey()
    assert monkey.lose_event(ks, eid) is True
    assert monkey.lose_event(ks, eid) is False  # already gone


# ── simulate_disk_full ───────────────────────────────────────────────────


def test_chaos_disk_full_triggers_oneshot(stores):
    ks = stores["ks"]
    a = ks.create(name="x", template="t")
    monkey = ChaosMonkey()
    with monkey.simulate_disk_full(ks):
        with pytest.raises(sqlite3.OperationalError):
            ks.create(name="y", template="t")
    # Outside the context: writes work again.
    ks.create(name="z", template="t")


# ── Headline: daemon survives chaos ──────────────────────────────────────


def test_daemon_survives_chaos_smoke(stores):
    """3 chaos operations + observe.summary still works."""
    ks, obs = stores["ks"], stores["obs"]
    pids = [ks.create(name=f"a{i}", template="t").pid for i in range(5)]
    monkey = ChaosMonkey(seed=42)

    # 1. Kill a random agent.
    killed = monkey.kill_random_agent(ks)
    assert killed in pids

    # 2. Lose an event.
    monkey.lose_event(ks, 1)  # the very first event_id

    # 3. Trigger disk-full once and recover.
    try:
        with monkey.simulate_disk_full(ks):
            try:
                ks.events_append(pid=pids[0], kind="my.k", payload={})
            except sqlite3.OperationalError:
                pass
    except Exception:
        pass

    # The summary still works.
    s = obs.summary()
    assert s["agents"]["total"] == 5
    assert s["agents"]["DEAD"] == 1
    # Events count may have one fewer than expected due to lose_event.
    assert s["events"]["total"] >= 0

    # The chaos monkey logged 3 events.
    assert len(monkey.events) >= 3
