"""Tests for kernel.observability (RFC 0012)."""
from __future__ import annotations

import re
import time

import pytest

from cheetahclaws.kernel import (
    AgentFSStore,
    AgentState,
    CapabilityStore,
    KernelStore,
    LedgerStore,
    MailboxStore,
    ObservabilityStore,
    RegistryStore,
    SchedulerStore,
    ScheduleSpec,
)
from cheetahclaws.kernel.errors import InvalidPayload, UnknownPid


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
    yield {
        "ks": ks, "cap": cap, "led": led, "sch": sch,
        "mbx": mbx, "reg": reg, "fs": fs, "obs": obs,
    }
    ks.close()


# ── proc(pid) ────────────────────────────────────────────────────────────


def test_proc_returns_full_shape(stores):
    obs = stores["obs"]
    a = stores["ks"].create(name="alice", template="research")
    proc = obs.proc(a.pid)
    assert "process" in proc and proc["process"]["pid"] == a.pid
    # Sub-fields are present even when empty.
    assert proc["capability"] is None
    assert proc["ledger"] == []
    assert proc["mailbox"]["exists"] is False
    assert proc["scheduler"] == {
        "queued": 0, "dispatched": 0, "completed": 0,
        "expired": 0, "cancelled": 0,
    }
    assert proc["fs"] == {"object_count": 0, "total_bytes": 0}
    assert proc["registry"] == {"names": []}
    assert isinstance(proc["recent_events"], list)


def test_proc_aggregates_across_stores(stores):
    obs, ks = stores["obs"], stores["ks"]
    a = ks.create(name="alice", template="t")

    # Populate state across every store.
    stores["cap"].create(pid=a.pid, tool_grants=["Read"])
    stores["led"].create(pid=a.pid, grants={"tokens": 1000})
    stores["led"].charge(pid=a.pid, dim="tokens", amount=300)
    stores["sch"].enqueue(ScheduleSpec(pid=a.pid))
    stores["mbx"].create(pid=a.pid)
    stores["mbx"].subscribe(a.pid, "alerts")
    stores["mbx"].send(sender_pid=None, recipient_pid=a.pid,
                        kind="hello", payload={})
    stores["reg"].register(name="/agents/alice", pid=a.pid,
                            tags=["research"])
    stores["fs"].write(pid=a.pid, path="/memory/alice/note",
                        content=b"hi")

    proc = obs.proc(a.pid)
    assert proc["capability"]["tool_grants"] == ["Read"]
    assert proc["ledger"][0]["used"] == 300
    assert proc["scheduler"]["queued"] == 1
    assert proc["mailbox"]["exists"] is True
    assert proc["mailbox"]["pending"] == 1
    assert proc["mailbox"]["subscriptions"] == ["alerts"]
    assert proc["fs"]["object_count"] == 1
    assert proc["registry"]["names"] == ["/agents/alice"]
    assert len(proc["recent_events"]) >= 1


def test_proc_unknown_pid_returns_no_process(stores):
    obs = stores["obs"]
    proc = obs.proc(9999)
    assert proc["process"] is None
    assert proc["capability"] is None
    assert proc["ledger"] == []


def test_proc_invalid_pid_raises(stores):
    obs = stores["obs"]
    with pytest.raises(InvalidPayload):
        obs.proc("not-an-int")  # type: ignore[arg-type]


def test_proc_recent_events_chronological(stores):
    obs, ks = stores["obs"], stores["ks"]
    a = ks.create(name="x", template="t")
    ks.transition(a.pid, AgentState.RUNNING)
    ks.terminate(a.pid, exit_kind="completed")
    proc = obs.proc(a.pid)
    kinds = [e["kind"] for e in proc["recent_events"]]
    # Created → transitioned → terminated, oldest-first.
    assert kinds == [
        "kernel.process.created",
        "kernel.process.transitioned",
        "kernel.process.terminated",
    ]


# ── summary() ────────────────────────────────────────────────────────────


def test_summary_empty(stores):
    summary = stores["obs"].summary()
    assert summary["schema_version"] == 5
    assert summary["agents"]["total"] == 0
    assert summary["events"]["total"] == 0
    assert summary["scheduler"]["queued"] == 0


def test_summary_counts_after_population(stores):
    obs, ks = stores["obs"], stores["ks"]
    a = ks.create(name="a", template="t")
    b = ks.create(name="b", template="t")
    ks.transition(a.pid, AgentState.RUNNING)
    ks.terminate(b.pid, exit_kind="completed")

    stores["led"].create(pid=a.pid, grants={"tokens": 100})
    stores["led"].charge(pid=a.pid, dim="tokens", amount=200)  # breach

    stores["sch"].enqueue(ScheduleSpec(pid=a.pid))
    stores["mbx"].create(pid=a.pid)
    stores["mbx"].send(sender_pid=None, recipient_pid=a.pid,
                        kind="k", payload={})
    stores["fs"].write(pid=a.pid, path="/x", content=b"hello")
    stores["reg"].register(name="/agents/a", pid=a.pid)

    s = obs.summary()
    assert s["agents"]["total"] == 2
    assert s["agents"]["RUNNING"] == 1
    assert s["agents"]["DEAD"]    == 1
    assert s["events"]["total"] >= 3   # created x2 + transitions
    assert s["scheduler"]["queued"] == 1
    assert s["mailbox"]["mailboxes"] == 1
    assert s["mailbox"]["pending_messages"] == 1
    assert s["ledger"]["agents_with_budgets"] == 1
    assert s["ledger"]["breached"]            == 1
    assert s["fs"]["objects"] == 1
    assert s["fs"]["total_bytes"] == 5
    assert s["registry"]["entries"] == 1


def test_summary_uptime_increases(stores):
    s1 = stores["obs"].summary()["uptime_s"]
    time.sleep(0.05)
    s2 = stores["obs"].summary()["uptime_s"]
    assert s2 > s1


# ── trace() ──────────────────────────────────────────────────────────────


def test_trace_walks_causation_chain(stores):
    obs, ks = stores["obs"], stores["ks"]
    a = ks.create(name="x", template="t")
    # Build a manual causation chain via events_append.
    e1 = ks.events_append(pid=a.pid, kind="my.start",   payload={"step": 1})
    e2 = ks.events_append(pid=a.pid, kind="my.middle",  payload={"step": 2},
                           causation_id=e1)
    e3 = ks.events_append(pid=a.pid, kind="my.end",     payload={"step": 3},
                           causation_id=e2)

    result = obs.trace(e3, depth=10)
    ids = [e["event_id"] for e in result["events"]]
    # Walk goes downstream → upstream: start at e3, follow causation.
    assert ids == [e3, e2, e1]
    assert result["depth"] == 3
    assert result["truncated"] is False


def test_trace_depth_truncates(stores):
    obs, ks = stores["obs"], stores["ks"]
    a = ks.create(name="x", template="t")
    last = ks.events_append(pid=a.pid, kind="my.k", payload={})
    for i in range(5):
        last = ks.events_append(pid=a.pid, kind="my.k",
                                payload={"i": i}, causation_id=last)
    result = obs.trace(last, depth=2)
    assert result["depth"] == 2
    assert result["truncated"] is True


def test_trace_unknown_event_truncates(stores):
    obs = stores["obs"]
    result = obs.trace(99999, depth=10)
    # No event found at start — chain is empty, marked truncated.
    assert result["events"] == []
    assert result["truncated"] is True


def test_trace_invalid_event_id(stores):
    obs = stores["obs"]
    with pytest.raises(InvalidPayload):
        obs.trace("not-int")  # type: ignore[arg-type]


# ── prometheus_text() ────────────────────────────────────────────────────


def test_prometheus_text_format(stores):
    obs, ks = stores["obs"], stores["ks"]
    a = ks.create(name="x", template="t")
    ks.transition(a.pid, AgentState.RUNNING)
    text = obs.prometheus_text()

    # Required header for each metric.
    assert "# HELP cheetahclaws_kernel_schema_version" in text
    assert "# TYPE cheetahclaws_kernel_schema_version gauge" in text

    # Metric values present.
    assert "cheetahclaws_kernel_schema_version 5" in text
    assert 'cheetahclaws_kernel_agents{state="RUNNING"} 1' in text
    assert 'cheetahclaws_kernel_agents{state="READY"} 0' in text

    # Final newline.
    assert text.endswith("\n")


def test_prometheus_text_passes_basic_regex_contract(stores):
    """Each metric line: metric_name{label_pairs} value
    Where labels are double-quoted, values are numeric."""
    text = stores["obs"].prometheus_text()
    line_pattern = re.compile(
        r"^[a-zA-Z_][a-zA-Z0-9_]*"            # metric name
        r"(\{[^}]*\})?"                        # optional labels
        r"\s+-?\d+(\.\d+)?\s*$"                # numeric value
    )
    for line in text.split("\n"):
        if not line or line.startswith("#"):
            continue
        assert line_pattern.match(line), f"bad metric line: {line!r}"


def test_prometheus_text_label_escaping():
    """Make sure label-value escaping handles backslashes and quotes."""
    from cheetahclaws.kernel.observability import _esc_label
    assert _esc_label('plain') == 'plain'
    assert _esc_label('with"quote') == 'with\\"quote'
    assert _esc_label('with\\back') == 'with\\\\back'
    assert _esc_label('with\nnewline') == 'with\\nnewline'
