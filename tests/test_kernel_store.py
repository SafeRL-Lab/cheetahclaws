"""Tests for kernel.store.KernelStore (CRUD, events, info).

Uses tmp_path-scoped kernel.db files; no daemon, no network.
"""
from __future__ import annotations

import json
import sqlite3
import time

import pytest

from cheetahclaws.kernel import (
    AgentProcess,
    AgentState,
    IllegalTransition,
    KernelStore,
    UnknownPid,
)
from cheetahclaws.kernel.errors import (
    KERNEL_ILLEGAL_TRANSITION,
    KERNEL_UNKNOWN_PID,
    InvalidPayload,
)
from cheetahclaws.kernel.schema import (
    EXPECTED_SCHEMA_VERSION,
    get_schema_version,
    open_connection,
)
from cheetahclaws.kernel.store import (
    EV_PROCESS_CREATED,
    EV_PROCESS_RECOVERED,
    EV_PROCESS_TERMINATED,
    EV_PROCESS_TRANSITIONED,
)


@pytest.fixture
def store(tmp_path):
    s = KernelStore.open(tmp_path / "kernel.db")
    yield s
    s.close()


# ── Schema + open ──────────────────────────────────────────────────────────


def test_open_creates_kernel_db_and_schema(tmp_path):
    db = tmp_path / "kernel.db"
    assert not db.exists()
    s = KernelStore.open(db)
    try:
        assert db.exists()
        # Verify schema_version
        conn = sqlite3.connect(str(db))
        try:
            row = conn.execute(
                "SELECT value FROM kernel_meta WHERE key='schema_version'"
            ).fetchone()
            assert int(row[0]) == EXPECTED_SCHEMA_VERSION
        finally:
            conn.close()
    finally:
        s.close()


def test_open_is_idempotent(tmp_path):
    s1 = KernelStore.open(tmp_path / "kernel.db")
    s1.create(name="a", template="t")
    s1.close()
    s2 = KernelStore.open(tmp_path / "kernel.db")
    try:
        agents, total = s2.list()
        assert total == 1
        assert agents[0].name == "a"
    finally:
        s2.close()


def test_get_schema_version_helper(tmp_path):
    s = KernelStore.open(tmp_path / "kernel.db")
    try:
        # Open a separate read-only connection — the helper supports it.
        conn = open_connection(str(tmp_path / "kernel.db"))
        try:
            assert get_schema_version(conn) == EXPECTED_SCHEMA_VERSION
        finally:
            conn.close()
    finally:
        s.close()


# ── Create + get + list ────────────────────────────────────────────────────


def test_create_returns_ready_agent_with_pid(store):
    a = store.create(name="alice", template="research/surveyor")
    assert isinstance(a, AgentProcess)
    assert a.pid >= 1
    assert a.parent_pid is None
    assert a.name == "alice"
    assert a.template == "research/surveyor"
    assert a.state == AgentState.READY
    assert a.exit_kind is None


def test_create_persists(store):
    a = store.create(name="bob", template="t")
    fetched = store.get(a.pid)
    assert fetched.pid == a.pid
    assert fetched.name == "bob"


def test_create_with_metadata(store):
    meta = {"k": "v", "n": 42}
    a = store.create(name="c", template="t", metadata=meta)
    assert store.get(a.pid).metadata == meta


def test_create_with_parent_pid_validates(store):
    parent = store.create(name="p", template="t")
    child = store.create(name="ch", template="t", parent_pid=parent.pid)
    assert child.parent_pid == parent.pid


def test_create_with_unknown_parent_raises(store):
    with pytest.raises(UnknownPid) as e:
        store.create(name="ch", template="t", parent_pid=9999)
    assert e.value.code == KERNEL_UNKNOWN_PID
    assert e.value.pid == 9999


def test_create_rejects_empty_name(store):
    with pytest.raises(InvalidPayload):
        store.create(name="", template="t")


def test_create_rejects_non_string_template(store):
    with pytest.raises(InvalidPayload):
        store.create(name="a", template=42)  # type: ignore[arg-type]


def test_get_unknown_pid_raises(store):
    with pytest.raises(UnknownPid):
        store.get(9999)


def test_pids_are_monotonic(store):
    pids = [store.create(name=f"a{i}", template="t").pid for i in range(5)]
    assert pids == sorted(pids)
    assert len(set(pids)) == 5


def test_list_filters(store):
    a1 = store.create(name="a1", template="t")
    a2 = store.create(name="a2", template="t")
    store.transition(a1.pid, AgentState.RUNNING)
    agents, total = store.list(state=AgentState.RUNNING)
    assert total == 1
    assert agents[0].pid == a1.pid

    agents, total = store.list(state=AgentState.READY)
    assert total == 1
    assert agents[0].pid == a2.pid

    # Filter by parent_pid
    child = store.create(name="ch", template="t", parent_pid=a1.pid)
    agents, total = store.list(parent_pid=a1.pid)
    assert total == 1
    assert agents[0].pid == child.pid


def test_list_pagination(store):
    for i in range(5):
        store.create(name=f"a{i}", template="t")
    page1, total = store.list(limit=2, offset=0)
    page2, _ = store.list(limit=2, offset=2)
    page3, _ = store.list(limit=2, offset=4)
    assert total == 5
    assert len(page1) == 2 and len(page2) == 2 and len(page3) == 1
    pids = [a.pid for a in page1 + page2 + page3]
    assert pids == sorted(pids)


# ── Transition ─────────────────────────────────────────────────────────────


def test_transition_legal_path(store):
    a = store.create(name="x", template="t")
    prev, new, eid1 = store.transition(a.pid, AgentState.RUNNING)
    assert (prev, new) == (AgentState.READY, AgentState.RUNNING)
    assert eid1 > 0

    a2 = store.get(a.pid)
    assert a2.state == AgentState.RUNNING
    assert a2.started_at is not None

    prev, new, eid2 = store.transition(a.pid, AgentState.WAITING, reason="tool_call")
    assert (prev, new) == (AgentState.RUNNING, AgentState.WAITING)
    assert eid2 > eid1

    a3 = store.get(a.pid)
    assert a3.state == AgentState.WAITING
    assert a3.state_reason == "tool_call"


def test_transition_illegal_raises(store):
    a = store.create(name="x", template="t")
    # READY -> WAITING is illegal (must go through RUNNING).
    with pytest.raises(IllegalTransition) as e:
        store.transition(a.pid, AgentState.WAITING)
    assert e.value.code == KERNEL_ILLEGAL_TRANSITION
    assert e.value.prev_state == AgentState.READY
    assert e.value.target_state == AgentState.WAITING

    # Row should be untouched.
    assert store.get(a.pid).state == AgentState.READY


def test_transition_unknown_pid(store):
    with pytest.raises(UnknownPid):
        store.transition(9999, AgentState.RUNNING)


def test_transition_to_invalid_state(store):
    a = store.create(name="x", template="t")
    with pytest.raises(InvalidPayload):
        store.transition(a.pid, "BOGUS")


def test_dead_is_terminal(store):
    a = store.create(name="x", template="t")
    store.terminate(a.pid, exit_kind="cancelled")
    # Any further transition is illegal.
    for tgt in (AgentState.READY, AgentState.RUNNING, AgentState.WAITING,
                AgentState.SUSPENDED):
        with pytest.raises(IllegalTransition):
            store.transition(a.pid, tgt)
    # And a second terminate is also illegal (DEAD -> DEAD).
    with pytest.raises(IllegalTransition):
        store.terminate(a.pid, exit_kind="cancelled")


# ── Terminate ──────────────────────────────────────────────────────────────


def test_terminate_records_exit_kind(store):
    a = store.create(name="x", template="t")
    store.transition(a.pid, AgentState.RUNNING)
    prev, eid = store.terminate(a.pid, exit_kind="failed",
                                 exit_detail={"err": "oom"})
    assert prev == AgentState.RUNNING
    assert eid > 0
    a2 = store.get(a.pid)
    assert a2.state == AgentState.DEAD
    assert a2.exit_kind == "failed"
    assert a2.exit_detail == {"err": "oom"}
    assert a2.ended_at is not None


def test_terminate_invalid_exit_kind(store):
    a = store.create(name="x", template="t")
    with pytest.raises(InvalidPayload):
        store.terminate(a.pid, exit_kind="bogus")


# ── Events ─────────────────────────────────────────────────────────────────


def test_create_emits_one_event(store):
    a = store.create(name="x", template="t")
    events = store.events_tail(pid=a.pid)
    assert len(events) == 1
    e = events[0]
    assert e.kind == EV_PROCESS_CREATED
    assert e.pid == a.pid
    assert e.payload["name"] == "x"
    assert e.payload["template"] == "t"


def test_transition_emits_event(store):
    a = store.create(name="x", template="t")
    store.transition(a.pid, AgentState.RUNNING, reason="manual_start")
    events = store.events_tail(pid=a.pid)
    assert len(events) == 2
    assert events[1].kind == EV_PROCESS_TRANSITIONED
    assert events[1].payload == {
        "prev_state": AgentState.READY,
        "new_state": AgentState.RUNNING,
        "reason": "manual_start",
    }


def test_terminate_emits_terminated_event(store):
    a = store.create(name="x", template="t")
    store.transition(a.pid, AgentState.RUNNING)
    store.terminate(a.pid, exit_kind="completed")
    events = store.events_tail(pid=a.pid)
    kinds = [e.kind for e in events]
    assert kinds == [EV_PROCESS_CREATED, EV_PROCESS_TRANSITIONED,
                     EV_PROCESS_TERMINATED]


def test_event_ids_are_monotonic_across_agents(store):
    a1 = store.create(name="a1", template="t")
    a2 = store.create(name="a2", template="t")
    store.transition(a1.pid, AgentState.RUNNING)
    store.transition(a2.pid, AgentState.RUNNING)

    all_events = store.events_tail(limit=100)
    ids = [e.event_id for e in all_events]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


def test_events_tail_filter_by_kind(store):
    a = store.create(name="x", template="t")
    store.transition(a.pid, AgentState.RUNNING)
    only_created = store.events_tail(kind=EV_PROCESS_CREATED)
    assert len(only_created) == 1
    assert only_created[0].kind == EV_PROCESS_CREATED


def test_events_tail_since_cursor(store):
    a = store.create(name="x", template="t")  # event 1 (created)
    store.transition(a.pid, AgentState.RUNNING)  # event 2

    first = store.events_tail(pid=a.pid, limit=1)
    assert len(first) == 1
    cursor = first[-1].event_id
    rest = store.events_tail(pid=a.pid, since_event_id=cursor)
    assert len(rest) == 1
    assert rest[0].event_id > cursor


def test_events_append_user_event(store):
    a = store.create(name="x", template="t")
    eid = store.events_append(pid=a.pid, kind="my.app.foo",
                               payload={"detail": 1})
    events = store.events_tail(pid=a.pid)
    assert any(e.event_id == eid and e.kind == "my.app.foo" for e in events)
    assert store.get(a.pid).last_event_id == eid


def test_events_append_rejects_kernel_prefix(store):
    a = store.create(name="x", template="t")
    with pytest.raises(InvalidPayload):
        store.events_append(pid=a.pid, kind="kernel.something",
                            payload={})


def test_events_append_unknown_pid(store):
    with pytest.raises(UnknownPid):
        store.events_append(pid=9999, kind="my.x", payload={})


def test_events_append_with_causation_and_correlation(store):
    a = store.create(name="x", template="t")
    e1 = store.events_append(pid=a.pid, kind="my.x", payload={"step": 1})
    e2 = store.events_append(pid=a.pid, kind="my.x",
                             payload={"step": 2},
                             causation_id=e1, correlation_id="trace-123")
    events = store.events_tail(pid=a.pid, kind="my.x")
    assert len(events) == 2
    assert events[1].causation_id == e1
    assert events[1].correlation_id == "trace-123"


# ── Bus integration ────────────────────────────────────────────────────────


class _FakeBus:
    def __init__(self):
        self.published: list[tuple[str, dict]] = []

    def publish(self, ev_type: str, data: dict) -> int:
        self.published.append((ev_type, data))
        return len(self.published)


def test_bus_receives_kernel_events_after_commit(tmp_path):
    bus = _FakeBus()
    store = KernelStore.open(tmp_path / "kernel.db", bus=bus)
    try:
        a = store.create(name="x", template="t")
        store.transition(a.pid, AgentState.RUNNING)
        store.terminate(a.pid, exit_kind="completed")
        kinds = [k for (k, _d) in bus.published]
        assert kinds == [
            EV_PROCESS_CREATED,
            EV_PROCESS_TRANSITIONED,
            EV_PROCESS_TERMINATED,
        ]
    finally:
        store.close()


# ── Info ───────────────────────────────────────────────────────────────────


def test_info_reports_counts(store):
    assert store.info() == {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "next_pid": 1,
        "next_event_id": 1,
        "agent_count": 0,
        "event_count": 0,
        "live_states": {s: 0 for s in (AgentState.READY, AgentState.RUNNING,
                                       AgentState.WAITING, AgentState.SUSPENDED)},
    }
    a = store.create(name="x", template="t")
    store.transition(a.pid, AgentState.RUNNING)
    info = store.info()
    assert info["agent_count"] == 1
    assert info["event_count"] == 2
    assert info["live_states"][AgentState.RUNNING] == 1
    assert info["next_pid"] >= 2
    assert info["next_event_id"] >= 3


# ── Concurrency smoke test ─────────────────────────────────────────────────
#
# Not a stress test — just confirms that two writer threads don't crash
# the lock or corrupt the table. Real fuzz/chaos testing belongs in
# RFC 0012 (Observability + chaos suite).


def test_concurrent_writers_do_not_corrupt(store):
    import threading

    a = store.create(name="x", template="t")
    store.transition(a.pid, AgentState.RUNNING)
    errors: list[Exception] = []

    def worker(n: int):
        try:
            for i in range(n):
                store.events_append(pid=a.pid, kind="my.t",
                                    payload={"i": i})
        except Exception as e:  # noqa: BLE001 — caller asserts later
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(20,)) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == []
    events = store.events_tail(pid=a.pid, kind="my.t", limit=1000)
    assert len(events) == 80
    # Event ids are unique and monotonic.
    eids = [e.event_id for e in events]
    assert eids == sorted(eids)
    assert len(set(eids)) == len(eids)
