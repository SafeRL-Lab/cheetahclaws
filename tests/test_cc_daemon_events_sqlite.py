"""F-2 tests for cc_daemon/events.py SQLite-backed bus.

The spike-era ring-buffer tests live in test_daemon_spike.py; this file
exercises the new persistence + retention + cross-restart behaviour.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cc_daemon import events, schema


@pytest.fixture(autouse=True)
def _per_test_db(tmp_path: Path):
    """Each test gets a private sessions.db so daemon_events doesn't bleed
    across tests (the spike's in-memory ring auto-isolated; SQLite doesn't)."""
    schema.set_db_path(tmp_path / "sessions.db")
    schema._local.conn = None
    events.reset_bus_for_tests()
    yield
    schema._local.conn = None
    schema.set_db_path(schema.get_default_db_path())  # restore default


# ── Persistence: events written to daemon_events ──────────────────────────

def test_publish_persists_to_daemon_events_table():
    bus = events.EventBus()
    eid = bus.publish("text_chunk", {"text": "hello"})
    conn = schema.get_conn()
    row = conn.execute(
        "SELECT id, kind, payload_json FROM daemon_events WHERE id=?",
        (eid,),
    ).fetchone()
    assert row is not None
    assert row[1] == "text_chunk"
    assert "hello" in row[2]


def test_publish_returns_monotonic_ids():
    bus = events.EventBus()
    a = bus.publish("k", {"i": 1})
    b = bus.publish("k", {"i": 2})
    assert b == a + 1


def test_replay_since_reads_from_sqlite():
    bus = events.EventBus()
    bus.publish("a", {"n": 0})
    bus.publish("a", {"n": 1})
    bus.publish("a", {"n": 2})
    out = list(bus.replay_since(1))
    # No gap, ids 2 and 3
    assert all(e["type"] != "gap" for e in out)
    assert [e["data"]["n"] for e in out] == [1, 2]


def test_replay_zero_includes_all():
    bus = events.EventBus()
    bus.publish("k", {"n": 1})
    bus.publish("k", {"n": 2})
    out = list(bus.replay_since(0))
    assert [e["data"]["n"] for e in out] == [1, 2]


def test_originator_round_trips_through_sqlite():
    bus = events.EventBus()
    bus.publish("permission_request", {"tool": "Bash"},
                originator={"client_id": "abc", "client_kind": "repl"})
    out = list(bus.replay_since(0))
    assert out[0]["originator"] == {"client_id": "abc", "client_kind": "repl"}


# ── Retention: prune by row count ─────────────────────────────────────────

def test_retention_rows_prunes_oldest():
    bus = events.EventBus(retention_rows=3, prune_every_n=1)
    for i in range(10):
        bus.publish("k", {"i": i})
    conn = schema.get_conn()
    count = conn.execute("SELECT COUNT(*) FROM daemon_events").fetchone()[0]
    assert count == 3
    # The surviving rows are the newest 3 (ids 8, 9, 10)
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM daemon_events ORDER BY id"
    ).fetchall()]
    assert ids == [8, 9, 10]


def test_retention_hours_prunes_old_rows(monkeypatch):
    bus = events.EventBus(retention_hours=0.0001, prune_every_n=1)  # ~360 ms
    bus.publish("k", {"i": 1})
    time.sleep(0.5)
    bus.publish("k", {"i": 2})  # this triggers the prune; old row evicted
    conn = schema.get_conn()
    rows = [r[0] for r in conn.execute(
        "SELECT id FROM daemon_events ORDER BY id"
    ).fetchall()]
    assert 1 not in rows
    assert 2 in rows


# ── Gap detection ─────────────────────────────────────────────────────────

def test_gap_emitted_when_since_is_pre_retention_window():
    bus = events.EventBus(retention_rows=3, prune_every_n=1)
    for i in range(10):
        bus.publish("k", {"i": i})
    out = list(bus.replay_since(1))
    assert out[0]["type"] == "gap"
    assert out[0]["data"]["reason"] == "retention_prune"
    assert out[0]["data"]["missed_from"] == 2
    assert all(e["id"] > 1 for e in out[1:])


def test_no_gap_when_since_is_in_window():
    bus = events.EventBus(retention_rows=10, prune_every_n=100)
    for _ in range(5):
        bus.publish("k", {})
    out = list(bus.replay_since(2))
    assert all(e["type"] != "gap" for e in out)


def test_no_gap_on_empty_table():
    bus = events.EventBus()
    out = list(bus.replay_since(0))
    assert out == []


# ── Subscribers receive live events while persistence happens ─────────────

def test_subscriber_gets_published_event_live():
    bus = events.EventBus()
    sub = bus.subscribe()
    eid = bus.publish("hello", {"x": 1})
    evt = sub.get(timeout=1.0)
    assert evt["id"] == eid
    assert evt["type"] == "hello"
    assert evt["data"] == {"x": 1}
    bus.unsubscribe(sub)


def test_unsubscribe_drops_subscriber():
    bus = events.EventBus()
    sub = bus.subscribe()
    assert bus.subscriber_count() == 1
    bus.unsubscribe(sub)
    assert bus.subscriber_count() == 0


# ── reset_bus_for_tests truncates the table ───────────────────────────────

def test_reset_bus_clears_daemon_events():
    bus = events.EventBus()
    bus.publish("k", {})
    bus.publish("k", {})
    events.reset_bus_for_tests()
    conn = schema.get_conn()
    count = conn.execute("SELECT COUNT(*) FROM daemon_events").fetchone()[0]
    assert count == 0


def test_reset_bus_resets_id_sequence():
    bus = events.EventBus()
    bus.publish("k", {})
    bus.publish("k", {})
    events.reset_bus_for_tests()
    bus2 = events.EventBus()
    new_id = bus2.publish("k", {})
    assert new_id == 1


# ── Cross-restart simulation: drop the EventBus instance, recreate ────────

def test_replay_works_after_bus_recreate():
    """Simulates daemon restart: drop EventBus, recreate against same DB,
    SSE client that survived the restart can still ?since=N up."""
    bus1 = events.EventBus()
    bus1.publish("k", {"n": 1})
    bus1.publish("k", {"n": 2})
    bus1.publish("k", {"n": 3})

    # Drop and recreate (simulates daemon process restart)
    bus2 = events.EventBus()
    out = list(bus2.replay_since(1))
    assert [e["data"]["n"] for e in out] == [2, 3]
    assert all(e["type"] != "gap" for e in out)
