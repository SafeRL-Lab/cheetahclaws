"""F-3 unit tests for cc_daemon/monitor_methods.py — RPC wrappers."""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import monitor.store as store
import monitor.scheduler as scheduler
from cc_daemon import events, monitor_methods, schema
from cc_daemon.rpc import CallContext, RpcRegistry


# ── Fakes / fixtures ──────────────────────────────────────────────────────

class _FakeState:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.shutdown_event = threading.Event()


def _ctx(client_id: str = "test-client") -> CallContext:
    return CallContext(client_id=client_id, transport="tcp", api_version="0")


def _call(registry: RpcRegistry, method: str, params=None) -> dict:
    envelope = {"jsonrpc": "2.0", "id": 1, "method": method,
                "params": params or {}}
    response, _status = registry.dispatch(envelope, _ctx())
    return response


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(store, "STORE_PATH",
                        tmp_path / "monitor_subscriptions.json")
    schema.set_db_path(tmp_path / "sessions.db")
    schema._local.conn = None
    store._migration_done_in_process = False
    events.reset_bus_for_tests()
    # Stub fetch/summarize/deliver so monitor.run doesn't hit network.
    monkeypatch.setattr(scheduler, "fetch", lambda _t: "raw")
    monkeypatch.setattr(scheduler, "summarize", lambda _r, _c: "SUMMARY")
    monkeypatch.setattr(scheduler, "deliver",
                        lambda _r, channels, _c: {ch: "" for ch in channels})
    monkeypatch.setattr(scheduler, "auto_channels", lambda _c: ["console"])
    yield
    schema._local.conn = None
    schema.set_db_path(schema.get_default_db_path())


# ── monitor.subscribe ─────────────────────────────────────────────────────

def test_subscribe_creates_subscription():
    registry = RpcRegistry()
    monitor_methods.register(registry, _FakeState())
    resp = _call(registry, "monitor.subscribe",
                 {"topic": "arxiv", "schedule": "daily",
                  "channels": ["console"]})
    assert resp["result"]["topic"] == "arxiv"
    assert resp["result"]["schedule"] == "daily"
    # Persisted in SQLite
    assert store.get_subscription("arxiv") is not None


def test_subscribe_is_upsert():
    registry = RpcRegistry()
    monitor_methods.register(registry, _FakeState())
    _call(registry, "monitor.subscribe",
          {"topic": "arxiv", "schedule": "daily"})
    resp = _call(registry, "monitor.subscribe",
                 {"topic": "arxiv", "schedule": "6h",
                  "channels": ["telegram"]})
    assert resp["result"]["schedule"] == "6h"
    assert len(store.list_subscriptions()) == 1


def test_subscribe_rejects_missing_topic():
    registry = RpcRegistry()
    monitor_methods.register(registry, _FakeState())
    resp = _call(registry, "monitor.subscribe", {})
    assert "error" in resp


def test_subscribe_rejects_non_list_channels():
    registry = RpcRegistry()
    monitor_methods.register(registry, _FakeState())
    resp = _call(registry, "monitor.subscribe",
                 {"topic": "arxiv", "channels": "telegram"})  # str, not list
    assert "error" in resp


# ── monitor.unsubscribe ──────────────────────────────────────────────────

def test_unsubscribe_removes_existing():
    registry = RpcRegistry()
    monitor_methods.register(registry, _FakeState())
    store.add_subscription("arxiv", schedule="daily")
    resp = _call(registry, "monitor.unsubscribe", {"topic": "arxiv"})
    assert resp["result"] == {"topic": "arxiv", "removed": True}
    assert store.get_subscription("arxiv") is None


def test_unsubscribe_idempotent_on_missing():
    registry = RpcRegistry()
    monitor_methods.register(registry, _FakeState())
    resp = _call(registry, "monitor.unsubscribe", {"topic": "never-was"})
    assert resp["result"] == {"topic": "never-was", "removed": False}


# ── monitor.list ─────────────────────────────────────────────────────────

def test_list_returns_all_subscriptions():
    registry = RpcRegistry()
    monitor_methods.register(registry, _FakeState())
    store.add_subscription("arxiv", schedule="daily")
    store.add_subscription("news", schedule="6h", channels=["telegram"])
    resp = _call(registry, "monitor.list", {})
    topics = {s["topic"] for s in resp["result"]["subscriptions"]}
    assert topics == {"arxiv", "news"}


def test_list_when_empty_returns_empty_array():
    registry = RpcRegistry()
    monitor_methods.register(registry, _FakeState())
    resp = _call(registry, "monitor.list", {})
    assert resp["result"]["subscriptions"] == []


# ── monitor.run ──────────────────────────────────────────────────────────

def test_run_returns_report_and_persists():
    registry = RpcRegistry()
    monitor_methods.register(registry, _FakeState({"model": "stub"}))
    store.add_subscription("arxiv", schedule="daily", channels=["console"])
    resp = _call(registry, "monitor.run", {"topic": "arxiv"})
    assert "SUMMARY" in resp["result"]["report"]
    assert resp["result"]["topic"] == "arxiv"
    assert len(store.list_reports("arxiv")) == 1


def test_run_rejects_missing_topic():
    registry = RpcRegistry()
    monitor_methods.register(registry, _FakeState())
    resp = _call(registry, "monitor.run", {})
    assert "error" in resp


def test_run_publishes_monitor_report_event():
    registry = RpcRegistry()
    state = _FakeState({"model": "stub"})
    monitor_methods.register(registry, state)
    store.add_subscription("arxiv", schedule="daily", channels=["console"])
    bus = events.get_bus()
    sub = bus.subscribe()
    try:
        _call(registry, "monitor.run", {"topic": "arxiv"})
        evt = sub.get(timeout=2.0)
        assert evt["type"] == "monitor_report"
        assert evt["data"]["topic"] == "arxiv"
    finally:
        bus.unsubscribe(sub)


# ── Coexistence with other registered methods ───────────────────────────

def test_register_does_not_clash_with_system_methods():
    from cc_daemon import system_methods
    registry = RpcRegistry()
    system_methods.register(registry, _FakeState())
    monitor_methods.register(registry, _FakeState())
    methods = registry.methods()
    for m in ("system.ping", "system.shutdown",
               "monitor.subscribe", "monitor.unsubscribe",
               "monitor.list", "monitor.run"):
        assert m in methods
