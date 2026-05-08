"""F-3 tests for monitor/scheduler.py — report persistence + SSE event emission.

Fetcher and summarizer are mocked so these tests don't touch the network
or LLM providers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import monitor.store as store
import monitor.scheduler as scheduler
from cc_daemon import events, schema


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(store, "STORE_PATH",
                        tmp_path / "monitor_subscriptions.json")
    schema.set_db_path(tmp_path / "sessions.db")
    schema._local.conn = None
    store._migration_done_in_process = False
    events.reset_bus_for_tests()

    # Replace fetch / summarize / deliver in scheduler's namespace so tests
    # don't depend on network or LLM.  notifier.deliver returns
    # {channel: error_or_empty} — empty string == success.
    monkeypatch.setattr(scheduler, "fetch", lambda _topic: "raw stub")
    monkeypatch.setattr(scheduler, "summarize",
                        lambda raw, _config: f"SUMMARY: {raw}")
    monkeypatch.setattr(scheduler, "deliver",
                        lambda _report, channels, _config:
                            {ch: "" for ch in channels})
    monkeypatch.setattr(scheduler, "auto_channels", lambda _config: ["console"])

    yield

    schema._local.conn = None
    schema.set_db_path(schema.get_default_db_path())


# ── run_one persistence path ──────────────────────────────────────────────

def test_run_one_persists_report_to_monitor_reports():
    store.add_subscription("arxiv", schedule="daily", channels=["console"])
    out = scheduler.run_one("arxiv", config={})
    assert "SUMMARY:" in out
    reports = store.list_reports("arxiv")
    assert len(reports) == 1
    assert reports[0]["body"].startswith("SUMMARY:")
    assert reports[0]["sent_to"] == ["console"]


def test_run_one_records_failed_delivery_in_report():
    store.add_subscription("arxiv", schedule="daily", channels=["telegram"])
    # Force telegram delivery to fail
    import monitor.scheduler as sched
    def _failing_deliver(_report, channels, _config):
        return {ch: "no token" for ch in channels}
    sched.deliver = _failing_deliver
    out = scheduler.run_one("arxiv", config={})
    assert "[Delivery errors:" in out
    reports = store.list_reports("arxiv")
    assert len(reports) == 1
    assert "Delivery errors" in reports[0]["body"]
    # Successful sent_to is empty when all channels failed
    assert reports[0]["sent_to"] == []


def test_run_one_updates_last_run_on_subscription_row():
    store.add_subscription("arxiv", schedule="daily", channels=["console"])
    assert store.get_subscription("arxiv")["last_run"] is None
    scheduler.run_one("arxiv", config={})
    assert store.get_subscription("arxiv")["last_run"] is not None


# ── monitor_report SSE event ──────────────────────────────────────────────

def test_run_one_publishes_monitor_report_event():
    store.add_subscription("arxiv", schedule="daily", channels=["console"])
    bus = events.get_bus()
    sub = bus.subscribe()
    try:
        scheduler.run_one("arxiv", config={})
        evt = sub.get(timeout=2.0)
        assert evt["type"] == "monitor_report"
        assert evt["data"]["topic"] == "arxiv"
        assert evt["data"]["sent_to"] == ["console"]
        assert evt["data"]["errors"] == []
        assert evt["data"]["body"].startswith("SUMMARY:")
        # report_id ties the SSE event back to the row in monitor_reports
        assert evt["data"]["report_id"]
    finally:
        bus.unsubscribe(sub)


def test_event_report_id_matches_persisted_row():
    store.add_subscription("arxiv", schedule="daily", channels=["console"])
    bus = events.get_bus()
    sub = bus.subscribe()
    try:
        scheduler.run_one("arxiv", config={})
        evt = sub.get(timeout=2.0)
        report_id = evt["data"]["report_id"]
        rows = store.list_reports("arxiv")
        assert any(r["id"] == report_id for r in rows)
    finally:
        bus.unsubscribe(sub)


def test_event_carries_error_list_on_partial_failure():
    store.add_subscription("arxiv", schedule="daily",
                            channels=["telegram", "console"])
    import monitor.scheduler as sched
    sched.deliver = lambda _r, channels, _c: {
        "telegram": "no token", "console": ""}
    bus = events.get_bus()
    sub = bus.subscribe()
    try:
        scheduler.run_one("arxiv", config={})
        evt = sub.get(timeout=2.0)
        assert evt["data"]["sent_to"] == ["console"]
        assert any("telegram" in e for e in evt["data"]["errors"])
    finally:
        bus.unsubscribe(sub)


# ── Survives daemon restart (key F-3 win) ────────────────────────────────

def test_reports_survive_simulated_daemon_restart():
    """Persist a few reports, drop the events singleton (mimic restart),
    and verify list_reports still returns them — the SQLite source of
    truth means cross-restart history is intact."""
    store.add_subscription("arxiv", schedule="daily", channels=["console"])
    scheduler.run_one("arxiv", config={})
    scheduler.run_one("arxiv", config={})
    events.reset_bus_for_tests()  # bus fresh, but daemon_events also cleared
    # daemon_events is event log, not report log — restart it to mimic
    # a fresh daemon process picking up persisted reports.
    reports = store.list_reports("arxiv")
    assert len(reports) == 2
