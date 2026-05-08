"""F-3 tests for monitor/store.py — SQLite backing + JSON migration."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import monitor.store as store
from cc_daemon import schema


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch):
    """Each test gets a private sessions.db AND a tmp STORE_PATH so the
    legacy migration can't touch the developer's real file."""
    monkeypatch.setattr(store, "STORE_PATH", tmp_path / "monitor_subscriptions.json")
    schema.set_db_path(tmp_path / "sessions.db")
    schema._local.conn = None
    store._migration_done_in_process = False
    yield
    schema._local.conn = None
    schema.set_db_path(schema.get_default_db_path())


# ── CRUD ────────────────────────────────────────────────────────────────────

def test_add_then_list_returns_subscription():
    store.add_subscription("arxiv", schedule="daily", channels=["console"])
    subs = store.list_subscriptions()
    assert len(subs) == 1
    assert subs[0]["topic"] == "arxiv"
    assert subs[0]["schedule"] == "daily"
    assert subs[0]["channels"] == ["console"]


def test_add_is_upsert():
    store.add_subscription("arxiv", schedule="daily", channels=["console"])
    store.add_subscription("arxiv", schedule="6h", channels=["telegram"])
    subs = store.list_subscriptions()
    assert len(subs) == 1
    assert subs[0]["schedule"] == "6h"
    assert subs[0]["channels"] == ["telegram"]


def test_get_subscription_returns_none_when_absent():
    assert store.get_subscription("nope") is None


def test_get_subscription_returns_match():
    store.add_subscription("arxiv", schedule="daily", channels=["console"])
    sub = store.get_subscription("arxiv")
    assert sub is not None
    assert sub["topic"] == "arxiv"


def test_remove_subscription_returns_true_when_removed():
    store.add_subscription("arxiv", schedule="daily")
    assert store.remove_subscription("arxiv") is True
    assert store.list_subscriptions() == []


def test_remove_subscription_returns_false_when_absent():
    assert store.remove_subscription("never-was") is False


def test_update_last_run_records_timestamp_and_preview():
    store.add_subscription("arxiv", schedule="daily")
    store.update_last_run("arxiv", "Today's findings: …" + "x" * 1000)
    sub = store.get_subscription("arxiv")
    assert sub["last_run"] is not None
    # last_report preview is ≤500 chars per legacy behaviour
    assert len(sub["last_report"]) <= 500


def test_subscription_persists_across_connections():
    """Drop and re-create the thread-local connection to mimic a fresh
    process reading the same DB."""
    store.add_subscription("arxiv", schedule="daily", channels=["console"])
    schema._local.conn = None  # drop conn → next call re-opens
    subs = store.list_subscriptions()
    assert any(s["topic"] == "arxiv" for s in subs)


# ── JSON → SQLite migration ──────────────────────────────────────────────

def _legacy_payload(subs: list[dict]) -> str:
    return json.dumps({"subscriptions": subs}, ensure_ascii=False)


def test_migration_imports_legacy_subscriptions(tmp_path):
    legacy = [
        {"id": "abc12345", "topic": "arxiv", "schedule": "daily",
         "channels": ["console"], "created_at": "2026-04-01T10:00:00",
         "last_run": "2026-04-02T10:00:00", "next_run": None,
         "last_report": "Yesterday's report …"},
        {"id": "def67890", "topic": "news", "schedule": "6h",
         "channels": ["telegram"], "created_at": "2026-04-01T10:00:00",
         "last_run": None, "next_run": None, "last_report": None},
    ]
    store.STORE_PATH.write_text(_legacy_payload(legacy), encoding="utf-8")
    subs = store.list_subscriptions()
    topics = {s["topic"] for s in subs}
    assert {"arxiv", "news"} <= topics


def test_migration_carries_last_run_and_channels():
    legacy = [{"id": "x", "topic": "arxiv", "schedule": "daily",
               "channels": ["telegram", "console"],
               "created_at": "2026-04-01T10:00:00",
               "last_run": "2026-04-02T10:00:00", "next_run": None,
               "last_report": "old digest"}]
    store.STORE_PATH.write_text(_legacy_payload(legacy), encoding="utf-8")
    sub = store.get_subscription("arxiv")
    assert sub["last_run"] == "2026-04-02T10:00:00"
    assert set(sub["channels"]) == {"telegram", "console"}


def test_migration_seeds_last_report_into_monitor_reports():
    """A subscription with last_report should produce one row in
    monitor_reports so the post-upgrade /monitor history view isn't
    empty."""
    legacy = [{"id": "x", "topic": "arxiv", "schedule": "daily",
               "channels": ["console"], "created_at": "",
               "last_run": "2026-04-02T10:00:00", "next_run": None,
               "last_report": "Old digest body"}]
    store.STORE_PATH.write_text(_legacy_payload(legacy), encoding="utf-8")
    store.list_subscriptions()  # triggers migration
    reports = store.list_reports("arxiv")
    assert len(reports) == 1
    assert reports[0]["body"] == "Old digest body"


def test_migration_is_idempotent():
    legacy = [{"id": "x", "topic": "arxiv", "schedule": "daily",
               "channels": [], "created_at": "", "last_run": None,
               "next_run": None, "last_report": None}]
    store.STORE_PATH.write_text(_legacy_payload(legacy), encoding="utf-8")
    store.list_subscriptions()
    store.list_subscriptions()
    store.list_subscriptions()
    subs = store.list_subscriptions()
    assert sum(1 for s in subs if s["topic"] == "arxiv") == 1


def test_migration_marker_is_recorded():
    store.STORE_PATH.write_text(_legacy_payload([]), encoding="utf-8")
    store.list_subscriptions()
    row = schema.get_conn().execute(
        "SELECT value FROM schema_meta WHERE key='monitor_migrated_from_json'"
    ).fetchone()
    assert row is not None
    assert row[0] == "1"


def test_migration_keeps_legacy_file_in_place():
    legacy = [{"id": "x", "topic": "arxiv", "schedule": "daily",
               "channels": [], "created_at": "", "last_run": None,
               "next_run": None, "last_report": None}]
    store.STORE_PATH.write_text(_legacy_payload(legacy), encoding="utf-8")
    store.list_subscriptions()
    assert store.STORE_PATH.exists()  # one-release fallback


def test_migration_tolerates_corrupt_json():
    store.STORE_PATH.write_text("{ this is not json", encoding="utf-8")
    # Should not raise.
    assert store.list_subscriptions() == []


# ── Reports ─────────────────────────────────────────────────────────────────

def test_save_report_returns_id_and_persists():
    store.add_subscription("arxiv", schedule="daily")
    rid = store.save_report("arxiv", "Today's digest", sent_to=["console"])
    assert rid
    reports = store.list_reports("arxiv")
    assert len(reports) == 1
    assert reports[0]["id"] == rid
    assert reports[0]["body"] == "Today's digest"
    assert reports[0]["sent_to"] == ["console"]


def test_list_reports_filters_by_topic():
    store.add_subscription("arxiv", schedule="daily")
    store.add_subscription("news", schedule="6h")
    store.save_report("arxiv", "A", sent_to=[])
    store.save_report("news", "N", sent_to=[])
    store.save_report("arxiv", "B", sent_to=[])
    arxiv_reports = store.list_reports("arxiv")
    assert {r["body"] for r in arxiv_reports} == {"A", "B"}


def test_list_reports_orders_newest_first():
    store.add_subscription("arxiv", schedule="daily")
    store.save_report("arxiv", "first", sent_to=[])
    store.save_report("arxiv", "second", sent_to=[])
    store.save_report("arxiv", "third", sent_to=[])
    reports = store.list_reports("arxiv")
    # Most recent first
    assert reports[0]["body"] == "third"
    assert reports[-1]["body"] == "first"
