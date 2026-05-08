"""Tests for cc_daemon/schema.py — additive tables + idempotent init."""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cc_daemon import schema


# ── Idempotent init ────────────────────────────────────────────────────────

EXPECTED_TABLES = {
    "schema_meta",
    "daemon_events",
    "agent_runs",
    "agent_iterations",
    "jobs",
    "monitor_subscriptions",
    "monitor_reports",
    "bridges",
}


def _table_names(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def test_init_creates_all_expected_tables(tmp_path: Path):
    db = tmp_path / "sessions.db"
    schema.init_schema(db)
    present = _table_names(db)
    for t in EXPECTED_TABLES:
        assert t in present, f"missing table: {t}"


def test_init_is_idempotent(tmp_path: Path):
    db = tmp_path / "sessions.db"
    schema.init_schema(db)
    schema.init_schema(db)  # must not raise
    schema.init_schema(db)
    # Schema version row remains a single entry.
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT key, value FROM schema_meta WHERE key='schema_version'"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0][1] == str(schema.CURRENT_SCHEMA_VERSION)


def test_init_records_schema_version(tmp_path: Path):
    db = tmp_path / "sessions.db"
    schema.init_schema(db)
    assert schema.get_schema_version(db) == schema.CURRENT_SCHEMA_VERSION


def test_get_schema_version_on_virgin_db(tmp_path: Path):
    db = tmp_path / "absent.db"
    assert schema.get_schema_version(db) is None


def test_init_stamps_updated_at(tmp_path: Path):
    db = tmp_path / "sessions.db"
    schema.init_schema(db)
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT updated_at FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    # ISO 8601 with Z suffix
    ts = row[0]
    assert ts.endswith("Z") and "T" in ts


# ── Coexistence with session_store ────────────────────────────────────────

def test_init_does_not_drop_existing_sessions_table(tmp_path: Path):
    """session_store.py creates `sessions` + `sessions_fts`. Daemon init
    must not disturb them."""
    db = tmp_path / "sessions.db"
    # Stand up a sessions table the way session_store would.
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, title TEXT, model TEXT,
                saved_at TEXT NOT NULL, turn_count INTEGER,
                input_tokens INTEGER, output_tokens INTEGER,
                messages TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE sessions_fts USING fts5(
                id, title, content, tokenize='unicode61'
            );
            INSERT INTO sessions (id, saved_at, messages)
                VALUES ('test-session', '2026-01-01 00:00:00', '[]');
        """)
        conn.commit()
    finally:
        conn.close()

    schema.init_schema(db)

    # sessions row still present + new daemon tables alongside.
    conn = sqlite3.connect(str(db))
    try:
        sess_count = conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]
    finally:
        conn.close()
    assert sess_count == 1

    present = _table_names(db)
    assert "sessions" in present
    assert "sessions_fts" in present
    for t in EXPECTED_TABLES:
        assert t in present


# ── Connection accessor ────────────────────────────────────────────────────

def test_get_conn_returns_same_conn_per_thread(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(schema, "_db_path", tmp_path / "sessions.db")
    schema._local.conn = None  # reset thread-local
    a = schema.get_conn()
    b = schema.get_conn()
    assert a is b


def test_get_conn_auto_inits_schema(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(schema, "_db_path", tmp_path / "sessions.db")
    schema._local.conn = None
    conn = schema.get_conn()
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {r[0] for r in rows}
    for t in EXPECTED_TABLES:
        assert t in names


def test_set_db_path_drops_stale_thread_conn(tmp_path: Path, monkeypatch):
    db1 = tmp_path / "first.db"
    db2 = tmp_path / "second.db"
    schema.set_db_path(db1)
    schema._local.conn = None
    conn1 = schema.get_conn()
    schema.set_db_path(db2)
    conn2 = schema.get_conn()
    assert conn1 is not conn2


# ── Schema-shape sanity (catches accidental column drift) ─────────────────

def test_daemon_events_columns(tmp_path: Path):
    db = tmp_path / "sessions.db"
    schema.init_schema(db)
    conn = sqlite3.connect(str(db))
    try:
        cols = {row[1] for row in
                conn.execute("PRAGMA table_info(daemon_events)").fetchall()}
    finally:
        conn.close()
    assert {"id", "ts", "kind", "payload_json"} <= cols


def test_jobs_columns(tmp_path: Path):
    db = tmp_path / "sessions.db"
    schema.init_schema(db)
    conn = sqlite3.connect(str(db))
    try:
        cols = {row[1] for row in
                conn.execute("PRAGMA table_info(jobs)").fetchall()}
    finally:
        conn.close()
    expected = {"id", "title", "prompt", "source", "status", "created_at",
                "started_at", "done_at", "duration_s", "steps_json",
                "step_count", "current_step", "result", "error", "retry_of"}
    assert expected <= cols


def test_monitor_subscriptions_columns(tmp_path: Path):
    db = tmp_path / "sessions.db"
    schema.init_schema(db)
    conn = sqlite3.connect(str(db))
    try:
        cols = {row[1] for row in
                conn.execute("PRAGMA table_info(monitor_subscriptions)").fetchall()}
    finally:
        conn.close()
    expected = {"topic", "schedule", "enabled", "last_run_at",
                "next_run_at", "recipients_json", "config_json"}
    assert expected <= cols
