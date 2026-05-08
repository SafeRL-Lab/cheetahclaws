"""schema.py — daemon-owned SQLite tables in ~/.cheetahclaws/sessions.db.

Roadmap reference: ``docs/RFC/0002-daemon-foundation-roadmap.md`` §F-2.

The daemon shares the same SQLite file as :mod:`session_store` (so users
end up with one ``sessions.db`` to back up, not two).  This module owns
seven additive tables — the existing ``sessions`` / ``sessions_fts`` are
left untouched.

Tables:
  ``daemon_events``       — append-only event log (replaces F-1's in-memory ring)
  ``agent_runs``          — one row per ``/agent`` runner (populated in F-4)
  ``agent_iterations``    — per-iteration log for those runners
  ``jobs``                — replaces the F-1 ``~/.cheetahclaws/jobs.json`` file
  ``monitor_subscriptions`` — durable ``/monitor subscribe ...`` state (F-3)
  ``monitor_reports``     — generated reports per subscription run
  ``bridges``             — bridge enabled/last-error state (F-6/7/8)
  ``schema_meta``         — schema version + last-init-at, for future migrations

All connections go through :func:`get_conn` which:
  * Uses a thread-local connection (matches ``session_store`` pattern).
  * Enables WAL + 5 s busy timeout.
  * Calls :func:`init_schema` on first connect for the thread.

:func:`init_schema` is idempotent — running on a fresh DB creates everything;
running on an already-initialised DB is a no-op.  Schema bumps live in
:func:`_apply_migrations` and bump ``CURRENT_SCHEMA_VERSION``.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────

CURRENT_SCHEMA_VERSION = 1

DEFAULT_EVENT_RETENTION_HOURS = 24
DEFAULT_EVENT_RETENTION_ROWS  = 100_000

_TABLES_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daemon_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    kind          TEXT NOT NULL,
    payload_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_daemon_events_ts ON daemon_events(ts);

CREATE TABLE IF NOT EXISTS agent_runs (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    template        TEXT NOT NULL,
    args            TEXT,
    status          TEXT NOT NULL,
    auto_approve    INTEGER NOT NULL DEFAULT 1,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    last_iteration  INTEGER DEFAULT 0,
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs(status);

CREATE TABLE IF NOT EXISTS agent_iterations (
    run_id      TEXT NOT NULL,
    iteration   INTEGER NOT NULL,
    ts          TEXT NOT NULL,
    status      TEXT NOT NULL,
    duration_s  REAL,
    summary     TEXT,
    in_tokens   INTEGER DEFAULT 0,
    out_tokens  INTEGER DEFAULT 0,
    cost_usd    REAL DEFAULT 0,
    PRIMARY KEY (run_id, iteration)
);

CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    title           TEXT,
    prompt          TEXT,
    source          TEXT,
    status          TEXT NOT NULL,
    created_at      TEXT,
    started_at      TEXT,
    done_at         TEXT,
    duration_s      REAL DEFAULT 0,
    steps_json      TEXT,
    step_count      INTEGER DEFAULT 0,
    current_step    TEXT,
    result          TEXT,
    error           TEXT,
    retry_of        TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);

CREATE TABLE IF NOT EXISTS monitor_subscriptions (
    topic           TEXT PRIMARY KEY,
    schedule        TEXT NOT NULL,
    enabled         INTEGER NOT NULL DEFAULT 1,
    last_run_at     TEXT,
    next_run_at     TEXT,
    recipients_json TEXT,
    config_json     TEXT
);

CREATE TABLE IF NOT EXISTS monitor_reports (
    id           TEXT PRIMARY KEY,
    topic        TEXT NOT NULL,
    ts           TEXT NOT NULL,
    body         TEXT,
    sent_to_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_monitor_reports_topic_ts
    ON monitor_reports(topic, ts);

CREATE TABLE IF NOT EXISTS bridges (
    kind          TEXT PRIMARY KEY,
    enabled       INTEGER NOT NULL DEFAULT 0,
    config_json   TEXT,
    last_poll_at  TEXT,
    last_error    TEXT
);
"""


# ── Module-level state ────────────────────────────────────────────────────

_db_path: Optional[Path] = None
_local = threading.local()
_init_lock = threading.Lock()


def get_default_db_path() -> Path:
    """Return ``~/.cheetahclaws/sessions.db`` (shared with session_store)."""
    from cc_config import CONFIG_DIR
    return CONFIG_DIR / "sessions.db"


def set_db_path(path: Path) -> None:
    """Override the daemon DB location (used by tests; otherwise default)."""
    global _db_path
    _db_path = path
    # Drop any existing thread-local connection so subsequent get_conn()
    # picks up the new path.
    if hasattr(_local, "conn") and _local.conn is not None:
        try:
            _local.conn.close()
        except Exception:
            pass
        _local.conn = None


def _resolve_path() -> Path:
    return _db_path if _db_path is not None else get_default_db_path()


# ── Init ──────────────────────────────────────────────────────────────────

def init_schema(db_path: Optional[Path] = None) -> None:
    """Idempotently create the daemon's tables on *db_path*.

    Safe to call multiple times.  Holds an internal lock so concurrent
    callers don't trip on each other.
    """
    target = db_path if db_path is not None else _resolve_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    with _init_lock:
        conn = sqlite3.connect(str(target), timeout=10)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            # synchronous=NORMAL is safe under WAL: durability across a power
            # loss is preserved at checkpoint boundaries, with the only risk
            # being loss of the *most recent* transactions on hard kernel
            # crash.  For an event log that is already retention-pruned in
            # 24 h windows that's an acceptable trade for ~5-10× throughput
            # — see #74 review §7 follow-up benchmark.
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.executescript(_TABLES_DDL)
            _record_schema_version(conn, CURRENT_SCHEMA_VERSION)
            _apply_migrations(conn)
            conn.commit()
        finally:
            conn.close()


def _record_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Stamp the schema version + init timestamp.  Idempotent UPSERT."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT INTO schema_meta (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
        "updated_at=excluded.updated_at",
        ("schema_version", str(version), now),
    )


def _apply_migrations(_conn: sqlite3.Connection) -> None:
    """Future-version migrations land here.

    Read ``schema_version`` from ``schema_meta``; for each version below
    ``CURRENT_SCHEMA_VERSION`` apply the corresponding ALTER/CREATE.  v1 is
    the initial layout, so this is a no-op today.
    """
    return


# ── Connection accessor ───────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    """Thread-local connection to the daemon DB.  Auto-inits schema."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn
    target = _resolve_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    init_schema(target)
    conn = sqlite3.connect(str(target), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    _local.conn = conn
    return conn


def get_schema_version(db_path: Optional[Path] = None) -> Optional[int]:
    """Return the recorded schema version, or None on a virgin DB."""
    target = db_path if db_path is not None else _resolve_path()
    if not target.exists():
        return None
    conn = sqlite3.connect(str(target), timeout=10)
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None
