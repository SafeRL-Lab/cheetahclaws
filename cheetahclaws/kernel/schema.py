"""schema.py — kernel.db DDL and idempotent initialisation.

The kernel uses its own SQLite database (``~/.cheetahclaws/kernel.db``)
separate from F-2's ``sessions.db``. Rationale is documented in
RFC 0003 §4 ("Why a separate database from sessions.db").

This module exposes a single entry point:

    init_schema(conn) -> int       # returns schema_version

It is safe to call repeatedly; tables are created with ``IF NOT EXISTS``
and ``schema_version`` is upserted only when missing. Callers should
have set the connection's row_factory before invoking this function so
later reads return mappings.

PRAGMAs (set on every fresh connection by the caller, not here):
    journal_mode=WAL    — concurrent readers, durable writes
    synchronous=NORMAL  — durable across power-loss after the next
                           checkpoint; matches the durability we
                           commit to in RFC 0003 §2 "Goals".
    foreign_keys=ON     — parent_pid integrity check
"""
from __future__ import annotations

import sqlite3
from typing import Optional

from .errors import SchemaMismatch


# Bumped on every breaking schema change. Forward migration is automatic
# *iff* every change since the last release is additive (only new tables
# / new columns with defaults). v1 → v2 added agent_capabilities (RFC
# 0005) and agent_ledgers (RFC 0006). v2 → v3 added agent_schedule_queue
# (RFC 0007). v3 → v4 added agent_mailboxes / agent_subscriptions /
# agent_messages (RFC 0009) and agent_registry (RFC 0010). v4 → v5
# added agent_fs_objects (RFC 0011). All purely additive so far.
EXPECTED_SCHEMA_VERSION = 5


# Order matters: agent_processes references itself via parent_pid, so it
# must be created first. agent_events references agent_processes(pid).
DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS kernel_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_processes (
        pid           INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_pid    INTEGER,
        name          TEXT NOT NULL,
        template      TEXT NOT NULL,
        state         TEXT NOT NULL CHECK(state IN
                          ('READY','RUNNING','WAITING','SUSPENDED','DEAD')),
        state_reason  TEXT,
        created_at    REAL NOT NULL,
        updated_at    REAL NOT NULL,
        started_at    REAL,
        ended_at      REAL,
        exit_kind     TEXT,
        exit_detail   TEXT,
        metadata      TEXT NOT NULL DEFAULT '{}',
        last_event_id INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (parent_pid) REFERENCES agent_processes(pid)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_processes_state
        ON agent_processes(state)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_processes_parent
        ON agent_processes(parent_pid)
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_events (
        event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        pid            INTEGER NOT NULL,
        ts             REAL NOT NULL,
        kind           TEXT NOT NULL,
        payload        TEXT NOT NULL,
        causation_id   INTEGER,
        correlation_id TEXT,
        FOREIGN KEY (pid) REFERENCES agent_processes(pid)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_events_pid
        ON agent_events(pid, event_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_events_kind
        ON agent_events(kind)
    """,
    # ── v2 (RFC 0005 — Capability) ────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS agent_capabilities (
        cap_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_cap_id INTEGER,
        pid           INTEGER NOT NULL UNIQUE,
        tool_grants   TEXT NOT NULL,
        fs_grants     TEXT NOT NULL,
        net_grants    TEXT NOT NULL,
        model_grants  TEXT NOT NULL,
        sub_agent     INTEGER NOT NULL CHECK(sub_agent IN (0,1)),
        created_at    REAL NOT NULL,
        FOREIGN KEY (pid)           REFERENCES agent_processes(pid),
        FOREIGN KEY (parent_cap_id) REFERENCES agent_capabilities(cap_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_capabilities_pid
        ON agent_capabilities(pid)
    """,
    # ── v2 (RFC 0006 — ResourceLedger) ────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS agent_ledgers (
        pid          INTEGER NOT NULL,
        dim          TEXT    NOT NULL,
        used         INTEGER NOT NULL DEFAULT 0,
        granted      INTEGER NOT NULL,
        hard_limit   INTEGER NOT NULL,
        warn_at      REAL    NOT NULL DEFAULT 0.8,
        created_at   REAL    NOT NULL,
        updated_at   REAL    NOT NULL,
        PRIMARY KEY (pid, dim),
        FOREIGN KEY (pid) REFERENCES agent_processes(pid)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_ledgers_pid
        ON agent_ledgers(pid)
    """,
    # ── v3 (RFC 0007 — AgentScheduler) ─────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS agent_schedule_queue (
        sched_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        pid           INTEGER NOT NULL,
        priority      INTEGER NOT NULL DEFAULT 0,
        runnable_at   REAL    NOT NULL DEFAULT 0,
        deadline      REAL,
        trigger       TEXT    NOT NULL DEFAULT 'manual',
        payload       TEXT    NOT NULL DEFAULT '{}',
        state         TEXT    NOT NULL DEFAULT 'queued'
                      CHECK(state IN ('queued','dispatched','completed','expired','cancelled')),
        worker_id     TEXT,
        created_at    REAL    NOT NULL,
        dispatched_at REAL,
        completed_at  REAL,
        exit_kind     TEXT,
        FOREIGN KEY (pid) REFERENCES agent_processes(pid)
    )
    """,
    # Composite index serving the claim() hot-path SELECT.
    """
    CREATE INDEX IF NOT EXISTS idx_sched_queue_pickable
        ON agent_schedule_queue (state, priority DESC, runnable_at, sched_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sched_queue_pid
        ON agent_schedule_queue (pid)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sched_queue_deadline
        ON agent_schedule_queue (deadline)
        WHERE deadline IS NOT NULL
    """,
    # ── v4 (RFC 0009 — AgentMailbox) ───────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS agent_mailboxes (
        pid          INTEGER PRIMARY KEY,
        queue_size   INTEGER NOT NULL DEFAULT 1024,
        retention_s  REAL    NOT NULL DEFAULT 3600,
        created_at   REAL    NOT NULL,
        FOREIGN KEY (pid) REFERENCES agent_processes(pid)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_subscriptions (
        pid          INTEGER NOT NULL,
        topic        TEXT    NOT NULL,
        created_at   REAL    NOT NULL,
        PRIMARY KEY (pid, topic),
        FOREIGN KEY (pid) REFERENCES agent_processes(pid)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_subscriptions_topic
        ON agent_subscriptions(topic)
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_messages (
        msg_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        sender_pid    INTEGER,
        recipient_pid INTEGER NOT NULL,
        topic         TEXT,
        kind          TEXT    NOT NULL,
        payload       TEXT    NOT NULL,
        posted_at     REAL    NOT NULL,
        delivered_at  REAL,
        expires_at    REAL,
        FOREIGN KEY (recipient_pid) REFERENCES agent_processes(pid)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_messages_pending
        ON agent_messages (recipient_pid, msg_id)
        WHERE delivered_at IS NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_messages_expires
        ON agent_messages (expires_at)
        WHERE expires_at IS NOT NULL
    """,
    # ── v4 (RFC 0010 — AgentRegistry) ──────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS agent_registry (
        name           TEXT    PRIMARY KEY,
        pid            INTEGER NOT NULL,
        tags           TEXT    NOT NULL DEFAULT '[]',
        metadata       TEXT    NOT NULL DEFAULT '{}',
        registered_at  REAL    NOT NULL,
        FOREIGN KEY (pid) REFERENCES agent_processes(pid)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_registry_pid
        ON agent_registry(pid)
    """,
    # ── v5 (RFC 0011 — AgentFS) ────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS agent_fs_objects (
        path        TEXT    PRIMARY KEY,
        owner_pid   INTEGER NOT NULL,
        content     BLOB    NOT NULL,
        size        INTEGER NOT NULL,
        mode        TEXT    NOT NULL DEFAULT 'rw'
                    CHECK(mode IN ('rw', 'ro')),
        metadata    TEXT    NOT NULL DEFAULT '{}',
        created_at  REAL    NOT NULL,
        updated_at  REAL    NOT NULL,
        accessed_at REAL,
        FOREIGN KEY (owner_pid) REFERENCES agent_processes(pid)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_fs_owner
        ON agent_fs_objects(owner_pid)
    """,
)


def init_schema(conn: sqlite3.Connection) -> int:
    """Create tables if absent, forward-migrate, return the recorded
    schema version after migration.

    Forward migration is automatic when every change between the
    recorded version and ``EXPECTED_SCHEMA_VERSION`` is additive
    (CREATE TABLE / CREATE INDEX with IF NOT EXISTS, no destructive
    drop / rename / type change). v1 → v2 satisfies that, so we
    transparently bump the stamp after running all DDL.

    Raises ``SchemaMismatch`` only when:
      * the DB stamp is greater than the code's expectation (operator
        is running an older binary against a newer DB), or
      * the stamp is malformed / missing-after-write (data corruption).

    No automatic *backward* migration. If a future schema bump is
    destructive, this function will need a per-version migration step
    here.
    """
    with conn:  # transaction
        for stmt in DDL_STATEMENTS:
            conn.execute(stmt)
        row = conn.execute(
            "SELECT value FROM kernel_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO kernel_meta(key, value) VALUES ('schema_version', ?)",
                (str(EXPECTED_SCHEMA_VERSION),),
            )
            return EXPECTED_SCHEMA_VERSION
        try:
            found = int(row[0]) if not isinstance(row, sqlite3.Row) else int(row["value"])
        except (TypeError, ValueError):
            raise SchemaMismatch(EXPECTED_SCHEMA_VERSION, None)
        if found > EXPECTED_SCHEMA_VERSION:
            raise SchemaMismatch(EXPECTED_SCHEMA_VERSION, found)
        if found < EXPECTED_SCHEMA_VERSION:
            # Forward migration: DDL above already created the new
            # tables (IF NOT EXISTS); only the version stamp needs to
            # bump. This branch is the safety check that the migration
            # was actually safe — destructive changes must add a
            # per-version migration above this and not rely on this
            # automatic bump.
            conn.execute(
                "UPDATE kernel_meta SET value = ? WHERE key = 'schema_version'",
                (str(EXPECTED_SCHEMA_VERSION),),
            )
            return EXPECTED_SCHEMA_VERSION
        return found


def open_connection(db_path: str) -> sqlite3.Connection:
    """Open a sqlite3 connection with the kernel's standard PRAGMAs.

    Uses ``check_same_thread=False`` so the daemon's request handler
    threads can share one connection, with all writes serialised by
    ``KernelStore``'s lock.
    """
    conn = sqlite3.connect(
        db_path,
        check_same_thread=False,
        detect_types=0,
        isolation_level="DEFERRED",
    )
    conn.row_factory = sqlite3.Row
    # Durability + concurrency PRAGMAs. WAL must be set outside a
    # transaction; it persists across reopens once set.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_schema_version(conn: sqlite3.Connection) -> Optional[int]:
    """Return the recorded schema_version or None if the meta table
    doesn't exist yet. Used by tests; production code calls
    ``init_schema``."""
    try:
        row = conn.execute(
            "SELECT value FROM kernel_meta WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    try:
        return int(row["value"]) if isinstance(row, sqlite3.Row) else int(row[0])
    except (TypeError, ValueError):
        return None
