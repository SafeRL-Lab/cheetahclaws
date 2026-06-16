"""event_log.py — Append-only event store helpers.

This module is deliberately **stateless**: every function takes a
sqlite3 connection passed in by the caller. ``KernelStore`` (in
store.py) is the sole owner of the connection and the write lock; it
calls into here for SQL plumbing.

Event durability is achieved by the caller wrapping each
``append_event`` in the same transaction as the related state-table
update — see RFC 0003 §2 "Single-writer + transaction shape".
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class Event:
    event_id:       int
    pid:            int
    ts:             float
    kind:           str
    payload:        dict
    causation_id:   Optional[int]
    correlation_id: Optional[str]

    def to_dict(self) -> dict:
        return {
            "event_id":       self.event_id,
            "pid":            self.pid,
            "ts":             self.ts,
            "kind":           self.kind,
            "payload":        self.payload,
            "causation_id":   self.causation_id,
            "correlation_id": self.correlation_id,
        }


def _row_to_event(row: sqlite3.Row) -> Event:
    payload_raw = row["payload"]
    try:
        payload = json.loads(payload_raw) if payload_raw else {}
    except json.JSONDecodeError:
        # The kernel always writes valid JSON; corruption here is a real
        # bug but we don't want a single bad row to take out a whole
        # tail() call. Fall back to the raw string.
        payload = {"_raw": payload_raw}
    return Event(
        event_id=row["event_id"],
        pid=row["pid"],
        ts=row["ts"],
        kind=row["kind"],
        payload=payload,
        causation_id=row["causation_id"],
        correlation_id=row["correlation_id"],
    )


def append_event(
    conn: sqlite3.Connection,
    *,
    pid: int,
    kind: str,
    payload: dict,
    causation_id: Optional[int] = None,
    correlation_id: Optional[str] = None,
    ts: Optional[float] = None,
) -> int:
    """Insert one event row inside the caller's transaction.

    Returns the assigned ``event_id``. Also bumps
    ``agent_processes.last_event_id`` and ``updated_at`` for the owning
    pid so a single-row read of the process gives the most recent event
    cursor without joining ``agent_events``.
    """
    if ts is None:
        ts = time.time()
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    cur = conn.execute(
        """
        INSERT INTO agent_events
            (pid, ts, kind, payload, causation_id, correlation_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (pid, ts, kind, payload_json, causation_id, correlation_id),
    )
    event_id = cur.lastrowid
    if event_id is None:  # SQLite always assigns one for INTEGER PK AUTOINCREMENT
        raise RuntimeError("sqlite3 did not return a lastrowid for agent_events")
    conn.execute(
        "UPDATE agent_processes SET last_event_id = ?, updated_at = ? WHERE pid = ?",
        (event_id, ts, pid),
    )
    return event_id


def read_events(
    conn: sqlite3.Connection,
    *,
    pid: Optional[int] = None,
    kind: Optional[str] = None,
    since_event_id: int = 0,
    limit: int = 100,
) -> list[Event]:
    """Return events with event_id > ``since_event_id``, oldest first.

    Filters by pid and/or kind if supplied. ``limit`` is clamped to
    [1, 10_000] to keep one-shot reads bounded.
    """
    if limit < 1:
        limit = 1
    elif limit > 10_000:
        limit = 10_000

    sql_parts = ["SELECT * FROM agent_events WHERE event_id > ?"]
    params: list = [since_event_id]
    if pid is not None:
        sql_parts.append("AND pid = ?")
        params.append(pid)
    if kind is not None:
        sql_parts.append("AND kind = ?")
        params.append(kind)
    sql_parts.append("ORDER BY event_id ASC LIMIT ?")
    params.append(limit)
    sql = " ".join(sql_parts)
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_event(r) for r in rows]


def count_events(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM agent_events").fetchone()
    return int(row["n"]) if isinstance(row, sqlite3.Row) else int(row[0])


def max_event_id(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(event_id), 0) AS m FROM agent_events"
    ).fetchone()
    return int(row["m"]) if isinstance(row, sqlite3.Row) else int(row[0])
