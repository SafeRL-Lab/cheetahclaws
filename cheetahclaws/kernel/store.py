"""store.py — KernelStore: the high-level CRUD + state-machine API.

Owns the SQLite connection and the write lock. Every state transition
runs as one transaction containing both the row UPDATE and the matching
INSERT into agent_events, so a crash either commits both or commits
neither (RFC 0003 §2 "Single-writer + transaction shape").

After commit, events are also published to the daemon's in-memory
EventBus so SSE subscribers see live activity. The bus publish is
strictly **after** the durable commit; a crash between commit and
publish is harmless because clients can backfill via
``kernel.events.tail``.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import event_log
from .errors import (
    IllegalTransition,
    InvalidPayload,
    UnknownPid,
)
from .event_log import Event
from .process import (
    ALLOWED_TRANSITIONS,
    AgentProcess,
    AgentState,
    is_transition_allowed,
)
from .schema import (
    EXPECTED_SCHEMA_VERSION,
    init_schema,
    open_connection,
)


# Bus is structurally typed: anything with a ``publish(kind, dict)``
# method. We don't import ``daemon.events.EventBus`` directly to keep
# this module independent of the daemon (so it can be tested standalone).
class _BusProtocol:
    def publish(self, ev_type: str, data: dict) -> int: ...  # noqa


# Recovery policies for stale RUNNING/WAITING rows on daemon startup.
RECOVERY_SUSPEND   = "suspend"
RECOVERY_MARK_DEAD = "mark-dead"
_RECOVERY_POLICIES = (RECOVERY_SUSPEND, RECOVERY_MARK_DEAD)


# ── Kernel event kinds ─────────────────────────────────────────────────────
EV_PROCESS_CREATED      = "kernel.process.created"
EV_PROCESS_TRANSITIONED = "kernel.process.transitioned"
EV_PROCESS_TERMINATED   = "kernel.process.terminated"
EV_PROCESS_RECOVERED    = "kernel.process.recovered"


def _row_to_agent(row: sqlite3.Row) -> AgentProcess:
    return AgentProcess(
        pid           = row["pid"],
        parent_pid    = row["parent_pid"],
        name          = row["name"],
        template      = row["template"],
        state         = row["state"],
        state_reason  = row["state_reason"],
        created_at    = row["created_at"],
        updated_at    = row["updated_at"],
        started_at    = row["started_at"],
        ended_at      = row["ended_at"],
        exit_kind     = row["exit_kind"],
        exit_detail   = json.loads(row["exit_detail"]) if row["exit_detail"] else None,
        metadata      = json.loads(row["metadata"]) if row["metadata"] else {},
        last_event_id = row["last_event_id"] or 0,
    )


class KernelStore:
    """Thread-safe owner of kernel.db.

    Use ``open(...)`` (alternate constructor) for the production path —
    it opens the connection, sets PRAGMAs, and runs schema init in one
    call. Tests can construct directly with a pre-opened connection.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        bus: Optional[_BusProtocol] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self._conn = conn
        self._bus = bus
        self._db_path = db_path
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def open(
        cls,
        db_path: str | Path,
        *,
        bus: Optional[_BusProtocol] = None,
    ) -> "KernelStore":
        path = str(db_path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = open_connection(path)
        init_schema(conn)
        return cls(conn, bus=bus, db_path=path)

    # ── Sibling stores (RFC 0005, 0006) attach via these accessors ───
    # so they can share our connection + write lock. See
    # CapabilityStore / LedgerStore docstrings for why a shared lock is
    # mandatory.

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    @property
    def write_lock(self) -> threading.Lock:
        return self._lock

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            try:
                self._conn.close()
            finally:
                self._closed = True

    # ── Recovery ──────────────────────────────────────────────────────

    def recover(self, policy: str = RECOVERY_SUSPEND) -> int:
        """Coerce stale RUNNING/WAITING rows to a safe terminal-or-paused
        state. Called once on daemon startup. Returns count of rows
        touched. Idempotent: re-running finds no stale rows because the
        first call already moved them out of RUNNING/WAITING.
        """
        if policy not in _RECOVERY_POLICIES:
            raise InvalidPayload(
                f"unknown recovery policy: {policy!r} "
                f"(use one of {_RECOVERY_POLICIES})",
                field="policy",
            )

        target_state = (
            AgentState.SUSPENDED if policy == RECOVERY_SUSPEND else AgentState.DEAD
        )
        reason = "daemon_restart"
        published: list[tuple[int, str, str, int]] = []

        with self._lock:
            with self._conn:
                rows = self._conn.execute(
                    "SELECT pid, state FROM agent_processes "
                    "WHERE state IN ('RUNNING', 'WAITING')"
                ).fetchall()
                for r in rows:
                    pid = r["pid"]
                    prev = r["state"]
                    now = time.time()
                    if target_state == AgentState.DEAD:
                        self._conn.execute(
                            "UPDATE agent_processes SET state=?, state_reason=?, "
                            "updated_at=?, ended_at=?, exit_kind=? WHERE pid=?",
                            (target_state, reason, now, now, "crashed", pid),
                        )
                    else:
                        self._conn.execute(
                            "UPDATE agent_processes SET state=?, state_reason=?, "
                            "updated_at=? WHERE pid=?",
                            (target_state, reason, now, pid),
                        )
                    event_id = event_log.append_event(
                        self._conn,
                        pid=pid,
                        kind=EV_PROCESS_RECOVERED,
                        payload={
                            "prev_state": prev,
                            "new_state": target_state,
                            "policy": policy,
                            "reason": reason,
                        },
                        ts=now,
                    )
                    published.append((pid, prev, target_state, event_id))

        # Bus publish is best-effort and outside the lock + transaction.
        if self._bus is not None:
            for pid, prev, new, event_id in published:
                self._bus.publish(EV_PROCESS_RECOVERED, {
                    "pid": pid,
                    "prev_state": prev,
                    "new_state": new,
                    "policy": policy,
                    "event_id": event_id,
                })
        return len(published)

    # ── Process CRUD ──────────────────────────────────────────────────

    def create(
        self,
        *,
        name: str,
        template: str,
        parent_pid: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> AgentProcess:
        if not isinstance(name, str) or not name:
            raise InvalidPayload("name must be a non-empty string", field="name")
        if not isinstance(template, str) or not template:
            raise InvalidPayload("template must be a non-empty string", field="template")
        if parent_pid is not None and not isinstance(parent_pid, int):
            raise InvalidPayload("parent_pid must be int or null", field="parent_pid")
        if metadata is not None and not isinstance(metadata, dict):
            raise InvalidPayload("metadata must be an object", field="metadata")

        meta_json = json.dumps(metadata or {}, separators=(",", ":"), sort_keys=True)
        now = time.time()

        with self._lock:
            with self._conn:
                if parent_pid is not None:
                    parent_row = self._conn.execute(
                        "SELECT pid FROM agent_processes WHERE pid = ?",
                        (parent_pid,),
                    ).fetchone()
                    if parent_row is None:
                        raise UnknownPid(parent_pid)
                cur = self._conn.execute(
                    """
                    INSERT INTO agent_processes
                        (parent_pid, name, template, state, state_reason,
                         created_at, updated_at, metadata)
                    VALUES (?, ?, ?, 'READY', NULL, ?, ?, ?)
                    """,
                    (parent_pid, name, template, now, now, meta_json),
                )
                pid = cur.lastrowid
                event_id = event_log.append_event(
                    self._conn,
                    pid=pid,
                    kind=EV_PROCESS_CREATED,
                    payload={
                        "name": name,
                        "template": template,
                        "parent_pid": parent_pid,
                        "metadata": metadata or {},
                    },
                    ts=now,
                )
                # Re-read to get the now-bumped last_event_id.
                row = self._conn.execute(
                    "SELECT * FROM agent_processes WHERE pid = ?", (pid,),
                ).fetchone()
                agent = _row_to_agent(row)

        if self._bus is not None:
            self._bus.publish(EV_PROCESS_CREATED, {
                "pid": pid,
                "parent_pid": parent_pid,
                "name": name,
                "template": template,
                "state": AgentState.READY,
                "event_id": event_id,
            })
        return agent

    def get(self, pid: int) -> AgentProcess:
        if not isinstance(pid, int):
            raise InvalidPayload("pid must be int", field="pid")
        # Serialise SELECTs through the kernel write lock. Python's
        # sqlite3 with ``check_same_thread=False`` sharing one
        # connection across threads has fragile transaction-state
        # semantics under concurrent SELECT load — one thread's
        # in-flight transaction can make another thread's SELECT
        # return phantom-empty results. Serialising at the Python
        # level kills that race at minor read-throughput cost.
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM agent_processes WHERE pid = ?", (pid,),
            ).fetchone()
        if row is None:
            raise UnknownPid(pid)
        return _row_to_agent(row)

    def list(
        self,
        *,
        state: Optional[str] = None,
        parent_pid: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[AgentProcess], int]:
        if state is not None and state not in AgentState.ALL:
            raise InvalidPayload(
                f"state must be one of {AgentState.ALL} or null", field="state",
            )
        if not isinstance(limit, int) or limit < 1:
            limit = 100
        elif limit > 10_000:
            limit = 10_000
        if not isinstance(offset, int) or offset < 0:
            offset = 0

        # Build WHERE clauses
        where = []
        params: list[Any] = []
        if state is not None:
            where.append("state = ?")
            params.append(state)
        if parent_pid is not None:
            where.append("parent_pid = ?")
            params.append(parent_pid)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        total_row = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM agent_processes{where_sql}", params,
        ).fetchone()
        total = int(total_row["n"]) if isinstance(total_row, sqlite3.Row) else int(total_row[0])

        rows = self._conn.execute(
            f"SELECT * FROM agent_processes{where_sql} "
            "ORDER BY pid ASC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return [_row_to_agent(r) for r in rows], total

    # ── State transitions ─────────────────────────────────────────────

    def transition(
        self,
        pid: int,
        target_state: str,
        *,
        reason: Optional[str] = None,
    ) -> tuple[str, str, int]:
        """Drive a state transition. Returns (prev_state, new_state, event_id).

        ``terminate`` is preferred for moves into DEAD because it also
        records exit_kind. Calling ``transition(pid, "DEAD")`` works but
        leaves exit_kind NULL.
        """
        if target_state not in AgentState.ALL:
            raise InvalidPayload(
                f"target_state must be one of {AgentState.ALL}",
                field="target_state",
            )
        return self._do_transition(pid, target_state, reason=reason, exit_kind=None,
                                    exit_detail=None)

    def terminate(
        self,
        pid: int,
        *,
        exit_kind: str,
        exit_detail: Optional[dict] = None,
    ) -> tuple[str, int]:
        """Move an agent to DEAD with an exit_kind. Returns (prev_state, event_id)."""
        if exit_kind not in ("completed", "cancelled", "failed", "crashed"):
            raise InvalidPayload(
                "exit_kind must be one of "
                "'completed','cancelled','failed','crashed'",
                field="exit_kind",
            )
        prev, new, event_id = self._do_transition(
            pid, AgentState.DEAD, reason=None,
            exit_kind=exit_kind, exit_detail=exit_detail,
        )
        # `new` is always DEAD here.
        del new
        return prev, event_id

    def _do_transition(
        self,
        pid: int,
        target_state: str,
        *,
        reason: Optional[str],
        exit_kind: Optional[str],
        exit_detail: Optional[dict],
    ) -> tuple[str, str, int]:
        if not isinstance(pid, int):
            raise InvalidPayload("pid must be int", field="pid")
        if exit_detail is not None and not isinstance(exit_detail, dict):
            raise InvalidPayload("exit_detail must be an object", field="exit_detail")

        published: Optional[tuple[str, str, int]] = None
        kind: str

        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT * FROM agent_processes WHERE pid = ?", (pid,),
                ).fetchone()
                if row is None:
                    raise UnknownPid(pid)
                prev_state = row["state"]
                if not is_transition_allowed(prev_state, target_state):
                    raise IllegalTransition(prev_state, target_state)
                now = time.time()
                started_at = row["started_at"]
                ended_at = row["ended_at"]
                if target_state == AgentState.RUNNING and started_at is None:
                    started_at = now
                if target_state == AgentState.DEAD and ended_at is None:
                    ended_at = now
                detail_json = (
                    json.dumps(exit_detail, separators=(",", ":"), sort_keys=True)
                    if exit_detail is not None else None
                )
                self._conn.execute(
                    """
                    UPDATE agent_processes SET
                        state = ?, state_reason = ?,
                        updated_at = ?, started_at = ?, ended_at = ?,
                        exit_kind = COALESCE(?, exit_kind),
                        exit_detail = COALESCE(?, exit_detail)
                    WHERE pid = ?
                    """,
                    (target_state, reason, now, started_at, ended_at,
                     exit_kind, detail_json, pid),
                )
                if target_state == AgentState.DEAD:
                    kind = EV_PROCESS_TERMINATED
                    payload = {
                        "prev_state": prev_state,
                        "new_state": target_state,
                        "reason": reason,
                        "exit_kind": exit_kind,
                        "exit_detail": exit_detail,
                    }
                else:
                    kind = EV_PROCESS_TRANSITIONED
                    payload = {
                        "prev_state": prev_state,
                        "new_state": target_state,
                        "reason": reason,
                    }
                event_id = event_log.append_event(
                    self._conn, pid=pid, kind=kind, payload=payload, ts=now,
                )
                published = (prev_state, target_state, event_id)

        if self._bus is not None and published is not None:
            prev, new, eid = published
            self._bus.publish(kind, {
                "pid": pid,
                "prev_state": prev,
                "new_state": new,
                "reason": reason,
                "exit_kind": exit_kind,
                "event_id": eid,
            })
        return published  # type: ignore[return-value]

    # ── Event read/append ─────────────────────────────────────────────

    def events_append(
        self,
        *,
        pid: int,
        kind: str,
        payload: dict,
        causation_id: Optional[int] = None,
        correlation_id: Optional[str] = None,
    ) -> int:
        if not isinstance(pid, int):
            raise InvalidPayload("pid must be int", field="pid")
        if not isinstance(kind, str) or not kind:
            raise InvalidPayload("kind must be a non-empty string", field="kind")
        if kind.startswith("kernel."):
            raise InvalidPayload(
                "the 'kernel.' event-kind prefix is reserved",
                field="kind",
            )
        if not isinstance(payload, dict):
            raise InvalidPayload("payload must be an object", field="payload")

        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT pid FROM agent_processes WHERE pid = ?", (pid,),
                ).fetchone()
                if row is None:
                    raise UnknownPid(pid)
                event_id = event_log.append_event(
                    self._conn, pid=pid, kind=kind, payload=payload,
                    causation_id=causation_id, correlation_id=correlation_id,
                )

        if self._bus is not None:
            self._bus.publish(kind, {
                "pid": pid,
                "event_id": event_id,
                "payload": payload,
                "causation_id": causation_id,
                "correlation_id": correlation_id,
            })
        return event_id

    def events_tail(
        self,
        *,
        pid: Optional[int] = None,
        kind: Optional[str] = None,
        since_event_id: int = 0,
        limit: int = 100,
    ) -> list[Event]:
        if pid is not None and not isinstance(pid, int):
            raise InvalidPayload("pid must be int or null", field="pid")
        if not isinstance(since_event_id, int) or since_event_id < 0:
            since_event_id = 0
        return event_log.read_events(
            self._conn,
            pid=pid, kind=kind,
            since_event_id=since_event_id, limit=limit,
        )

    # ── Info ──────────────────────────────────────────────────────────

    def info(self) -> dict:
        agent_count_row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM agent_processes"
        ).fetchone()
        agent_count = int(agent_count_row["n"]) if isinstance(agent_count_row, sqlite3.Row) else int(agent_count_row[0])
        event_count = event_log.count_events(self._conn)
        max_evid = event_log.max_event_id(self._conn)
        max_pid_row = self._conn.execute(
            "SELECT COALESCE(MAX(pid), 0) AS m FROM agent_processes"
        ).fetchone()
        max_pid = int(max_pid_row["m"]) if isinstance(max_pid_row, sqlite3.Row) else int(max_pid_row[0])
        return {
            "schema_version":   EXPECTED_SCHEMA_VERSION,
            "next_pid":         max_pid + 1,
            "next_event_id":    max_evid + 1,
            "agent_count":      agent_count,
            "event_count":      event_count,
            "live_states": {
                s: self._count_state(s) for s in AgentState.LIVE
            },
        }

    def _count_state(self, state: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM agent_processes WHERE state = ?",
            (state,),
        ).fetchone()
        return int(row["n"]) if isinstance(row, sqlite3.Row) else int(row[0])
