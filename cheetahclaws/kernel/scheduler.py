"""scheduler.py — AgentScheduler (RFC 0007).

The unified ready queue for the agent OS. Stores entries in
``agent_schedule_queue``; supervisor workers `claim` next-runnable
entries atomically (one transaction per claim, serialised by the
shared kernel write lock so two workers can't grab the same entry).

The scheduler does **not** drive the agent state machine
(RFC 0003) — its own queue state machine
(`queued / dispatched / completed / expired / cancelled`) is
disjoint. The supervisor reads a claim, transitions the agent
READY → RUNNING via ``KernelStore.transition``, runs the work, and
calls ``scheduler.complete``.

Strictly additive: nothing else in the codebase imports this yet.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

from .errors import (
    InvalidPayload,
    SchedIllegalTransition,
    SchedInvalidPayload,
    SchedUnknownId,
    UnknownPid,
)

if TYPE_CHECKING:
    from cheetahclaws.daemon.rpc import CallContext, RpcRegistry


# Queue states (also referenced from RPC param validation)
S_QUEUED     = "queued"
S_DISPATCHED = "dispatched"
S_COMPLETED  = "completed"
S_EXPIRED    = "expired"
S_CANCELLED  = "cancelled"
ALL_STATES   = (S_QUEUED, S_DISPATCHED, S_COMPLETED, S_EXPIRED, S_CANCELLED)
TERMINAL_STATES = (S_COMPLETED, S_EXPIRED, S_CANCELLED)

# Allowed exit_kinds for `complete` — same vocabulary as agent.terminate.
EXIT_KINDS = ("completed", "cancelled", "failed", "crashed")

# Triggers — open-ended; we document the standard set but accept any
# non-empty string.
STD_TRIGGERS = ("manual", "cron", "event", "resume")


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScheduleSpec:
    pid:         int
    priority:    int           = 0
    runnable_at: float         = 0.0
    deadline:    Optional[float] = None
    trigger:     str           = "manual"
    payload:     dict          = field(default_factory=dict)


@dataclass(frozen=True)
class ReadyEntry:
    sched_id:      int
    pid:           int
    priority:      int
    runnable_at:   float
    deadline:      Optional[float]
    trigger:       str
    payload:       dict
    state:         str
    worker_id:     Optional[str]
    created_at:    float
    dispatched_at: Optional[float]
    completed_at:  Optional[float]
    exit_kind:     Optional[str]

    def to_dict(self) -> dict:
        return {
            "sched_id":      self.sched_id,
            "pid":           self.pid,
            "priority":      self.priority,
            "runnable_at":   self.runnable_at,
            "deadline":      self.deadline,
            "trigger":       self.trigger,
            "payload":       self.payload,
            "state":         self.state,
            "worker_id":     self.worker_id,
            "created_at":    self.created_at,
            "dispatched_at": self.dispatched_at,
            "completed_at":  self.completed_at,
            "exit_kind":     self.exit_kind,
        }


def _row_to_entry(row: sqlite3.Row) -> ReadyEntry:
    payload_raw = row["payload"]
    try:
        payload = json.loads(payload_raw) if payload_raw else {}
    except json.JSONDecodeError:
        payload = {"_raw": payload_raw}
    return ReadyEntry(
        sched_id      = row["sched_id"],
        pid           = row["pid"],
        priority      = row["priority"],
        runnable_at   = row["runnable_at"],
        deadline      = row["deadline"],
        trigger       = row["trigger"],
        payload       = payload,
        state         = row["state"],
        worker_id     = row["worker_id"],
        created_at    = row["created_at"],
        dispatched_at = row["dispatched_at"],
        completed_at  = row["completed_at"],
        exit_kind     = row["exit_kind"],
    )


# ── Store ──────────────────────────────────────────────────────────────────


class SchedulerStore:
    """SQLite-backed ready queue. Shares connection + write lock with
    KernelStore (and Capability/Ledger). See CapabilityStore docstring
    for why a single shared lock is mandatory."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        write_lock: Optional[threading.Lock] = None,
    ) -> None:
        self._conn = conn
        self._lock = write_lock or threading.Lock()

    # ── enqueue ───────────────────────────────────────────────────────

    def enqueue(self, spec: ScheduleSpec) -> int:
        if not isinstance(spec, ScheduleSpec):
            raise SchedInvalidPayload(
                f"spec must be ScheduleSpec, got {type(spec).__name__}",
            )
        if not isinstance(spec.pid, int):
            raise SchedInvalidPayload("pid must be int", field="pid")
        if not isinstance(spec.priority, int):
            raise SchedInvalidPayload("priority must be int", field="priority")
        if not isinstance(spec.runnable_at, (int, float)) or spec.runnable_at < 0:
            raise SchedInvalidPayload(
                "runnable_at must be >= 0", field="runnable_at",
            )
        if spec.deadline is not None:
            if not isinstance(spec.deadline, (int, float)) or spec.deadline < spec.runnable_at:
                raise SchedInvalidPayload(
                    "deadline must be >= runnable_at", field="deadline",
                )
        if not isinstance(spec.trigger, str) or not spec.trigger:
            raise SchedInvalidPayload(
                "trigger must be a non-empty string", field="trigger",
            )
        if not isinstance(spec.payload, dict):
            raise SchedInvalidPayload(
                "payload must be an object", field="payload",
            )

        now = time.time()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT state FROM agent_processes WHERE pid = ?",
                    (spec.pid,),
                ).fetchone()
                if row is None:
                    raise UnknownPid(spec.pid)
                if row["state"] == "DEAD":
                    raise SchedInvalidPayload(
                        f"cannot enqueue work for DEAD agent pid={spec.pid}",
                        field="pid",
                    )
                cur = self._conn.execute(
                    """
                    INSERT INTO agent_schedule_queue
                        (pid, priority, runnable_at, deadline, trigger,
                         payload, state, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)
                    """,
                    (spec.pid, spec.priority, float(spec.runnable_at),
                     None if spec.deadline is None else float(spec.deadline),
                     spec.trigger,
                     json.dumps(spec.payload, sort_keys=True, separators=(",", ":")),
                     now),
                )
                sched_id = cur.lastrowid
        return int(sched_id)

    # ── claim ─────────────────────────────────────────────────────────

    def claim(
        self,
        *,
        worker_id: str,
        max_n: int = 1,
        now: Optional[float] = None,
        admission_check: bool = True,
    ) -> list[ReadyEntry]:
        if not isinstance(worker_id, str) or not worker_id:
            raise SchedInvalidPayload(
                "worker_id must be a non-empty string", field="worker_id",
            )
        if not isinstance(max_n, int) or max_n < 1:
            raise SchedInvalidPayload(
                "max_n must be a positive int", field="max_n",
            )
        if max_n > 1000:
            max_n = 1000
        if now is None:
            now = time.time()

        # Build the SELECT. The composite index
        # idx_sched_queue_pickable serves it.
        sql_parts = [
            "SELECT * FROM agent_schedule_queue ",
            "WHERE state = 'queued' ",
            "  AND runnable_at <= ? ",
            "  AND (deadline IS NULL OR deadline > ?) ",
        ]
        params: list = [now, now]
        if admission_check:
            # Skip entries whose owning agent has any over-limit ledger
            # row. Subquery is on the small `agent_ledgers` table.
            sql_parts.append(
                " AND NOT EXISTS ("
                "   SELECT 1 FROM agent_ledgers"
                "   WHERE agent_ledgers.pid = agent_schedule_queue.pid"
                "     AND agent_ledgers.used > agent_ledgers.hard_limit"
                ")"
            )
        sql_parts.append(
            "ORDER BY priority DESC, runnable_at ASC, sched_id ASC "
            "LIMIT ?"
        )
        params.append(max_n)
        select_sql = " ".join(sql_parts)

        claimed: list[ReadyEntry] = []
        with self._lock:
            with self._conn:
                rows = self._conn.execute(select_sql, params).fetchall()
                for r in rows:
                    self._conn.execute(
                        "UPDATE agent_schedule_queue "
                        "SET state = 'dispatched', "
                        "    worker_id = ?, "
                        "    dispatched_at = ? "
                        "WHERE sched_id = ? AND state = 'queued'",
                        (worker_id, now, r["sched_id"]),
                    )
                # Re-read to capture the post-UPDATE state cleanly.
                if rows:
                    ids = [r["sched_id"] for r in rows]
                    placeholders = ",".join("?" * len(ids))
                    refreshed = self._conn.execute(
                        f"SELECT * FROM agent_schedule_queue "
                        f"WHERE sched_id IN ({placeholders})",
                        ids,
                    ).fetchall()
                    by_id = {r["sched_id"]: r for r in refreshed}
                    for sid in ids:
                        claimed.append(_row_to_entry(by_id[sid]))
        return claimed

    # ── complete ──────────────────────────────────────────────────────

    def complete(
        self, sched_id: int, *, exit_kind: str,
    ) -> tuple[str, str]:
        """Mark a dispatched entry as completed. Returns
        (prev_state, new_state). Raises SchedIllegalTransition if the
        entry isn't in 'dispatched' state."""
        if not isinstance(sched_id, int):
            raise SchedInvalidPayload("sched_id must be int", field="sched_id")
        if exit_kind not in EXIT_KINDS:
            raise SchedInvalidPayload(
                f"exit_kind must be one of {EXIT_KINDS}",
                field="exit_kind",
            )
        now = time.time()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT state FROM agent_schedule_queue WHERE sched_id = ?",
                    (sched_id,),
                ).fetchone()
                if row is None:
                    raise SchedUnknownId(sched_id)
                prev = row["state"]
                if prev != S_DISPATCHED:
                    raise SchedIllegalTransition(prev, "complete")
                self._conn.execute(
                    "UPDATE agent_schedule_queue "
                    "SET state = 'completed', "
                    "    completed_at = ?, "
                    "    exit_kind = ? "
                    "WHERE sched_id = ?",
                    (now, exit_kind, sched_id),
                )
        return prev, S_COMPLETED

    # ── cancel ────────────────────────────────────────────────────────

    def cancel(self, sched_id: int) -> tuple[str, str]:
        if not isinstance(sched_id, int):
            raise SchedInvalidPayload("sched_id must be int", field="sched_id")
        now = time.time()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT state FROM agent_schedule_queue WHERE sched_id = ?",
                    (sched_id,),
                ).fetchone()
                if row is None:
                    raise SchedUnknownId(sched_id)
                prev = row["state"]
                if prev != S_QUEUED:
                    # Cancellation of a dispatched entry is supervisor's
                    # job (it must kill the work and call complete()).
                    raise SchedIllegalTransition(prev, "cancel")
                self._conn.execute(
                    "UPDATE agent_schedule_queue "
                    "SET state = 'cancelled', completed_at = ? "
                    "WHERE sched_id = ?",
                    (now, sched_id),
                )
        return prev, S_CANCELLED

    # ── gc_expired ────────────────────────────────────────────────────

    def gc_expired(self, now: Optional[float] = None) -> int:
        """Sweep queued entries with deadline < now into 'expired'.
        Returns count swept."""
        if now is None:
            now = time.time()
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    "UPDATE agent_schedule_queue "
                    "SET state = 'expired', completed_at = ? "
                    "WHERE state = 'queued' "
                    "  AND deadline IS NOT NULL "
                    "  AND deadline < ?",
                    (now, now),
                )
                return int(cur.rowcount or 0)

    # ── reads ─────────────────────────────────────────────────────────

    def get(self, sched_id: int) -> ReadyEntry:
        if not isinstance(sched_id, int):
            raise SchedInvalidPayload("sched_id must be int", field="sched_id")
        row = self._conn.execute(
            "SELECT * FROM agent_schedule_queue WHERE sched_id = ?",
            (sched_id,),
        ).fetchone()
        if row is None:
            raise SchedUnknownId(sched_id)
        return _row_to_entry(row)

    def list(
        self,
        *,
        state: Optional[str] = None,
        pid:   Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[ReadyEntry], int]:
        if state is not None and state not in ALL_STATES:
            raise SchedInvalidPayload(
                f"state must be one of {ALL_STATES} or null",
                field="state",
            )
        if not isinstance(limit, int) or limit < 1:
            limit = 100
        elif limit > 10_000:
            limit = 10_000
        if not isinstance(offset, int) or offset < 0:
            offset = 0

        where, params = [], []
        if state is not None:
            where.append("state = ?")
            params.append(state)
        if pid is not None:
            where.append("pid = ?")
            params.append(pid)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        total_row = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM agent_schedule_queue{where_sql}",
            params,
        ).fetchone()
        total = int(total_row["n"]) if isinstance(total_row, sqlite3.Row) else int(total_row[0])

        rows = self._conn.execute(
            f"SELECT * FROM agent_schedule_queue{where_sql} "
            "ORDER BY sched_id ASC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return [_row_to_entry(r) for r in rows], total


# ── RPC registration ───────────────────────────────────────────────────────


def register(registry: "RpcRegistry", store: SchedulerStore) -> None:
    from .errors import KernelError

    def _translate(fn):
        def wrapper(params, ctx):
            try:
                return fn(params, ctx)
            except (InvalidPayload, SchedInvalidPayload) as e:
                raise TypeError(str(e))
            except KernelError as e:
                raise RuntimeError(f"{type(e).__name__}: {e}")
        wrapper.__name__ = fn.__name__
        return wrapper

    @_translate
    def sched_enqueue(params, ctx):
        spec = ScheduleSpec(
            pid          = _req_int(params, "pid"),
            priority     = int(params.get("priority", 0)),
            runnable_at  = float(params.get("runnable_at", 0.0)),
            deadline     = (None if params.get("deadline") is None
                            else float(params["deadline"])),
            trigger      = str(params.get("trigger", "manual")),
            payload      = params.get("payload", {}) or {},
        )
        sched_id = store.enqueue(spec)
        return {"sched_id": sched_id}

    @_translate
    def sched_claim(params, ctx):
        worker_id = _req_str(params, "worker_id")
        max_n = int(params.get("max_n", 1))
        now = params.get("now")
        if now is not None:
            now = float(now)
        admission = bool(params.get("admission_check", True))
        entries = store.claim(
            worker_id=worker_id, max_n=max_n,
            now=now, admission_check=admission,
        )
        return {"entries": [e.to_dict() for e in entries]}

    @_translate
    def sched_complete(params, ctx):
        sched_id = _req_int(params, "sched_id")
        exit_kind = _req_str(params, "exit_kind")
        prev, new = store.complete(sched_id, exit_kind=exit_kind)
        return {"sched_id": sched_id, "prev_state": prev, "state": new}

    @_translate
    def sched_cancel(params, ctx):
        sched_id = _req_int(params, "sched_id")
        prev, new = store.cancel(sched_id)
        return {"sched_id": sched_id, "prev_state": prev, "state": new}

    @_translate
    def sched_get(params, ctx):
        sched_id = _req_int(params, "sched_id")
        return store.get(sched_id).to_dict()

    @_translate
    def sched_list(params, ctx):
        state  = params.get("state")
        pid    = params.get("pid")
        limit  = int(params.get("limit", 100))
        offset = int(params.get("offset", 0))
        if pid is not None and not isinstance(pid, int):
            raise SchedInvalidPayload("pid must be int or null", field="pid")
        entries, total = store.list(
            state=state, pid=pid, limit=limit, offset=offset,
        )
        return {
            "entries": [e.to_dict() for e in entries],
            "total":   total,
        }

    @_translate
    def sched_gc_expired(params, ctx):
        now = params.get("now")
        if now is not None:
            now = float(now)
        return {"swept": store.gc_expired(now=now)}

    registry.register("kernel.sched.enqueue",    sched_enqueue)
    registry.register("kernel.sched.claim",      sched_claim)
    registry.register("kernel.sched.complete",   sched_complete)
    registry.register("kernel.sched.cancel",     sched_cancel)
    registry.register("kernel.sched.get",        sched_get)
    registry.register("kernel.sched.list",       sched_list)
    registry.register("kernel.sched.gc_expired", sched_gc_expired)


def _req_int(params: dict, key: str) -> int:
    if key not in params:
        raise SchedInvalidPayload(f"missing required field {key!r}", field=key)
    v = params[key]
    if not isinstance(v, int):
        raise SchedInvalidPayload(f"{key!r} must be int", field=key)
    return v


def _req_str(params: dict, key: str) -> str:
    if key not in params:
        raise SchedInvalidPayload(f"missing required field {key!r}", field=key)
    v = params[key]
    if not isinstance(v, str):
        raise SchedInvalidPayload(f"{key!r} must be str", field=key)
    return v
