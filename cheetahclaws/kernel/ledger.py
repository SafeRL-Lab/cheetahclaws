"""ledger.py — ResourceLedger (RFC 0006).

Per-agent budgets across multiple dimensions. Each dimension has
``used``, ``granted`` (== ``hard_limit``), and ``warn_at``. ``charge``
is atomic and records over-limit usage; the supervisor is responsible
for acting on ``first_breach=true``.

Strictly additive — nothing else in the codebase imports this yet.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .errors import (
    InvalidPayload,
    LedgerExists,
    LedgerInvalidAmount,
    LedgerInvalidRefund,
    LedgerInvalidWarnAt,
    LedgerUnknownDim,
    UnknownPid,
)

if TYPE_CHECKING:
    from cheetahclaws.daemon.rpc import CallContext, RpcRegistry


# Standard dimension names — documented for the supervisor / runtime,
# not enforced by the kernel (custom dims are allowed).
STD_DIMS = ("tokens", "cost_micro", "cpu_s", "wall_s", "tool_calls", "fs_w_bytes")


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LedgerEntry:
    pid:         int
    dim:         str
    used:        int
    granted:     int
    hard_limit:  int
    warn_at:     float

    def to_dict(self) -> dict:
        return {
            "pid":        self.pid,
            "dim":        self.dim,
            "used":       self.used,
            "granted":    self.granted,
            "hard_limit": self.hard_limit,
            "warn_at":    self.warn_at,
        }


@dataclass(frozen=True)
class Ledger:
    pid:     int
    entries: tuple

    def to_dict(self) -> dict:
        return {
            "pid":     self.pid,
            "entries": [e.to_dict() for e in self.entries],
        }


@dataclass(frozen=True)
class ChargeResult:
    pid:          int
    dim:          str
    amount:       int
    used:         int
    granted:      int
    over_limit:   bool
    warned:       bool
    first_breach: bool

    def to_dict(self) -> dict:
        return {
            "pid":          self.pid,
            "dim":          self.dim,
            "amount":       self.amount,
            "used":         self.used,
            "granted":      self.granted,
            "over_limit":   self.over_limit,
            "warned":       self.warned,
            "first_breach": self.first_breach,
        }


@dataclass(frozen=True)
class CheckResult:
    pid:          int
    dim:          str
    used:         int
    granted:      int
    would_use:    int
    would_exceed: bool

    def to_dict(self) -> dict:
        return {
            "pid":          self.pid,
            "dim":          self.dim,
            "used":         self.used,
            "granted":      self.granted,
            "would_use":    self.would_use,
            "would_exceed": self.would_exceed,
        }


def _row_to_entry(row: sqlite3.Row) -> LedgerEntry:
    return LedgerEntry(
        pid        = row["pid"],
        dim        = row["dim"],
        used       = row["used"],
        granted    = row["granted"],
        hard_limit = row["hard_limit"],
        warn_at    = row["warn_at"],
    )


# ── Store ──────────────────────────────────────────────────────────────────


class LedgerStore:
    """SQLite-backed ledger store sharing kernel.db connection AND write
    lock with the other stores. See CapabilityStore docstring for the
    rationale (Python's implicit-transaction model on shared
    connections requires a shared writer lock)."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        write_lock: Optional[threading.Lock] = None,
    ) -> None:
        self._conn = conn
        self._lock = write_lock or threading.Lock()

    # ── Mutations ─────────────────────────────────────────────────────

    def create(
        self,
        *,
        pid: int,
        grants: dict,
        warn_at: float = 0.8,
    ) -> list[str]:
        """Create one row per dim in ``grants``. Returns the list of
        dims successfully created. Raises LedgerExists if any of the
        requested dims already has a row."""
        if not isinstance(pid, int):
            raise InvalidPayload("pid must be int", field="pid")
        if not isinstance(grants, dict) or not grants:
            raise InvalidPayload(
                "grants must be a non-empty {dim: int} object",
                field="grants",
            )
        if not isinstance(warn_at, (int, float)) or not (0.0 <= warn_at <= 1.0):
            raise LedgerInvalidWarnAt(warn_at)

        # Validate every grant value before opening the transaction so
        # we either insert all or insert none.
        cleaned: list[tuple[str, int]] = []
        for dim, value in grants.items():
            if not isinstance(dim, str) or not dim:
                raise InvalidPayload(
                    f"grant key must be a non-empty string, got {dim!r}",
                    field="grants",
                )
            if not isinstance(value, int) or value < 1:
                raise LedgerInvalidAmount(value)
            cleaned.append((dim, value))

        now = time.time()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT pid FROM agent_processes WHERE pid = ?", (pid,),
                ).fetchone()
                if row is None:
                    raise UnknownPid(pid)
                # Pre-check existence so we raise LedgerExists before any
                # insert (atomicity by uniform-failure rather than
                # rollback semantics).
                for dim, _ in cleaned:
                    exists = self._conn.execute(
                        "SELECT 1 FROM agent_ledgers WHERE pid = ? AND dim = ?",
                        (pid, dim),
                    ).fetchone()
                    if exists:
                        raise LedgerExists(pid, dim)
                for dim, value in cleaned:
                    self._conn.execute(
                        """
                        INSERT INTO agent_ledgers
                            (pid, dim, used, granted, hard_limit,
                             warn_at, created_at, updated_at)
                        VALUES (?, ?, 0, ?, ?, ?, ?, ?)
                        """,
                        (pid, dim, value, value, warn_at, now, now),
                    )
        return [d for d, _ in cleaned]

    def charge(self, *, pid: int, dim: str, amount: int) -> ChargeResult:
        if not isinstance(pid, int):
            raise InvalidPayload("pid must be int", field="pid")
        if not isinstance(dim, str) or not dim:
            raise InvalidPayload("dim must be a non-empty string", field="dim")
        if not isinstance(amount, int) or amount < 0:
            raise LedgerInvalidAmount(amount)

        now = time.time()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT used, granted, hard_limit, warn_at "
                    "FROM agent_ledgers WHERE pid = ? AND dim = ?",
                    (pid, dim),
                ).fetchone()
                if row is None:
                    raise LedgerUnknownDim(pid, dim)
                prev_used  = int(row["used"])
                granted    = int(row["granted"])
                hard_limit = int(row["hard_limit"])
                warn_at    = float(row["warn_at"])

                new_used = prev_used + amount
                self._conn.execute(
                    "UPDATE agent_ledgers SET used = ?, updated_at = ? "
                    "WHERE pid = ? AND dim = ?",
                    (new_used, now, pid, dim),
                )

        # Classify outside the lock; it's pure arithmetic.
        prev_over   = prev_used > hard_limit
        now_over    = new_used  > hard_limit
        first_breach = (not prev_over) and now_over

        warn_threshold = warn_at * granted
        prev_warn = prev_used >= warn_threshold
        new_warn  = new_used  >= warn_threshold
        warned    = (not prev_warn) and new_warn

        return ChargeResult(
            pid          = pid,
            dim          = dim,
            amount       = amount,
            used         = new_used,
            granted      = granted,
            over_limit   = now_over,
            warned       = warned,
            first_breach = first_breach,
        )

    def refund(self, *, pid: int, dim: str, amount: int) -> LedgerEntry:
        if not isinstance(pid, int):
            raise InvalidPayload("pid must be int", field="pid")
        if not isinstance(dim, str) or not dim:
            raise InvalidPayload("dim must be a non-empty string", field="dim")
        if not isinstance(amount, int) or amount < 0:
            raise LedgerInvalidAmount(amount)

        now = time.time()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT used FROM agent_ledgers WHERE pid = ? AND dim = ?",
                    (pid, dim),
                ).fetchone()
                if row is None:
                    raise LedgerUnknownDim(pid, dim)
                prev_used = int(row["used"])
                if amount > prev_used:
                    raise LedgerInvalidRefund(pid, dim, prev_used, amount)
                new_used = prev_used - amount
                self._conn.execute(
                    "UPDATE agent_ledgers SET used = ?, updated_at = ? "
                    "WHERE pid = ? AND dim = ?",
                    (new_used, now, pid, dim),
                )
                fetched = self._conn.execute(
                    "SELECT * FROM agent_ledgers WHERE pid = ? AND dim = ?",
                    (pid, dim),
                ).fetchone()
        return _row_to_entry(fetched)

    def update_grant(self, *, pid: int, dim: str, new_grant: int) -> LedgerEntry:
        if not isinstance(pid, int):
            raise InvalidPayload("pid must be int", field="pid")
        if not isinstance(dim, str) or not dim:
            raise InvalidPayload("dim must be a non-empty string", field="dim")
        if not isinstance(new_grant, int) or new_grant < 1:
            raise LedgerInvalidAmount(new_grant)

        now = time.time()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT used FROM agent_ledgers WHERE pid = ? AND dim = ?",
                    (pid, dim),
                ).fetchone()
                if row is None:
                    raise LedgerUnknownDim(pid, dim)
                self._conn.execute(
                    "UPDATE agent_ledgers SET granted = ?, hard_limit = ?, "
                    "updated_at = ? WHERE pid = ? AND dim = ?",
                    (new_grant, new_grant, now, pid, dim),
                )
                fetched = self._conn.execute(
                    "SELECT * FROM agent_ledgers WHERE pid = ? AND dim = ?",
                    (pid, dim),
                ).fetchone()
        return _row_to_entry(fetched)

    # ── Reads ─────────────────────────────────────────────────────────

    def check(self, *, pid: int, dim: str, amount: int) -> CheckResult:
        if not isinstance(pid, int):
            raise InvalidPayload("pid must be int", field="pid")
        if not isinstance(amount, int) or amount < 0:
            raise LedgerInvalidAmount(amount)
        row = self._conn.execute(
            "SELECT used, granted, hard_limit FROM agent_ledgers "
            "WHERE pid = ? AND dim = ?",
            (pid, dim),
        ).fetchone()
        if row is None:
            raise LedgerUnknownDim(pid, dim)
        prev_used = int(row["used"])
        hard_limit = int(row["hard_limit"])
        would_use = prev_used + amount
        return CheckResult(
            pid          = pid,
            dim          = dim,
            used         = prev_used,
            granted      = int(row["granted"]),
            would_use    = would_use,
            would_exceed = would_use > hard_limit,
        )

    def get(self, pid: int) -> Ledger:
        if not isinstance(pid, int):
            raise InvalidPayload("pid must be int", field="pid")
        rows = self._conn.execute(
            "SELECT * FROM agent_ledgers WHERE pid = ? ORDER BY created_at ASC",
            (pid,),
        ).fetchall()
        return Ledger(pid=pid, entries=tuple(_row_to_entry(r) for r in rows))

    def list_breached(self, limit: int = 100) -> list[LedgerEntry]:
        if not isinstance(limit, int) or limit < 1:
            limit = 100
        elif limit > 10_000:
            limit = 10_000
        rows = self._conn.execute(
            "SELECT * FROM agent_ledgers WHERE used > hard_limit "
            "ORDER BY pid ASC, dim ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_entry(r) for r in rows]


# ── RPC registration ───────────────────────────────────────────────────────


def register(registry: "RpcRegistry", store: LedgerStore) -> None:
    from .errors import KernelError

    def _translate(fn):
        def wrapper(params, ctx):
            try:
                return fn(params, ctx)
            except (InvalidPayload, LedgerInvalidAmount, LedgerInvalidWarnAt) as e:
                raise TypeError(str(e))
            except KernelError as e:
                raise RuntimeError(f"{type(e).__name__}: {e}")
        wrapper.__name__ = fn.__name__
        return wrapper

    @_translate
    def ledger_create(params, ctx):
        pid = _req_int(params, "pid")
        grants = params.get("grants")
        warn_at = params.get("warn_at", 0.8)
        dims = store.create(pid=pid, grants=grants, warn_at=float(warn_at))
        return {"pid": pid, "dims": dims}

    @_translate
    def ledger_charge(params, ctx):
        return store.charge(
            pid    = _req_int(params, "pid"),
            dim    = _req_str(params, "dim"),
            amount = _req_int(params, "amount"),
        ).to_dict()

    @_translate
    def ledger_check(params, ctx):
        return store.check(
            pid    = _req_int(params, "pid"),
            dim    = _req_str(params, "dim"),
            amount = _req_int(params, "amount"),
        ).to_dict()

    @_translate
    def ledger_get(params, ctx):
        return store.get(_req_int(params, "pid")).to_dict()

    @_translate
    def ledger_list_breached(params, ctx):
        limit = params.get("limit", 100)
        if not isinstance(limit, int):
            limit = 100
        return {"entries": [e.to_dict() for e in store.list_breached(limit=limit)]}

    @_translate
    def ledger_refund(params, ctx):
        e = store.refund(
            pid    = _req_int(params, "pid"),
            dim    = _req_str(params, "dim"),
            amount = _req_int(params, "amount"),
        )
        return {"pid": e.pid, "dim": e.dim,
                "used": e.used, "granted": e.granted}

    @_translate
    def ledger_update_grant(params, ctx):
        e = store.update_grant(
            pid       = _req_int(params, "pid"),
            dim       = _req_str(params, "dim"),
            new_grant = _req_int(params, "new_grant"),
        )
        return {"pid": e.pid, "dim": e.dim,
                "granted": e.granted, "used": e.used}

    registry.register("kernel.ledger.create",         ledger_create)
    registry.register("kernel.ledger.charge",         ledger_charge)
    registry.register("kernel.ledger.check",          ledger_check)
    registry.register("kernel.ledger.get",            ledger_get)
    registry.register("kernel.ledger.list_breached",  ledger_list_breached)
    registry.register("kernel.ledger.refund",         ledger_refund)
    registry.register("kernel.ledger.update_grant",   ledger_update_grant)


def _req_int(params: dict, key: str) -> int:
    if key not in params:
        raise InvalidPayload(f"missing required field {key!r}", field=key)
    v = params[key]
    if not isinstance(v, int):
        raise InvalidPayload(f"{key!r} must be int", field=key)
    return v


def _req_str(params: dict, key: str) -> str:
    if key not in params:
        raise InvalidPayload(f"missing required field {key!r}", field=key)
    v = params[key]
    if not isinstance(v, str):
        raise InvalidPayload(f"{key!r} must be str", field=key)
    return v
