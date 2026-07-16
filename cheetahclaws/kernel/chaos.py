"""chaos.py — fault injection primitives for tests (RFC 0012 §6).

Production code MUST NOT import from this module. The chaos test
runner picks operations at random (deterministic given a seed) and
asserts the kernel survives.
"""
from __future__ import annotations

import contextlib
import random
import sqlite3
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .mailbox import MailboxStore
    from .store import KernelStore


class _FaultyConn:
    """One-shot disk-full simulator. Forwards everything to the wrapped
    sqlite3.Connection except ``execute``, where it raises once on the
    next write call. ``__enter__`` / ``__exit__`` proxy through so
    ``with conn:`` blocks still work (they delegate the BEGIN/COMMIT
    on the underlying real connection)."""

    def __init__(self, real, triggered) -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_triggered", triggered)

    def execute(self, sql, *args, **kwargs):
        sql_strip = sql.strip().upper() if isinstance(sql, str) else ""
        is_write = sql_strip.startswith(("INSERT", "UPDATE", "DELETE"))
        if is_write and not self._triggered["fired"]:
            self._triggered["fired"] = True
            raise sqlite3.OperationalError("database or disk is full")
        return self._real.execute(sql, *args, **kwargs)

    def __enter__(self):
        return self._real.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._real.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)


class ChaosMonkey:
    """Deterministic fault injector for kernel tests."""

    def __init__(self, *, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self.events: list[dict] = []  # log of operations performed

    def _record(self, op: str, **detail) -> None:
        self.events.append({"op": op, "ts": time.time(), **detail})

    # ── kill ──────────────────────────────────────────────────────────

    def kill_random_agent(self, kernel_store: "KernelStore") -> Optional[int]:
        """Pick a non-DEAD agent at random and terminate(crashed).
        Returns the killed pid, or None if no live agents exist."""
        agents, _ = kernel_store.list(limit=10_000)
        live = [a for a in agents if a.state != "DEAD"]
        if not live:
            self._record("kill_random_agent", killed=None,
                          reason="no live agents")
            return None
        victim = self._rng.choice(live)
        try:
            kernel_store.terminate(victim.pid, exit_kind="crashed")
        except Exception as e:
            self._record("kill_random_agent", pid=victim.pid,
                          error=type(e).__name__)
            raise
        self._record("kill_random_agent", pid=victim.pid)
        return victim.pid

    # ── fill mailbox ─────────────────────────────────────────────────

    def fill_mailbox(self, mailbox_store: "MailboxStore", pid: int) -> int:
        """Send messages to pid until MailboxFull. Returns count of
        successful sends."""
        from .errors import MailboxFull
        sent = 0
        while True:
            try:
                mailbox_store.send(
                    sender_pid=None, recipient_pid=pid,
                    kind="chaos.fill", payload={"i": sent},
                )
                sent += 1
            except MailboxFull:
                break
        self._record("fill_mailbox", pid=pid, sent=sent)
        return sent

    # ── disk full ────────────────────────────────────────────────────

    @contextlib.contextmanager
    def simulate_disk_full(self, *stores_to_corrupt):
        """Context manager that wraps each store's ``._conn`` with a
        one-shot fault: the next write (INSERT/UPDATE/DELETE) raises
        ``OperationalError('database or disk is full')``. Reads pass
        through untouched.

        Why we wrap instances rather than monkey-patch the class:
        Python 3.13 makes ``sqlite3.Connection`` an immutable C type,
        so attribute assignment fails. A per-instance wrapper gives us
        the same fault-injection capability without touching CPython
        internals.

        Pass any store-like objects with a ``._conn`` attribute holding
        a sqlite3.Connection — KernelStore, MailboxStore, etc.
        """
        triggered = {"fired": False}
        saved: list = []
        for store in stores_to_corrupt:
            original = store._conn
            store._conn = _FaultyConn(original, triggered)
            saved.append((store, original))
        self._record("simulate_disk_full", phase="enter",
                     stores=len(stores_to_corrupt))
        try:
            yield
        finally:
            for store, original in saved:
                store._conn = original
            self._record("simulate_disk_full", phase="exit",
                         fired=triggered["fired"])

    # ── lose event ───────────────────────────────────────────────────

    def lose_event(self, kernel_store: "KernelStore", event_id: int) -> bool:
        """Manually delete an event row. Tests robustness against log
        corruption (which shouldn't happen, but the kernel should
        degrade gracefully)."""
        with kernel_store.write_lock:
            with kernel_store.connection:
                cur = kernel_store.connection.execute(
                    "DELETE FROM agent_events WHERE event_id = ?",
                    (event_id,),
                )
                deleted = (cur.rowcount or 0) > 0
        self._record("lose_event", event_id=event_id, deleted=deleted)
        return deleted

    # ── seed introspection ──────────────────────────────────────────

    def reset(self) -> None:
        self.events.clear()
