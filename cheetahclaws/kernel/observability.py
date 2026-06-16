"""observability.py — kernel introspection + Prometheus + trace (RFC 0012).

Read-only aggregation across all kernel stores. No new tables; every
view is computed on demand from existing data. Cheap reads via SQLite
WAL.
"""
from __future__ import annotations

import sqlite3
import time
from typing import TYPE_CHECKING, Optional

from . import event_log
from .errors import InvalidPayload, UnknownPid
from .process import AgentState
from .schema import EXPECTED_SCHEMA_VERSION

if TYPE_CHECKING:
    from cheetahclaws.daemon.rpc import CallContext, RpcRegistry
    from .agentfs import AgentFSStore
    from .capability import CapabilityStore
    from .ledger import LedgerStore
    from .mailbox import MailboxStore
    from .registry import RegistryStore
    from .scheduler import SchedulerStore
    from .store import KernelStore


TRACE_DEPTH_MAX = 100
RECENT_EVENTS_FOR_PROC = 20


# ── Store ──────────────────────────────────────────────────────────────────


class ObservabilityStore:
    """Read-only aggregator. Holds references to every other store so
    it can build cross-store views in a single API call."""

    def __init__(
        self,
        *,
        kernel_store:     "KernelStore",
        capability_store: "CapabilityStore",
        ledger_store:     "LedgerStore",
        scheduler_store:  "SchedulerStore",
        mailbox_store:    "MailboxStore",
        registry_store:   "RegistryStore",
        agentfs_store:    "AgentFSStore",
    ) -> None:
        self._kernel  = kernel_store
        self._cap     = capability_store
        self._ledger  = ledger_store
        self._sched   = scheduler_store
        self._mbox    = mailbox_store
        self._reg     = registry_store
        self._fs      = agentfs_store
        self._conn    = kernel_store.connection
        self._started_at = time.time()

    # ── proc ──────────────────────────────────────────────────────────

    def proc(self, pid: int) -> dict:
        if not isinstance(pid, int):
            raise InvalidPayload("pid must be int", field="pid")

        # Process row.
        try:
            agent = self._kernel.get(pid)
            process_view = agent.to_dict()
        except UnknownPid:
            process_view = None

        # Capability.
        try:
            cap = self._cap.get(pid)
            cap_view = cap.to_dict()
        except Exception:
            cap_view = None

        # Ledger.
        ledger_view = [e.to_dict() for e in self._ledger.get(pid).entries]

        # Mailbox.
        mb_row = self._conn.execute(
            "SELECT queue_size FROM agent_mailboxes WHERE pid = ?", (pid,),
        ).fetchone()
        if mb_row is None:
            mailbox_view = {
                "exists": False, "queue_size": None, "pending": None,
                "subscriptions": [],
            }
        else:
            pending_row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM agent_messages "
                "WHERE recipient_pid = ? AND delivered_at IS NULL",
                (pid,),
            ).fetchone()
            mailbox_view = {
                "exists":      True,
                "queue_size":  int(mb_row["queue_size"]),
                "pending":     int(pending_row["n"]),
                "subscriptions": self._mbox.list_subscriptions(pid),
            }

        # Scheduler — counts per state for this pid.
        sched_view = {s: 0 for s in
                      ("queued", "dispatched", "completed",
                       "expired", "cancelled")}
        for r in self._conn.execute(
            "SELECT state, COUNT(*) AS n FROM agent_schedule_queue "
            "WHERE pid = ? GROUP BY state",
            (pid,),
        ):
            sched_view[r["state"]] = int(r["n"])

        # AgentFS — total objects + bytes for this pid.
        fs_row = self._conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(size), 0) AS b "
            "FROM agent_fs_objects WHERE owner_pid = ?",
            (pid,),
        ).fetchone()
        fs_view = {
            "object_count": int(fs_row["n"]),
            "total_bytes":  int(fs_row["b"]),
        }

        # Registry — names that point at this pid.
        reg_view = {
            "names": [r["name"] for r in self._conn.execute(
                "SELECT name FROM agent_registry WHERE pid = ? "
                "ORDER BY name ASC",
                (pid,),
            ).fetchall()]
        }

        # Recent events (last 20).
        recent_rows = self._conn.execute(
            "SELECT * FROM agent_events WHERE pid = ? "
            "ORDER BY event_id DESC LIMIT ?",
            (pid, RECENT_EVENTS_FOR_PROC),
        ).fetchall()
        recent_events = [event_log._row_to_event(r).to_dict()
                         for r in recent_rows]
        # Reverse to chronological order.
        recent_events.reverse()

        return {
            "process":       process_view,
            "capability":    cap_view,
            "ledger":        ledger_view,
            "mailbox":       mailbox_view,
            "scheduler":     sched_view,
            "fs":            fs_view,
            "registry":      reg_view,
            "recent_events": recent_events,
        }

    # ── summary ───────────────────────────────────────────────────────

    def summary(self) -> dict:
        from . import KERNEL_VERSION

        # Agents per state.
        agents = {s: 0 for s in AgentState.ALL}
        for r in self._conn.execute(
            "SELECT state, COUNT(*) AS n FROM agent_processes GROUP BY state"
        ):
            agents[r["state"]] = int(r["n"])
        agents["total"] = sum(agents.values())

        # Events.
        ev_total = self._conn.execute(
            "SELECT COUNT(*) AS n FROM agent_events"
        ).fetchone()["n"]
        ev_max = self._conn.execute(
            "SELECT COALESCE(MAX(event_id), 0) AS m FROM agent_events"
        ).fetchone()["m"]

        # Scheduler.
        sched = {s: 0 for s in
                 ("queued", "dispatched", "completed", "expired", "cancelled")}
        for r in self._conn.execute(
            "SELECT state, COUNT(*) AS n FROM agent_schedule_queue GROUP BY state"
        ):
            sched[r["state"]] = int(r["n"])

        # Mailboxes.
        mb_count = self._conn.execute(
            "SELECT COUNT(*) AS n FROM agent_mailboxes"
        ).fetchone()["n"]
        mb_pending = self._conn.execute(
            "SELECT COUNT(*) AS n FROM agent_messages WHERE delivered_at IS NULL"
        ).fetchone()["n"]

        # Ledger.
        led_agents = self._conn.execute(
            "SELECT COUNT(DISTINCT pid) AS n FROM agent_ledgers"
        ).fetchone()["n"]
        led_breached = self._conn.execute(
            "SELECT COUNT(DISTINCT pid) AS n FROM agent_ledgers "
            "WHERE used > hard_limit"
        ).fetchone()["n"]

        # FS.
        fs_row = self._conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(size), 0) AS b "
            "FROM agent_fs_objects"
        ).fetchone()

        # Registry.
        reg_count = self._conn.execute(
            "SELECT COUNT(*) AS n FROM agent_registry"
        ).fetchone()["n"]

        return {
            "kernel_version":  KERNEL_VERSION,
            "schema_version":  EXPECTED_SCHEMA_VERSION,
            "started_at":      self._started_at,
            "uptime_s":        time.time() - self._started_at,
            "agents":          agents,
            "events":          {"total": int(ev_total),
                                "max_event_id": int(ev_max)},
            "scheduler":       sched,
            "mailbox":         {"mailboxes": int(mb_count),
                                "pending_messages": int(mb_pending)},
            "ledger":          {"agents_with_budgets": int(led_agents),
                                "breached": int(led_breached)},
            "fs":              {"objects": int(fs_row["n"]),
                                "total_bytes": int(fs_row["b"])},
            "registry":        {"entries": int(reg_count)},
        }

    # ── trace ─────────────────────────────────────────────────────────

    def trace(self, event_id: int, depth: int = 10) -> dict:
        if not isinstance(event_id, int):
            raise InvalidPayload("event_id must be int", field="event_id")
        if not isinstance(depth, int) or depth < 1:
            depth = 10
        elif depth > TRACE_DEPTH_MAX:
            depth = TRACE_DEPTH_MAX

        chain = []
        truncated = False
        cur_id: Optional[int] = event_id
        for _ in range(depth):
            if cur_id is None:
                break
            row = self._conn.execute(
                "SELECT * FROM agent_events WHERE event_id = ?",
                (cur_id,),
            ).fetchone()
            if row is None:
                # Corrupt log — referenced cause missing.
                truncated = True
                break
            ev = event_log._row_to_event(row)
            chain.append(ev.to_dict())
            cur_id = ev.causation_id
        else:
            # for-else: hit depth limit without breaking
            if cur_id is not None:
                truncated = True

        return {
            "events":    chain,
            "depth":     len(chain),
            "truncated": truncated,
        }

    # ── prometheus_text ───────────────────────────────────────────────

    def prometheus_text(self) -> str:
        s = self.summary()
        lines: list[str] = []

        def emit(name: str, help_text: str, mtype: str,
                 entries: list[tuple[dict, float]]) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {mtype}")
            for labels, value in entries:
                if labels:
                    label_str = "{" + ",".join(
                        f'{k}="{_esc_label(v)}"'
                        for k, v in sorted(labels.items())
                    ) + "}"
                else:
                    label_str = ""
                lines.append(f"{name}{label_str} {value}")

        emit("cheetahclaws_kernel_schema_version",
             "Kernel schema version", "gauge",
             [({}, s["schema_version"])])
        emit("cheetahclaws_kernel_uptime_seconds",
             "Seconds since kernel registered", "counter",
             [({}, round(s["uptime_s"], 3))])

        agent_entries = [
            ({"state": k}, v)
            for k, v in s["agents"].items() if k != "total"
        ]
        emit("cheetahclaws_kernel_agents",
             "Number of agents by state", "gauge",
             agent_entries)

        emit("cheetahclaws_kernel_events_total",
             "Lifetime event count", "counter",
             [({}, s["events"]["total"])])

        sched_entries = [
            ({"state": k}, v) for k, v in s["scheduler"].items()
        ]
        emit("cheetahclaws_kernel_scheduler_queue",
             "Items in scheduler queue by state", "gauge",
             sched_entries)

        emit("cheetahclaws_kernel_ledger_breached",
             "Agents with at least one breached dim", "gauge",
             [({}, s["ledger"]["breached"])])
        emit("cheetahclaws_kernel_mailbox_pending",
             "Pending messages across all inboxes", "gauge",
             [({}, s["mailbox"]["pending_messages"])])
        emit("cheetahclaws_kernel_fs_objects",
             "Total AgentFS objects", "gauge",
             [({}, s["fs"]["objects"])])
        emit("cheetahclaws_kernel_fs_bytes",
             "Total AgentFS bytes stored", "gauge",
             [({}, s["fs"]["total_bytes"])])
        emit("cheetahclaws_kernel_registry_entries",
             "AgentRegistry entries", "gauge",
             [({}, s["registry"]["entries"])])

        return "\n".join(lines) + "\n"


def _esc_label(value) -> str:
    """Prometheus label-value escaping: backslash, double quote, newline."""
    if not isinstance(value, str):
        value = str(value)
    return (value.replace("\\", "\\\\")
                  .replace('"', '\\"')
                  .replace("\n", "\\n"))


# ── RPC registration ───────────────────────────────────────────────────────


def register(registry: "RpcRegistry", store: ObservabilityStore) -> None:
    from .errors import KernelError

    def _translate(fn):
        def wrapper(params, ctx):
            try:
                return fn(params, ctx)
            except InvalidPayload as e:
                raise TypeError(str(e))
            except KernelError as e:
                raise RuntimeError(f"{type(e).__name__}: {e}")
        wrapper.__name__ = fn.__name__
        return wrapper

    @_translate
    def observe_proc(params, ctx):
        if "pid" not in params:
            raise InvalidPayload("missing pid", field="pid")
        pid = params["pid"]
        if not isinstance(pid, int):
            raise InvalidPayload("pid must be int", field="pid")
        return store.proc(pid)

    @_translate
    def observe_summary(params, ctx):
        return store.summary()

    @_translate
    def observe_trace(params, ctx):
        if "event_id" not in params:
            raise InvalidPayload("missing event_id", field="event_id")
        ev = params["event_id"]
        if not isinstance(ev, int):
            raise InvalidPayload("event_id must be int", field="event_id")
        depth = int(params.get("depth", 10))
        return store.trace(ev, depth)

    @_translate
    def observe_prometheus(params, ctx):
        return {"text": store.prometheus_text()}

    registry.register("kernel.observe.proc",       observe_proc)
    registry.register("kernel.observe.summary",    observe_summary)
    registry.register("kernel.observe.trace",      observe_trace)
    registry.register("kernel.observe.prometheus", observe_prometheus)
