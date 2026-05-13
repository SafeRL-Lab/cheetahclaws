"""proactive_methods.py — ``proactive.*`` JSON-RPC methods (RFC 0002 F-5).

Thin wrappers over :mod:`cc_daemon.proactive_state` so external clients
(REPL ``/proactive``, future Web UI, bridge integrations) can manage
the daemon-owned proactive watcher through the same SQLite-backed
settings the scheduler reads from.

Exposed methods:

    proactive.set(enabled: bool, interval_s: int)
        Persist the watcher state. ``interval_s`` rejects values < 1.

    proactive.get()
        Return ``{enabled, interval_s, last_tick_at, scheduler_running}``.

    proactive.tickle()
        Reset ``last_tick_at`` to "now". The REPL / bridges call this
        every time the user interacts so the watcher doesn't fire
        during active conversations.

Same threat model as :mod:`cc_daemon.monitor_methods` (single-user) —
any authenticated caller may invoke these. Per-method authorisation
arrives with the originator-routed RPCs in a later phase.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .rpc import RpcRegistry

if TYPE_CHECKING:
    from .server import DaemonState


def register(registry: RpcRegistry, _daemon_state: "DaemonState") -> None:

    def proactive_set(params: dict, _ctx) -> dict:
        if "enabled" not in params:
            raise TypeError("proactive.set requires 'enabled'")
        enabled = bool(params["enabled"])
        try:
            interval_s = int(params.get("interval_s", 300))
        except (TypeError, ValueError) as e:
            raise TypeError(f"proactive.set: 'interval_s' must be int: {e}")
        from . import proactive_state
        try:
            proactive_state.set_state(
                enabled=enabled, interval_s=interval_s
            )
        except ValueError as e:
            raise TypeError(str(e))
        ena, iv, last = proactive_state.get_state()
        return {"enabled": ena, "interval_s": iv, "last_tick_at": last}

    def proactive_get(_params: dict, _ctx) -> dict:
        from . import proactive_state, proactive_scheduler
        ena, iv, last = proactive_state.get_state()
        return {
            "enabled":            ena,
            "interval_s":         iv,
            "last_tick_at":       last,
            "scheduler_running":  proactive_scheduler.is_running(),
        }

    def proactive_tickle(_params: dict, _ctx) -> dict:
        from . import proactive_state
        proactive_state.tickle()
        ena, iv, last = proactive_state.get_state()
        return {"last_tick_at": last, "enabled": ena}

    registry.register("proactive.set",    proactive_set)
    registry.register("proactive.get",    proactive_get)
    registry.register("proactive.tickle", proactive_tickle)
