"""system_methods.py — Always-available daemon-control RPC methods.

These ride on top of the spike's :mod:`cc_daemon.methods` (which carries
the demo and permission methods) so the contract surface is the same on
day one as it will be once the real ``agent.run`` integration lands.

  ``system.ping``      — returns the literal string ``"pong"``.
                          Same purpose as ``echo.ping`` but matches the
                          method name committed to in RFC 0001 and the
                          F-1 acceptance criteria.

  ``system.shutdown``  — triggers ``DaemonState.shutdown()`` which sets
                          ``shutdown_event``.  The cli.py serve loop
                          watches that event and, on a side thread,
                          invokes ``server.shutdown()`` so the ongoing
                          RPC response can finish writing before the
                          listener tears down.

                          This is the only cross-platform graceful
                          shutdown we have — Windows can't deliver
                          SIGTERM cleanly to another Python process, so
                          relying on signals would force users on Windows
                          to ``TerminateProcess`` (no clean cleanup).

  ``system.status``    — RFC 0002 §F-9.  Returns the four budget keys
                          (``session_token_budget`` /
                          ``session_cost_budget`` /
                          ``daily_token_budget`` / ``daily_cost_budget``)
                          so ``cheetahclaws daemon status`` can confirm
                          the serve-mode defaults are in effect.  Also
                          surfaces the running runner / bridge counts
                          for quick triage.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .rpc import RpcRegistry

if TYPE_CHECKING:
    from .server import DaemonState


_BUDGET_KEYS = (
    "session_token_budget",
    "session_cost_budget",
    "daily_token_budget",
    "daily_cost_budget",
)


def register(registry: RpcRegistry, daemon_state: "DaemonState") -> None:
    def system_ping(_params, _ctx):
        return "pong"

    def system_shutdown(_params, _ctx):
        daemon_state.shutdown()
        return "shutdown_initiated"

    def system_status(_params, _ctx):
        """Return budgets + live runner / bridge counts. Used by
        ``cheetahclaws daemon status`` to confirm serve-mode defaults
        are in effect."""
        cfg = daemon_state.config or {}
        budgets = {k: cfg.get(k) for k in _BUDGET_KEYS}
        # Live counts are best-effort — if the supervisor module can't
        # be imported (rare in tests that exercise system_methods in
        # isolation), report zero rather than failing the whole status
        # call.
        try:
            from . import runner_supervisor as _rs
            runners = len(_rs.list_all())
        except Exception:
            runners = 0
        try:
            from . import bridge_supervisor as _bs
            bridges = len(_bs.list_all())
        except Exception:
            bridges = 0
        return {
            "budgets": budgets,
            "runners": runners,
            "bridges": bridges,
        }

    registry.register("system.ping",     system_ping)
    registry.register("system.shutdown", system_shutdown)
    registry.register("system.status",   system_status)
