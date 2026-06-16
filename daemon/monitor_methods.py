"""monitor_methods.py — `monitor.*` JSON-RPC methods.

Thin wrappers over :mod:`monitor.store` and :mod:`monitor.scheduler` so
external clients (Web UI, third-party tools) can manage subscriptions
and trigger runs through the same SQLite store that the in-process
scheduler reads from.

Exposed methods:

    monitor.subscribe(topic, schedule="daily", channels=None)
        Add or update a subscription.  Returns the new subscription dict.

    monitor.unsubscribe(topic)
        Remove a subscription.  Returns {"removed": bool}.

    monitor.list()
        Return all subscriptions.

    monitor.run(topic)
        Force-run a subscription now (fetch + summarize + deliver).
        Returns {"topic", "report"}.  The report is also persisted to
        ``monitor_reports`` and a ``monitor_report`` event fires on the
        SSE channel — see :func:`monitor.scheduler.run_one`.

F-3 keeps these methods open to any authenticated caller (single-user
threat model from RFC 0001 §3).  Per-method authorisation arrives with
the agent.run integration in F-3+ when permission requests carry an
originator.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .rpc import RpcRegistry

if TYPE_CHECKING:
    from .server import DaemonState


def register(registry: RpcRegistry, daemon_state: "DaemonState") -> None:

    def monitor_subscribe(params: dict, _ctx) -> dict:
        topic = params.get("topic")
        if not isinstance(topic, str) or not topic:
            raise TypeError("monitor.subscribe requires non-empty 'topic'")
        schedule = params.get("schedule") or "daily"
        channels = params.get("channels")
        if channels is not None and not isinstance(channels, list):
            raise TypeError("'channels' must be a list of strings")
        from monitor.store import add_subscription
        sub = add_subscription(topic, schedule=schedule, channels=channels)
        return sub

    def monitor_unsubscribe(params: dict, _ctx) -> dict:
        topic = params.get("topic")
        if not isinstance(topic, str) or not topic:
            raise TypeError("monitor.unsubscribe requires non-empty 'topic'")
        from monitor.store import remove_subscription
        return {"topic": topic, "removed": remove_subscription(topic)}

    def monitor_list(_params: dict, _ctx) -> dict:
        from monitor.store import list_subscriptions
        return {"subscriptions": list_subscriptions()}

    def monitor_run(params: dict, _ctx) -> dict:
        topic = params.get("topic")
        if not isinstance(topic, str) or not topic:
            raise TypeError("monitor.run requires non-empty 'topic'")
        from monitor.scheduler import run_one
        report = run_one(topic, config=daemon_state.config or {})
        return {"topic": topic, "report": report}

    registry.register("monitor.subscribe", monitor_subscribe)
    registry.register("monitor.unsubscribe", monitor_unsubscribe)
    registry.register("monitor.list", monitor_list)
    registry.register("monitor.run", monitor_run)
