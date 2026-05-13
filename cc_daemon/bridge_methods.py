"""bridge_methods.py — `bridge.*` JSON-RPC methods (RFC 0002 F-6/7/8).

Thin wrappers over :mod:`cc_daemon.bridge_supervisor` so external clients
(REPL `/telegram` command, future Web UI, third-party tools) can manage
daemon-owned bridges through the same RPC channel they use for
``agent.*`` and ``monitor.*``.

Exposed methods:

    bridge.start(kind, config)
        Spawn a bridge worker inside the daemon. ``kind`` is one of
        ``telegram`` / ``slack`` / ``wechat``; ``config`` is the
        per-bridge config dict (tokens, chat_id, etc.). Requires the
        corresponding ``CHEETAHCLAWS_ENABLE_F<6|7|8>`` flag to be set,
        otherwise raises RuntimeError surfaced as -32603 over RPC.

    bridge.stop(kind, timeout_s=5.0)
        Stop a bridge worker. Returns ``{"kind", "stopped": bool}``.

    bridge.list()
        Return all currently-tracked bridges plus any rows persisted in
        the ``bridges`` table from previous runs.

    bridge.send(kind, text)
        Send an outbound text via the running bridge. Used by the F-4
        runner's ``notify`` IPC routing.

    bridge.status(kind)
        Return one bridge's status. ``{"kind", "found": False}`` when
        unknown.

F-6/7/8 keep these methods open to any authenticated caller — same
single-user threat model as F-3's ``monitor.*`` methods. Per-method
authorisation arrives with the per-bridge originator routing in the
Phase 2 inbound refactor.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .rpc import RpcRegistry

if TYPE_CHECKING:
    from .server import DaemonState


def _handle_to_dict(handle) -> dict:
    """Serialise a BridgeHandle for the wire. Drops the thread / stop_event
    references (not JSON-serialisable)."""
    from . import bridge_supervisor as bs
    return {
        "kind":          handle.kind,
        "alive":         handle.is_alive(),
        "started_at":    handle.started_at,
        "last_poll_at":  handle.last_poll_at,
        "last_error":    handle.last_error,
        # Redact secrets before exposing config over the wire.
        "config":        bs._safe_cfg(handle.config),
        # RFC 0002 F-6 Phase 2 surface.
        "daemon_phase2": handle.daemon_phase2,
        "session_id":    handle.session_id() if handle.daemon_phase2 else None,
    }


def register(registry: RpcRegistry, daemon_state: "DaemonState") -> None:

    def bridge_start(params: dict, _ctx) -> dict:
        kind = params.get("kind")
        if not isinstance(kind, str) or not kind:
            raise TypeError("bridge.start requires non-empty 'kind'")
        config = params.get("config", {})
        if not isinstance(config, dict):
            raise TypeError("bridge.start: 'config' must be an object")
        # Merge with the daemon-level config so callers don't have to
        # repeat tokens that are already stored under cc_config.
        merged = dict(daemon_state.config or {})
        merged.update(config)
        # RFC 0002 F-6 Phase 2 — when ``daemon_phase2`` is True, the
        # worker uses the slim daemon-driven loop (session.send event +
        # session_outbound subscriber) instead of the legacy in-REPL
        # supervisor. Defaults to False so existing callers keep Phase 1
        # semantics; opt in with ``{"daemon_phase2": true}`` in the RPC.
        daemon_phase2 = bool(params.get("daemon_phase2", False))
        from . import bridge_supervisor as bs
        handle = bs.start(kind, merged, daemon_phase2=daemon_phase2)
        return _handle_to_dict(handle)

    def bridge_stop(params: dict, _ctx) -> dict:
        kind = params.get("kind")
        if not isinstance(kind, str) or not kind:
            raise TypeError("bridge.stop requires non-empty 'kind'")
        try:
            timeout_s = float(params.get("timeout_s", 5.0))
        except (TypeError, ValueError) as e:
            raise TypeError(f"bridge.stop: 'timeout_s' must be numeric: {e}")
        from . import bridge_supervisor as bs
        return {"kind": kind, "stopped": bs.stop(kind, timeout_s=timeout_s)}

    def bridge_list(_params: dict, _ctx) -> dict:
        from . import bridge_supervisor as bs
        live = [_handle_to_dict(h) for h in bs.list_all()]
        live_kinds = {row["kind"] for row in live}
        # Merge in persisted rows that don't have a live handle (e.g.
        # disabled bridges from a previous run). Callers can tell the
        # two apart via ``alive``.
        persisted: list[dict] = []
        for row in bs.list_persisted():
            if row["kind"] in live_kinds:
                continue
            persisted.append({
                "kind":         row["kind"],
                "alive":        False,
                "started_at":   None,
                "last_poll_at": row["last_poll_at"],
                "last_error":   row["last_error"],
                "config":       row["config"],
            })
        return {"bridges": live + persisted}

    def bridge_send(params: dict, _ctx) -> dict:
        kind = params.get("kind")
        if not isinstance(kind, str) or not kind:
            raise TypeError("bridge.send requires non-empty 'kind'")
        text = params.get("text", "")
        if not isinstance(text, str) or not text:
            raise TypeError("bridge.send requires non-empty 'text'")
        from . import bridge_supervisor as bs
        return {"kind": kind, "delivered": bs.notify(kind, text)}

    def bridge_status(params: dict, _ctx) -> dict:
        kind = params.get("kind")
        if not isinstance(kind, str) or not kind:
            raise TypeError("bridge.status requires non-empty 'kind'")
        from . import bridge_supervisor as bs
        h = bs.get(kind)
        if h is None:
            return {"kind": kind, "found": False}
        d = _handle_to_dict(h)
        d["found"] = True
        return d

    registry.register("bridge.start",  bridge_start)
    registry.register("bridge.stop",   bridge_stop)
    registry.register("bridge.list",   bridge_list)
    registry.register("bridge.send",   bridge_send)
    registry.register("bridge.status", bridge_status)
