"""session_methods.py — `session.*` JSON-RPC methods (RFC 0002 F-6 Phase 2).

These methods give bridges (and any other inbound source) a uniform way
to push messages into the daemon, and give agent drivers (REPL, Web UI,
another bridge) a uniform way to push replies back out. Both directions
ride the SSE event bus so any number of subscribers see the same
sequence of events.

Methods:

    session.send(session_id, text, origin=None, message_id=None)
        Mint an inbound message for ``session_id``. Publishes
        ``session_inbound`` on the event bus. ``origin`` is a free-form
        string identifying the source (e.g. ``"telegram"``,
        ``"slack"``, ``"wechat"``, ``"api"``); used for the permission
        store's originator routing.

    session.reply(session_id, text, target_bridges=None, message_id=None)
        Mint an outbound message for ``session_id``. Publishes
        ``session_outbound`` on the event bus.  ``target_bridges`` is a
        list of bridge kinds (or omitted for broadcast) — a Phase 2
        bridge worker subscribes to outbound events and forwards to its
        chat iff its kind is in the target list (or the list is empty).

    session.list_recent(limit=20)
        Return the most recent session_id ↔ origin pairs seen by the
        daemon. Backed by an in-memory LRU; survives the lifetime of
        the daemon process.

This module is intentionally thin: it doesn't drive any agent — that's
the responsibility of whatever subscribes to ``session_inbound`` (REPL,
Web UI, or another bridge that wants to act as the agent driver).
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

from .rpc import RpcRegistry

if TYPE_CHECKING:
    from .server import DaemonState


# In-memory LRU of (session_id, origin) tuples — keyed by session_id so
# repeat sends from the same chat don't keep duplicating entries.
_RECENT_LRU: "OrderedDict[str, dict]" = OrderedDict()
_RECENT_LRU_MAX = 256
_RECENT_LOCK = threading.Lock()


def _record_session(session_id: str, origin: Optional[str]) -> None:
    with _RECENT_LOCK:
        existing = _RECENT_LRU.pop(session_id, None)
        entry = existing or {"session_id": session_id, "origin": origin,
                              "first_seen": time.time()}
        if origin:
            entry["origin"] = origin
        entry["last_seen"] = time.time()
        _RECENT_LRU[session_id] = entry
        while len(_RECENT_LRU) > _RECENT_LRU_MAX:
            _RECENT_LRU.popitem(last=False)


def _list_recent_snapshot(limit: int) -> list[dict]:
    with _RECENT_LOCK:
        # OrderedDict iterates oldest → newest; reverse for newest-first.
        items = list(_RECENT_LRU.values())[-limit:]
    items.reverse()
    return items


def register(registry: RpcRegistry, daemon_state: "DaemonState") -> None:
    from . import events as _events

    def session_send(params: dict, ctx) -> dict:
        session_id = params.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise TypeError("session.send requires non-empty 'session_id'")
        text = params.get("text")
        if not isinstance(text, str) or not text:
            raise TypeError("session.send requires non-empty 'text'")
        origin = params.get("origin")
        if origin is not None and not isinstance(origin, str):
            raise TypeError("session.send: 'origin' must be a string")
        message_id = str(params.get("message_id")
                         or f"msg_{uuid.uuid4().hex[:12]}")

        # Stamp originator on the event so PermissionStore routing
        # (RFC 0001 §2) can pin permission requests back to this
        # bridge / API client. Use the client_id from the RPC call
        # context if no explicit origin was supplied — that way a
        # raw API user gets the same originator-trace treatment as
        # a bridge worker.
        eff_origin = origin or getattr(ctx, "client_id", "") or "api"

        bus = _events.get_bus()
        bus.publish("session_inbound", {
            "session_id": session_id,
            "text":       text,
            "origin":     eff_origin,
            "message_id": message_id,
            "ts":         time.time(),
        })
        _record_session(session_id, eff_origin)
        return {"session_id": session_id, "message_id": message_id}

    def session_reply(params: dict, _ctx) -> dict:
        session_id = params.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise TypeError("session.reply requires non-empty 'session_id'")
        text = params.get("text")
        if not isinstance(text, str) or not text:
            raise TypeError("session.reply requires non-empty 'text'")
        targets = params.get("target_bridges", None)
        if targets is not None:
            if not isinstance(targets, list) or not all(
                    isinstance(t, str) for t in targets):
                raise TypeError(
                    "session.reply: 'target_bridges' must be a list of strings")
        message_id = str(params.get("message_id")
                         or f"out_{uuid.uuid4().hex[:12]}")

        bus = _events.get_bus()
        bus.publish("session_outbound", {
            "session_id":     session_id,
            "text":           text,
            "target_bridges": targets,    # None = broadcast
            "message_id":     message_id,
            "ts":             time.time(),
        })
        return {"session_id": session_id, "message_id": message_id}

    def session_list_recent(params: dict, _ctx) -> dict:
        try:
            limit = int(params.get("limit", 20))
        except (TypeError, ValueError) as e:
            raise TypeError(f"session.list_recent: 'limit' must be int: {e}")
        if limit < 1:
            raise TypeError("session.list_recent: 'limit' must be ≥ 1")
        return {"sessions": _list_recent_snapshot(limit)}

    registry.register("session.send",        session_send)
    registry.register("session.reply",       session_reply)
    registry.register("session.list_recent", session_list_recent)
