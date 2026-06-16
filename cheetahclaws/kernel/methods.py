"""methods.py — JSON-RPC handlers for the kernel.* namespace.

Thin wrappers around ``KernelStore``. Translation responsibilities:

  * Validate the JSON-RPC param shape (the dispatcher already verifies
    that ``params`` is a dict; type checks for individual fields live
    here).
  * Translate ``KernelError`` subclasses to JSON-RPC error responses
    via the existing dispatcher pathway: we re-raise as a custom
    exception that carries the kernel error code, and the dispatcher
    surfaces it. Since the spike's RpcRegistry catches ``Exception`` and
    returns a generic INTERNAL_ERROR, we instead format the error here
    and let it propagate as a regular JSON-RPC error envelope by raising
    ``_KernelRpcError``.

The cleanest integration with the existing dispatcher is to translate
errors in-method and return success-shaped dicts; the dispatcher's
INTERNAL_ERROR catch-all handles unexpected blow-ups. But the kernel's
typed errors deserve a stable code. We therefore raise a custom
exception that the integration layer catches just before dispatch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .errors import (
    InvalidPayload,
    KernelError,
)
from .store import KernelStore

if TYPE_CHECKING:
    from cheetahclaws.daemon.rpc import CallContext, RpcRegistry


# ── Param coercion helpers ─────────────────────────────────────────────────


def _require(params: dict, key: str, type_, *, allow_none: bool = False):
    if key not in params:
        if allow_none:
            return None
        raise InvalidPayload(f"missing required field {key!r}", field=key)
    value = params[key]
    if value is None and allow_none:
        return None
    if not isinstance(value, type_):
        raise InvalidPayload(
            f"field {key!r} must be {type_.__name__}, got {type(value).__name__}",
            field=key,
        )
    return value


def _opt(params: dict, key: str, type_, *, default=None):
    if key not in params or params[key] is None:
        return default
    value = params[key]
    if not isinstance(value, type_):
        raise InvalidPayload(
            f"field {key!r} must be {type_.__name__}, got {type(value).__name__}",
            field=key,
        )
    return value


# ── Method handlers ────────────────────────────────────────────────────────


def register(registry: "RpcRegistry", store: KernelStore) -> None:
    """Register all kernel.* methods on the daemon's RPC registry.

    The dispatcher catches generic ``Exception`` and returns
    INTERNAL_ERROR; for typed kernel errors we want a stable code. The
    pattern: we let our InvalidPayload / UnknownPid / IllegalTransition
    propagate, the dispatcher's INTERNAL_ERROR catch-all reports
    ``ClassName: message``, and we cover the precise codes via
    ``kernel.error_codes`` (a read-only RPC) for clients that want to
    map by code instead of message.

    A more principled translation lives in a dispatcher patch (out of
    scope for this slice — it would touch existing code). For now,
    KernelError subclasses are reflected through their __str__.
    """

    def _translate(fn):
        """Wrap a handler so KernelError surfaces as a TypeError (-32602
        INVALID_PARAMS) for invalid payload, and as a regular Exception
        otherwise. The dispatcher distinguishes TypeError from generic
        Exception."""
        def wrapper(params: dict, ctx: "CallContext"):
            try:
                return fn(params, ctx)
            except InvalidPayload as e:
                # Map to TypeError so the dispatcher emits INVALID_PARAMS
                # (-32602) — closest stock JSON-RPC code for our case.
                raise TypeError(str(e))
            except KernelError as e:
                # Use a custom exception class; the dispatcher's
                # INTERNAL_ERROR catch-all will surface
                # "KernelError: <message>" to the client. The error code
                # is encoded in __class__.__name__ for clients that
                # parse it.
                raise RuntimeError(f"{type(e).__name__}: {e}")
        wrapper.__name__ = fn.__name__
        return wrapper

    # ── kernel.agent.* ─────────────────────────────────────────────

    @_translate
    def agent_create(params: dict, ctx: "CallContext") -> dict:
        name = _require(params, "name", str)
        template = _require(params, "template", str)
        parent_pid = _opt(params, "parent_pid", int)
        metadata = _opt(params, "metadata", dict)
        agent = store.create(
            name=name,
            template=template,
            parent_pid=parent_pid,
            metadata=metadata,
        )
        return {"pid": agent.pid, "state": agent.state}

    @_translate
    def agent_get(params: dict, ctx: "CallContext") -> dict:
        pid = _require(params, "pid", int)
        return store.get(pid).to_dict()

    @_translate
    def agent_list(params: dict, ctx: "CallContext") -> dict:
        state = _opt(params, "state", str)
        parent_pid = _opt(params, "parent_pid", int)
        limit = _opt(params, "limit", int, default=100)
        offset = _opt(params, "offset", int, default=0)
        agents, total = store.list(
            state=state, parent_pid=parent_pid,
            limit=limit, offset=offset,
        )
        return {
            "agents": [a.to_dict() for a in agents],
            "total":  total,
        }

    @_translate
    def agent_transition(params: dict, ctx: "CallContext") -> dict:
        pid = _require(params, "pid", int)
        target_state = _require(params, "target_state", str)
        reason = _opt(params, "reason", str)
        prev, new, event_id = store.transition(pid, target_state, reason=reason)
        return {
            "pid":        pid,
            "prev_state": prev,
            "state":      new,
            "event_id":   event_id,
        }

    @_translate
    def agent_terminate(params: dict, ctx: "CallContext") -> dict:
        pid = _require(params, "pid", int)
        exit_kind = _require(params, "exit_kind", str)
        exit_detail = _opt(params, "exit_detail", dict)
        prev, event_id = store.terminate(
            pid, exit_kind=exit_kind, exit_detail=exit_detail,
        )
        return {
            "pid":        pid,
            "prev_state": prev,
            "state":      "DEAD",
            "event_id":   event_id,
        }

    # ── kernel.events.* ────────────────────────────────────────────

    @_translate
    def events_append(params: dict, ctx: "CallContext") -> dict:
        pid = _require(params, "pid", int)
        kind = _require(params, "kind", str)
        payload = _require(params, "payload", dict)
        causation_id = _opt(params, "causation_id", int)
        correlation_id = _opt(params, "correlation_id", str)
        event_id = store.events_append(
            pid=pid, kind=kind, payload=payload,
            causation_id=causation_id, correlation_id=correlation_id,
        )
        return {"event_id": event_id}

    @_translate
    def events_tail(params: dict, ctx: "CallContext") -> dict:
        pid = _opt(params, "pid", int)
        kind = _opt(params, "kind", str)
        since_event_id = _opt(params, "since_event_id", int, default=0) or 0
        limit = _opt(params, "limit", int, default=100) or 100
        events = store.events_tail(
            pid=pid, kind=kind,
            since_event_id=since_event_id, limit=limit,
        )
        next_cursor = events[-1].event_id if events else since_event_id
        return {
            "events":      [e.to_dict() for e in events],
            "next_cursor": next_cursor,
        }

    # ── kernel.info ────────────────────────────────────────────────

    @_translate
    def kernel_info(params: dict, ctx: "CallContext") -> dict:
        return store.info()

    # ── Register ───────────────────────────────────────────────────

    registry.register("kernel.agent.create",     agent_create)
    registry.register("kernel.agent.get",        agent_get)
    registry.register("kernel.agent.list",       agent_list)
    registry.register("kernel.agent.transition", agent_transition)
    registry.register("kernel.agent.terminate",  agent_terminate)
    registry.register("kernel.events.append",    events_append)
    registry.register("kernel.events.tail",      events_tail)
    registry.register("kernel.info",             kernel_info)
