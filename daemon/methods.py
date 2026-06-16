"""methods.py — Three demo methods that exercise the contract.

  echo.ping                   sync return + side-effect event
  permission.demo             create a PermissionRequest aimed at an originator
  permission.answer           answer a request (originator-only)
  permission.refresh_timeout  extend a pending request's timeout
"""
from __future__ import annotations

import time
from typing import Optional

from . import events, permission
from .rpc import CallContext, RpcRegistry

_START = time.monotonic()


def register(registry: RpcRegistry, store: permission.PermissionStore) -> None:
    def echo_ping(params: dict, ctx: CallContext) -> dict:
        events.get_bus().publish(
            "ping_received",
            {"from_client": ctx.client_id, "params": params},
        )
        return {
            "pong": True,
            "ts": time.time(),
            "server_uptime_s": round(time.monotonic() - _START, 3),
            "echo": params,
        }

    def permission_demo(params: dict, ctx: CallContext) -> dict:
        # `originator` defaults to the caller; tests can target a different
        # client_id to set up the not_originator scenario.
        originator: str = params.get("originator") or ctx.client_id
        tool: str = params.get("tool", "Bash")
        tool_input: dict = params.get("input", {})
        rationale: str = params.get("rationale", "demo request from spike")
        timeout_s: float = float(params.get("timeout_s", permission.DEFAULT_TIMEOUT_INTERACTIVE_S))
        req = store.create(
            originator=originator,
            tool=tool,
            tool_input=tool_input,
            rationale=rationale,
            timeout_s=timeout_s,
        )
        return {
            "request_id": req.request_id,
            "originator": req.originator,
            "expires_at": req.expires_at,
        }

    def permission_answer(params: dict, ctx: CallContext) -> dict:
        request_id = params["request_id"]
        result = params.get("result", {"approve": False})
        req = store.answer(request_id, ctx.client_id, result)
        return {
            "request_id": req.request_id,
            "resolved_at": req.resolved_at,
            "answer": req.answer,
        }

    def permission_refresh_timeout(params: dict, ctx: CallContext) -> dict:
        request_id = params["request_id"]
        extend_s = float(params.get("extend_s", 600))
        req = store.refresh_timeout(request_id, ctx.client_id, extend_s)
        return {"request_id": req.request_id, "expires_at": req.expires_at}

    def permission_list(_params: dict, ctx: CallContext) -> dict:
        pending = store.list_pending_for(ctx.client_id)
        return {
            "pending": [
                {
                    "request_id": r.request_id,
                    "tool": r.tool,
                    "input": r.input,
                    "rationale": r.rationale,
                    "expires_at": r.expires_at,
                }
                for r in pending
            ]
        }

    registry.register("echo.ping", echo_ping)
    registry.register("permission.demo", permission_demo)
    registry.register("permission.answer", permission_answer)
    registry.register("permission.refresh_timeout", permission_refresh_timeout)
    registry.register("permission.list", permission_list)
