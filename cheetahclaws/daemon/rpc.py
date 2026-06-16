"""rpc.py — JSON-RPC 2.0 dispatcher with method registry.

The dispatcher is intentionally minimal: validate envelope, look up method,
call it with (params, ctx), shape result into a JSON-RPC response.

Application errors (e.g. NotOriginator) get HTTP 403 at the handler level;
the dispatcher just signals the kind via raised exceptions.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from . import permission

# JSON-RPC 2.0 standard error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Application error codes
APP_NOT_ORIGINATOR = -32001
APP_UNKNOWN_REQUEST = -32002


@dataclass
class CallContext:
    """Per-call context handed to method functions."""
    client_id: str
    transport: str  # "unix" | "tcp"
    api_version: str


MethodFn = Callable[[dict, CallContext], Any]


class RpcRegistry:
    def __init__(self) -> None:
        self._methods: dict[str, MethodFn] = {}
        self._lock = threading.Lock()

    def register(self, name: str, fn: MethodFn) -> None:
        with self._lock:
            self._methods[name] = fn

    def methods(self) -> list[str]:
        with self._lock:
            return sorted(self._methods)

    def dispatch(self, envelope: dict, ctx: CallContext) -> tuple[Optional[dict], int]:
        """Returns (response_envelope_or_None_for_notify, http_status).

        http_status is 200 for success and most JSON-RPC errors, but escalates
        to 403 for not_originator and 401 for unauthenticated (the handler
        translates Unauthenticated before reaching here, so 401 isn't returned
        from this path).
        """
        if envelope.get("jsonrpc") != "2.0":
            return _err(envelope.get("id"), INVALID_REQUEST, "jsonrpc must be '2.0'"), 200
        method = envelope.get("method")
        if not isinstance(method, str):
            return _err(envelope.get("id"), INVALID_REQUEST, "method must be string"), 200
        params = envelope.get("params") or {}
        if not isinstance(params, dict):
            return _err(envelope.get("id"), INVALID_PARAMS, "params must be object"), 200
        msg_id = envelope.get("id")  # None means notification

        with self._lock:
            fn = self._methods.get(method)
        if fn is None:
            return _err(msg_id, METHOD_NOT_FOUND, f"method {method!r} not found"), 200

        try:
            result = fn(params, ctx)
        except permission.NotOriginator as e:
            return _err(msg_id, APP_NOT_ORIGINATOR, "not the originator", {"request_id": str(e)}), 403
        except permission.UnknownRequest as e:
            return _err(msg_id, APP_UNKNOWN_REQUEST, "unknown request", {"request_id": str(e)}), 200
        except TypeError as e:
            return _err(msg_id, INVALID_PARAMS, str(e)), 200
        except Exception as e:
            return _err(msg_id, INTERNAL_ERROR, type(e).__name__ + ": " + str(e)), 200

        if msg_id is None:
            return None, 204
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}, 200


def _err(msg_id, code: int, message: str, data: Optional[dict] = None) -> dict:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": error}
