"""permission.py — pending PermissionRequest store with originator-only answer.

Model (RFC §2):
- Each request has a request_id and an originator (client_id).
- Only the client whose client_id matches the originator may answer; anyone
  else gets 403 not_originator.
- Each request has expires_at; a janitor thread auto-denies on expiry and
  publishes a `permission_timeout` event.
- Defaults: 30 min interactive (RFC §9 patch), 5 min unattended.
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import events

DEFAULT_TIMEOUT_INTERACTIVE_S = 30 * 60
DEFAULT_TIMEOUT_UNATTENDED_S = 5 * 60
JANITOR_TICK_S = 1.0


class PermissionError(Exception):
    """Base for permission-routing errors."""


class NotOriginator(PermissionError):
    """Raised when a non-originator tries to answer a request."""


class UnknownRequest(PermissionError):
    """Raised when answering an unknown / already-resolved request."""


@dataclass
class PermissionRequest:
    request_id: str
    originator: str  # client_id
    tool: str
    input: dict
    rationale: str
    created_at: float
    expires_at: float
    answer: Optional[dict] = field(default=None)
    resolved_at: Optional[float] = field(default=None)
    # Fired once with this request after `answer` is set (either by the
    # originator's RPC call or by the janitor's timeout path). Used by
    # internal callers — e.g. the F-4 agent supervisor — that need to
    # forward the result somewhere outside the RPC reply path. Never
    # invoked under the store's lock.
    on_answer: Optional[Callable[["PermissionRequest"], None]] = field(
        default=None, repr=False
    )


class PermissionStore:
    def __init__(self) -> None:
        self._pending: dict[str, PermissionRequest] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._janitor: Optional[threading.Thread] = None

    def start_janitor(self) -> None:
        if self._janitor is not None:
            return
        self._janitor = threading.Thread(
            target=self._janitor_loop, name="perm-janitor", daemon=True
        )
        self._janitor.start()

    def stop(self) -> None:
        self._stop.set()
        if self._janitor is not None:
            self._janitor.join(timeout=2.0)

    def create(
        self,
        *,
        originator: str,
        tool: str,
        tool_input: dict,
        rationale: str = "",
        timeout_s: float = DEFAULT_TIMEOUT_INTERACTIVE_S,
        on_answer: Optional[Callable[[PermissionRequest], None]] = None,
    ) -> PermissionRequest:
        rid = "pr_" + secrets.token_hex(8)
        now = time.time()
        req = PermissionRequest(
            request_id=rid,
            originator=originator,
            tool=tool,
            input=tool_input,
            rationale=rationale,
            created_at=now,
            expires_at=now + timeout_s,
            on_answer=on_answer,
        )
        with self._lock:
            self._pending[rid] = req
        events.get_bus().publish(
            "permission_request",
            {
                "request_id": rid,
                "tool": tool,
                "input": tool_input,
                "rationale": rationale,
                "expires_at": req.expires_at,
            },
            originator={"client_id": originator},
        )
        return req

    def answer(self, request_id: str, client_id: str, result: dict) -> PermissionRequest:
        with self._lock:
            req = self._pending.get(request_id)
            if req is None:
                raise UnknownRequest(request_id)
            if req.originator != client_id:
                raise NotOriginator(request_id)
            req.answer = result
            req.resolved_at = time.time()
            del self._pending[request_id]
        # Fire the optional callback outside the lock so a slow consumer
        # can't block other store operations.
        cb = req.on_answer
        if cb is not None:
            try:
                cb(req)
            except Exception:
                pass
        events.get_bus().publish(
            "permission_answered",
            {"request_id": request_id, "answer": result},
            originator={"client_id": client_id},
        )
        return req

    def refresh_timeout(
        self, request_id: str, client_id: str, extend_s: float
    ) -> PermissionRequest:
        with self._lock:
            req = self._pending.get(request_id)
            if req is None:
                raise UnknownRequest(request_id)
            if req.originator != client_id:
                raise NotOriginator(request_id)
            req.expires_at += extend_s
            new_exp = req.expires_at
        events.get_bus().publish(
            "permission_timeout_extended",
            {"request_id": request_id, "expires_at": new_exp},
            originator={"client_id": client_id},
        )
        return req

    def get(self, request_id: str) -> Optional[PermissionRequest]:
        with self._lock:
            return self._pending.get(request_id)

    def list_pending_for(self, client_id: str) -> list[PermissionRequest]:
        with self._lock:
            return [r for r in self._pending.values() if r.originator == client_id]

    def _janitor_loop(self) -> None:
        while not self._stop.wait(JANITOR_TICK_S):
            now = time.time()
            expired: list[PermissionRequest] = []
            with self._lock:
                for rid, req in list(self._pending.items()):
                    if req.expires_at <= now:
                        expired.append(req)
                        del self._pending[rid]
            for req in expired:
                # Synthesize an auto-deny so on_answer subscribers can
                # treat timeout and explicit denial uniformly: check
                # req.answer.get("approve") for the boolean outcome.
                req.answer = {"approve": False, "timeout": True}
                req.resolved_at = now
                cb = req.on_answer
                if cb is not None:
                    try:
                        cb(req)
                    except Exception:
                        pass
                events.get_bus().publish(
                    "permission_timeout",
                    {"request_id": req.request_id, "auto_answer": "deny"},
                    originator={"client_id": req.originator},
                )
