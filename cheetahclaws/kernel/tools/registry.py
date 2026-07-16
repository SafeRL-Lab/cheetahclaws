"""registry.py — Tool / ToolRegistry / ToolContext + dispatch (RFC 0021).

The dispatch function takes a tool_call message + agent context +
registry + a `Kernel`-like view of the capability store. It runs
the cap check, fs check, and handler invocation, and returns the
response message dict — ready to ship over IPC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from ..api import Kernel


# ── Errors ────────────────────────────────────────────────────────────────


class ToolError(Exception):
    """Root for tool-layer exceptions. Subclasses each map to a
    distinct ``error`` slug in the IPC tool_response payload."""
    error_slug: str = "tool_failed"


class ToolNotFound(ToolError):
    error_slug = "tool_not_found"


class ToolDenied(ToolError):
    """Capability check failed — agent's tool_grants doesn't cover
    the requested tool."""
    error_slug = "permission_denied"


class ToolFsDenied(ToolError):
    """Capability check failed for an fs path — agent's fs_grants
    doesn't cover one of the tool's requires_fs entries."""
    error_slug = "fs_denied"


class ToolNetDenied(ToolError):
    """Capability check failed for a network host (RFC 0025) — agent's
    net_grants doesn't cover the requested hostname, OR the hostname
    resolved to a private/loopback IP (DNS rebinding defence)."""
    error_slug = "net_denied"


class ToolInvalidArgs(ToolError):
    """Args validation failed inside the handler."""
    error_slug = "invalid_args"


class ToolFailed(ToolError):
    """Handler raised something unexpected. The supervisor wraps
    other exceptions in this when emitting the tool_response."""
    error_slug = "tool_failed"


# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolContext:
    """Per-call context passed to handlers. Tools that need to
    talk to AgentFS / mailbox use ``kernel`` (the facade).

    RFC 0028: ``on_chunk`` lets a streaming tool emit
    incremental output; the supervisor wires this to the
    wait()-time ``on_chunk`` callback. Always optional —
    handlers MUST treat ``None`` as "no streaming".
    """
    pid:      int
    kernel:   Optional["Kernel"]
    on_chunk: Optional[Callable[[dict], None]] = None


@dataclass(frozen=True)
class Tool:
    """A registered tool.

    ``handler(args: dict, ctx: ToolContext) -> dict`` is the work
    function. Args are the runner-supplied dict; the handler
    should validate + execute + return a result dict.

    ``requires_capability=True`` (default) gates the call on
    ``kernel.cap.check_tool(pid, name)``.

    ``requires_fs`` is a tuple of ``(mode, args_key)`` pairs.
    For each, the dispatcher reads ``args[args_key]`` (must be a
    string path) and runs ``kernel.cap.check_fs(pid, path, mode)``.
    Mode is "r" or "rw".
    """
    name:                str
    description:         str
    handler:             Callable[[dict, "ToolContext"], dict]
    requires_capability: bool = True
    requires_fs:         tuple = ()


# ── Registry ──────────────────────────────────────────────────────────────


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict = {}

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise ToolError(f"tool must be Tool, got {type(tool).__name__}")
        if not tool.name or not isinstance(tool.name, str):
            raise ToolError(f"tool.name must be non-empty str, got {tool.name!r}")
        if not callable(tool.handler):
            raise ToolError("tool.handler must be callable")
        # Validate requires_fs shape.
        for entry in tool.requires_fs:
            if not (isinstance(entry, tuple) and len(entry) == 2
                    and entry[0] in ("r", "rw")
                    and isinstance(entry[1], str)):
                raise ToolError(
                    f"requires_fs entries must be (mode, key) where mode in "
                    f"('r','rw'); got {entry!r}",
                )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolNotFound(f"no such tool: {name!r}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def list(self) -> list[str]:
        return sorted(self._tools)

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None


# ── Dispatch ─────────────────────────────────────────────────────────────


def dispatch_tool_call(
    *,
    msg:      dict,
    pid:      int,
    registry: ToolRegistry,
    kernel:   Optional["Kernel"] = None,
    on_chunk: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Process one tool_call IPC message and return the
    tool_response dict. Pure: no side effects in the kernel
    beyond what the handler itself does + the audit event the
    caller writes.

    The supervisor calls this synchronously inside its message
    drain loop and ships the returned dict over IPC.
    """
    tool_call_id = msg.get("tool_call_id", "")
    tool_name    = msg.get("tool")
    args         = msg.get("args") or {}
    if not isinstance(tool_name, str) or not tool_name:
        return _err(tool_call_id, "tool_not_found",
                     "missing 'tool' field")
    if not isinstance(args, dict):
        return _err(tool_call_id, "invalid_args",
                     "'args' must be an object")

    # 1) Lookup.
    try:
        tool = registry.get(tool_name)
    except ToolNotFound as e:
        return _err(tool_call_id, "tool_not_found", str(e))

    # 2) Capability check.
    if tool.requires_capability and kernel is not None:
        if not kernel.cap.check_tool(pid, tool.name):
            return _err(
                tool_call_id, "permission_denied",
                f"agent {pid} not granted tool {tool.name!r}",
            )

    # 3) Fs checks.
    for mode, key in tool.requires_fs:
        path = args.get(key)
        if not isinstance(path, str) or not path:
            return _err(
                tool_call_id, "invalid_args",
                f"'{key}' must be a non-empty path string",
            )
        if kernel is not None and not kernel.cap.check_fs(pid, path, mode):
            return _err(
                tool_call_id, "fs_denied",
                f"agent {pid} not granted {mode} on {path!r}",
            )

    # 4) Execute handler.
    ctx = ToolContext(pid=pid, kernel=kernel, on_chunk=on_chunk)
    try:
        result = tool.handler(args, ctx)
    except ToolError as e:
        return _err(tool_call_id, e.error_slug, str(e))
    except Exception as e:
        return _err(tool_call_id, "tool_failed",
                     f"{type(e).__name__}: {e}")

    if not isinstance(result, dict):
        return _err(
            tool_call_id, "tool_failed",
            f"handler returned non-dict: {type(result).__name__}",
        )

    return {
        "op":           "tool_response",
        "tool_call_id": tool_call_id,
        "ok":           True,
        "result":       result,
    }


def _err(tool_call_id: str, slug: str, message: str) -> dict:
    return {
        "op":           "tool_response",
        "tool_call_id": tool_call_id,
        "ok":           False,
        "error":        slug,
        "message":      message,
    }
