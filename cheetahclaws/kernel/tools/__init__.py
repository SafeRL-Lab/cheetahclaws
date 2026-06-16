"""kernel.tools — tool dispatch + registry (RFC 0021).

Public surface::

    Tool             — frozen dataclass: name, handler, capability + fs requirements
    ToolContext      — passed to handlers; carries pid + kernel
    ToolRegistry     — register / lookup / list

    ToolError                  — root for tool-layer exceptions
    ToolNotFound, ToolDenied,
    ToolFsDenied, ToolInvalidArgs, ToolFailed

    register_builtin_tools(registry, *, kernel=None)
        Registers Read, Write, Echo.

    dispatch_tool_call(...)
        Pure function used by the supervisor's IPC loop.
"""
from __future__ import annotations

from .registry import (
    Tool,
    ToolContext,
    ToolDenied,
    ToolError,
    ToolFailed,
    ToolFsDenied,
    ToolInvalidArgs,
    ToolNetDenied,
    ToolNotFound,
    ToolRegistry,
    dispatch_tool_call,
)
from .ast_tool import AST_TOOL, ast_handler
from .builtin import register_builtin_tools
from .diff_tool import DIFF_TOOL, diff_handler
from .exec_tool import EXEC_TOOL, register_exec_tool
from .fetch_tool import FETCH_TOOL, register_fetch_tool
from .git_tool import GIT_TOOL, register_git_tool

__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolError",
    "ToolNotFound",
    "ToolDenied",
    "ToolFsDenied",
    "ToolNetDenied",
    "ToolInvalidArgs",
    "ToolFailed",
    "dispatch_tool_call",
    "register_builtin_tools",
    # Built-in inspectors (RFC 0030, 0031): auto-registered.
    "DIFF_TOOL",
    "diff_handler",
    "AST_TOOL",
    "ast_handler",
    # Opt-in (RFC 0023, 0025, 0032): NOT in register_builtin_tools.
    "EXEC_TOOL",
    "register_exec_tool",
    "FETCH_TOOL",
    "register_fetch_tool",
    "GIT_TOOL",
    "register_git_tool",
]
