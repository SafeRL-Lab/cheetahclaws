"""builtin.py — built-in tool handlers (RFC 0021 §4).

Three starter tools:

  Echo   — no fs / net required
  Read   — host-fs read; requires_fs=(("r","path"),)
  Write  — host-fs write; requires_fs=(("rw","path"),)

Bash and other shell-execution tools are deliberately deferred to a
follow-up RFC because shell injection prevention is a separate
threat-model decision. Sandbox + capability check is necessary but
not sufficient for safe shell.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .registry import (
    Tool,
    ToolContext,
    ToolFailed,
    ToolFsDenied,
    ToolInvalidArgs,
    ToolRegistry,
)

if TYPE_CHECKING:
    from ..api import Kernel


# Per-call read cap: refuse to read more than this. v1 is generous
# because the caller (LLM agent) usually wants to ingest a moderate
# file. Tools that need bigger reads should use AgentFS / chunked
# tools.
DEFAULT_READ_MAX_BYTES  = 4 * 1024 * 1024
DEFAULT_WRITE_MAX_BYTES = 4 * 1024 * 1024

# Glob / List caps (RFC 0024). Default is generous; a human or LLM
# usually wants the first hundreds of matches, not tens of thousands.
DEFAULT_GLOB_MAX_RESULTS = 1000
MAX_GLOB_MAX_RESULTS     = 10000
DEFAULT_LIST_MAX_ENTRIES = 1000
MAX_LIST_MAX_ENTRIES     = 10000


# ── Echo ──────────────────────────────────────────────────────────────────


def _echo_handler(args: dict, ctx: ToolContext) -> dict:
    text = args.get("text")
    if not isinstance(text, str):
        raise ToolInvalidArgs("'text' must be a string")
    return {"text": text, "len": len(text)}


ECHO_TOOL = Tool(
    name="Echo",
    description="Return the input text. Useful for testing the dispatch path.",
    handler=_echo_handler,
    requires_capability=True,
    requires_fs=(),
)


# ── Read ──────────────────────────────────────────────────────────────────


def _make_read_handler(max_bytes: int):
    def _read_handler(args: dict, ctx: ToolContext) -> dict:
        path_str = args.get("path")
        if not isinstance(path_str, str) or not path_str:
            raise ToolInvalidArgs("'path' must be a non-empty string")
        try:
            p = Path(path_str)
            if not p.is_file():
                raise ToolFailed(f"not a file: {path_str}")
            size = p.stat().st_size
            if size > max_bytes:
                raise ToolFailed(
                    f"file too large: {size} bytes > limit {max_bytes}"
                )
            content = p.read_bytes()
        except (FileNotFoundError, IsADirectoryError) as e:
            raise ToolFailed(str(e)) from e
        except PermissionError as e:
            # OS-level denial (despite cap check passing — e.g. owner
            # restriction). Surface as tool_failed with clear message.
            raise ToolFailed(f"OS denied: {e}") from e
        # Decode as UTF-8 for the result. Binary callers can use
        # AgentFS or a future binary-mode flag.
        try:
            text = content.decode("utf-8")
            return {"content": text, "size": size, "encoding": "utf-8"}
        except UnicodeDecodeError:
            # Return base64 for non-UTF-8 content so the runner can
            # still distinguish — but the LLM-friendly path is UTF-8.
            import base64
            return {
                "content":  base64.b64encode(content).decode("ascii"),
                "size":     size,
                "encoding": "base64",
            }
    return _read_handler


def _make_read_tool(max_bytes: int = DEFAULT_READ_MAX_BYTES) -> Tool:
    return Tool(
        name="Read",
        description=(
            "Read a file from the host filesystem. "
            "Capability fs_grants on the path is required."
        ),
        handler=_make_read_handler(max_bytes),
        requires_capability=True,
        requires_fs=(("r", "path"),),
    )


# ── Write ─────────────────────────────────────────────────────────────────


def _make_write_handler(max_bytes: int):
    def _write_handler(args: dict, ctx: ToolContext) -> dict:
        path_str = args.get("path")
        content  = args.get("content")
        if not isinstance(path_str, str) or not path_str:
            raise ToolInvalidArgs("'path' must be a non-empty string")
        if not isinstance(content, (str, bytes)):
            raise ToolInvalidArgs("'content' must be a string or bytes")
        if isinstance(content, str):
            data = content.encode("utf-8")
        else:
            data = content
        if len(data) > max_bytes:
            raise ToolInvalidArgs(
                f"content too large: {len(data)} > {max_bytes}",
            )
        try:
            p = Path(path_str)
            # Don't auto-create parent dirs — that's a separate
            # privilege; callers can request mkdir via a future tool.
            if not p.parent.exists():
                raise ToolFailed(
                    f"parent directory does not exist: {p.parent}",
                )
            p.write_bytes(data)
        except IsADirectoryError as e:
            raise ToolFailed(str(e)) from e
        except PermissionError as e:
            raise ToolFailed(f"OS denied: {e}") from e
        return {"size": len(data), "path": str(p)}
    return _write_handler


def _make_write_tool(max_bytes: int = DEFAULT_WRITE_MAX_BYTES) -> Tool:
    return Tool(
        name="Write",
        description=(
            "Write a file to the host filesystem (overwrites). "
            "Capability fs_grants 'rw' on the path is required."
        ),
        handler=_make_write_handler(max_bytes),
        requires_capability=True,
        requires_fs=(("rw", "path"),),
    )


# ── Glob (RFC 0024) ───────────────────────────────────────────────────────


def _validate_glob_pattern(pattern) -> str:
    if not isinstance(pattern, str) or not pattern:
        raise ToolInvalidArgs("'pattern' must be a non-empty string")
    if "\x00" in pattern:
        raise ToolInvalidArgs("'pattern' contains NUL")
    # Reject path traversal segments. Pathlib's glob doesn't follow
    # ``..`` by default but we belt-and-braces.
    if "/../" in pattern or pattern.startswith("../") or \
            pattern.endswith("/..") or pattern == "..":
        raise ToolInvalidArgs("'pattern' contains '..' segment")
    return pattern


def _make_glob_handler(default_max: int):
    def _glob_handler(args: dict, ctx: ToolContext) -> dict:
        pattern = _validate_glob_pattern(args.get("pattern"))
        cwd_str = args.get("cwd")
        if not isinstance(cwd_str, str) or not cwd_str:
            raise ToolInvalidArgs("'cwd' must be a non-empty path")
        max_results_raw = args.get("max_results")
        if max_results_raw is None:
            max_results = default_max
        elif (not isinstance(max_results_raw, int)
              or max_results_raw < 1
              or max_results_raw > MAX_GLOB_MAX_RESULTS):
            raise ToolInvalidArgs(
                f"'max_results' must be 1..{MAX_GLOB_MAX_RESULTS}, "
                f"got {max_results_raw!r}",
            )
        else:
            max_results = max_results_raw

        cwd_path = Path(cwd_str)
        if not cwd_path.is_dir():
            raise ToolFailed(f"cwd not a directory: {cwd_str}")

        try:
            # +1 to detect truncation cleanly.
            raw = list(cwd_path.glob(pattern))
        except (ValueError, OSError) as e:
            raise ToolFailed(f"glob error: {e}") from e

        # Sort + dedupe + cap.
        raw.sort()
        truncated = len(raw) > max_results
        if truncated:
            raw = raw[:max_results]

        # Defense-in-depth: filter through fs_grants on each match.
        # Symlinks that escape cwd's grant are dropped here.
        kernel = ctx.kernel
        kept: list[str] = []
        filtered_out = 0
        if kernel is not None:
            for m in raw:
                p = str(m)
                if kernel.cap.check_fs(ctx.pid, p, "r"):
                    kept.append(p)
                else:
                    filtered_out += 1
        else:
            kept = [str(m) for m in raw]

        return {
            "matches":      kept,
            "count":        len(kept),
            "truncated":    truncated,
            "filtered_out": filtered_out,
        }
    return _glob_handler


def _make_glob_tool(default_max: int = DEFAULT_GLOB_MAX_RESULTS) -> Tool:
    return Tool(
        name="Glob",
        description=(
            "Return paths under cwd matching a pathlib glob pattern. "
            "Each match is fs_grants-filtered; results outside grants "
            "are silently dropped (counted in 'filtered_out')."
        ),
        handler=_make_glob_handler(default_max),
        requires_capability=True,
        requires_fs=(("r", "cwd"),),
    )


# ── List (RFC 0024) ───────────────────────────────────────────────────────


def _make_list_handler(default_max: int):
    def _list_handler(args: dict, ctx: ToolContext) -> dict:
        path_str = args.get("path")
        if not isinstance(path_str, str) or not path_str:
            raise ToolInvalidArgs("'path' must be a non-empty string")
        if "\x00" in path_str:
            raise ToolInvalidArgs("'path' contains NUL")
        max_entries_raw = args.get("max_entries")
        if max_entries_raw is None:
            max_entries = default_max
        elif (not isinstance(max_entries_raw, int)
              or max_entries_raw < 1
              or max_entries_raw > MAX_LIST_MAX_ENTRIES):
            raise ToolInvalidArgs(
                f"'max_entries' must be 1..{MAX_LIST_MAX_ENTRIES}, "
                f"got {max_entries_raw!r}",
            )
        else:
            max_entries = max_entries_raw
        include_hidden_raw = args.get("include_hidden", False)
        if not isinstance(include_hidden_raw, bool):
            raise ToolInvalidArgs("'include_hidden' must be bool")

        p = Path(path_str)
        if not p.is_dir():
            raise ToolFailed(f"not a directory: {path_str}")

        entries: list = []
        truncated = False
        try:
            for child in sorted(p.iterdir()):
                if not include_hidden_raw and child.name.startswith("."):
                    continue
                if len(entries) >= max_entries:
                    truncated = True
                    break
                entries.append(_describe_entry(child))
        except (OSError, PermissionError) as e:
            raise ToolFailed(f"iterdir failed: {e}") from e

        return {
            "path":      path_str,
            "entries":   entries,
            "truncated": truncated,
        }
    return _list_handler


def _describe_entry(p: Path) -> dict:
    """Build a single entry record. Defensive against stat failures
    (broken symlinks)."""
    try:
        st = p.lstat()
    except (OSError, PermissionError):
        return {
            "name": p.name, "type": "other",
            "size": None, "mtime": None,
        }
    import stat as _stat
    mode = st.st_mode
    if _stat.S_ISLNK(mode):
        kind = "symlink"
    elif _stat.S_ISDIR(mode):
        kind = "dir"
    elif _stat.S_ISREG(mode):
        kind = "file"
    else:
        kind = "other"
    return {
        "name":  p.name,
        "type":  kind,
        "size":  int(st.st_size) if kind == "file" else None,
        "mtime": float(st.st_mtime),
    }


def _make_list_tool(default_max: int = DEFAULT_LIST_MAX_ENTRIES) -> Tool:
    return Tool(
        name="List",
        description=(
            "List entries in a directory. Returns name + type "
            "(file/dir/symlink/other) + size (for files) + mtime."
        ),
        handler=_make_list_handler(default_max),
        requires_capability=True,
        requires_fs=(("r", "path"),),
    )


# ── register_builtin_tools ─────────────────────────────────────────────


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    kernel: Optional["Kernel"] = None,
    read_max_bytes:    int = DEFAULT_READ_MAX_BYTES,
    write_max_bytes:   int = DEFAULT_WRITE_MAX_BYTES,
    glob_max_results:  int = DEFAULT_GLOB_MAX_RESULTS,
    list_max_entries:  int = DEFAULT_LIST_MAX_ENTRIES,
) -> list[str]:
    """Register Echo, Read, Write, Glob, List. Returns the list of
    registered names. Idempotent — re-registering replaces silently.

    The ``kernel`` argument is unused by the built-in tools' Tool
    objects but reserved for future tools that need it. Glob's
    handler reaches kernel via ToolContext at dispatch time, not at
    registration.
    """
    del kernel  # built-ins don't need it at registration
    registry.register(ECHO_TOOL)
    registry.register(_make_read_tool(read_max_bytes))
    registry.register(_make_write_tool(write_max_bytes))
    registry.register(_make_glob_tool(glob_max_results))
    registry.register(_make_list_tool(list_max_entries))
    # RFC 0030 / 0031 — Diff + AST are no-side-effect inspectors,
    # safe to ship by default.
    from .diff_tool import DIFF_TOOL
    from .ast_tool import AST_TOOL
    registry.register(DIFF_TOOL)
    registry.register(AST_TOOL)
    return ["Echo", "Read", "Write", "Glob", "List", "Diff", "AST"]
