"""diff_tool.py — unified-diff built-in (RFC 0030).

Stdlib only (``difflib.unified_diff``). Two modes:

  * Path mode: ``path_a`` + ``path_b`` (fs-cap gated, "r" on each).
  * Text mode: ``text_a`` + ``text_b`` (no fs touch).

Mixing modes raises invalid_args. Auto-registered by
``register_builtin_tools``.
"""
from __future__ import annotations

import difflib
from pathlib import Path

from .registry import (
    Tool,
    ToolContext,
    ToolFailed,
    ToolFsDenied,
    ToolInvalidArgs,
)


# ── Limits ──────────────────────────────────────────────────────────────


DEFAULT_READ_MAX_BYTES = 4 * 1024 * 1024
DEFAULT_DIFF_CAP_BYTES = 2 * 1024 * 1024
DEFAULT_CONTEXT_LINES  = 3
MAX_CONTEXT_LINES      = 20


# ── Validation ──────────────────────────────────────────────────────────


def _validate_context_lines(n) -> int:
    if n is None:
        return DEFAULT_CONTEXT_LINES
    if not isinstance(n, int) or isinstance(n, bool):
        raise ToolInvalidArgs(
            f"'context_lines' must be int, got {type(n).__name__}",
        )
    if n < 0 or n > MAX_CONTEXT_LINES:
        raise ToolInvalidArgs(
            f"'context_lines' must be in [0, {MAX_CONTEXT_LINES}], "
            f"got {n}",
        )
    return n


def _validate_label(label, default: str) -> str:
    if label is None:
        return default
    if not isinstance(label, str) or not label:
        raise ToolInvalidArgs("label must be a non-empty string")
    if len(label) > 256:
        raise ToolInvalidArgs("label too long (max 256 chars)")
    if "\n" in label or "\r" in label:
        raise ToolInvalidArgs("label cannot contain newlines")
    return label


def _read_file(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        raise ToolFailed(f"path not found: {path_str!r}")
    if not p.is_file():
        raise ToolFailed(f"path is not a file: {path_str!r}")
    try:
        size = p.stat().st_size
    except OSError as e:
        raise ToolFailed(f"stat failed: {e}") from e
    if size > DEFAULT_READ_MAX_BYTES:
        raise ToolFailed(
            f"file too large for diff: {size} > "
            f"{DEFAULT_READ_MAX_BYTES} bytes",
        )
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise ToolFailed(f"read failed: {e}") from e


# ── Handler ─────────────────────────────────────────────────────────────


def diff_handler(args: dict, ctx: ToolContext) -> dict:
    path_a = args.get("path_a")
    path_b = args.get("path_b")
    text_a = args.get("text_a")
    text_b = args.get("text_b")

    has_path = path_a is not None or path_b is not None
    has_text = text_a is not None or text_b is not None
    if has_path and has_text:
        raise ToolInvalidArgs(
            "cannot mix path_* and text_* args; pick one mode",
        )
    if not has_path and not has_text:
        raise ToolInvalidArgs(
            "must provide either path_a/path_b or text_a/text_b",
        )

    context_lines = _validate_context_lines(args.get("context_lines"))

    if has_path:
        if not isinstance(path_a, str) or not path_a:
            raise ToolInvalidArgs("'path_a' must be non-empty string")
        if not isinstance(path_b, str) or not path_b:
            raise ToolInvalidArgs("'path_b' must be non-empty string")
        if ctx.kernel is not None:
            if not ctx.kernel.cap.check_fs(ctx.pid, path_a, "r"):
                raise ToolFsDenied(
                    f"agent {ctx.pid} not granted 'r' on {path_a!r}",
                )
            if not ctx.kernel.cap.check_fs(ctx.pid, path_b, "r"):
                raise ToolFsDenied(
                    f"agent {ctx.pid} not granted 'r' on {path_b!r}",
                )
        a_text = _read_file(path_a)
        b_text = _read_file(path_b)
        label_a = _validate_label(args.get("label_a"), path_a)
        label_b = _validate_label(args.get("label_b"), path_b)
    else:
        if not isinstance(text_a, str):
            raise ToolInvalidArgs("'text_a' must be a string")
        if not isinstance(text_b, str):
            raise ToolInvalidArgs("'text_b' must be a string")
        a_text = text_a
        b_text = text_b
        label_a = _validate_label(args.get("label_a"), "a")
        label_b = _validate_label(args.get("label_b"), "b")

    a_lines = a_text.splitlines(keepends=True)
    b_lines = b_text.splitlines(keepends=True)
    identical = a_text == b_text

    diff_iter = difflib.unified_diff(
        a_lines, b_lines,
        fromfile=label_a, tofile=label_b,
        n=context_lines,
    )

    out_buf: list = []
    out_size = 0
    truncated = False
    for line in diff_iter:
        line_bytes = len(line.encode("utf-8"))
        if out_size + line_bytes > DEFAULT_DIFF_CAP_BYTES:
            truncated = True
            out_buf.append(
                f"[diff truncated at {DEFAULT_DIFF_CAP_BYTES} bytes]\n",
            )
            break
        out_buf.append(line)
        out_size += line_bytes

    diff_text = "".join(out_buf)
    return {
        "diff":         diff_text,
        "label_a":      label_a,
        "label_b":      label_b,
        "lines_a":      len(a_lines),
        "lines_b":      len(b_lines),
        "identical":    identical,
        "diff_lines":   diff_text.count("\n"),
        "truncated":    truncated,
        "context_lines": context_lines,
    }


DIFF_TOOL = Tool(
    name="Diff",
    description=(
        "Compute a unified diff between two files (path_a/path_b) "
        "or two strings (text_a/text_b). Path mode requires 'r' "
        "fs_grants on each path. Output capped at 2 MB."
    ),
    handler=diff_handler,
    requires_capability=True,
    requires_fs=(),     # handler does its own fs check (two paths).
)


__all__ = ["DIFF_TOOL", "diff_handler", "DEFAULT_DIFF_CAP_BYTES"]
