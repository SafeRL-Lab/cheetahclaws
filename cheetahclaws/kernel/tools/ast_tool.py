"""ast_tool.py — Python AST inspector built-in (RFC 0031).

Stdlib only (``ast`` module). Two modes:

  * Path mode: ``path`` (fs-cap gated, "r"). Must end in ``.py``.
  * Text mode: ``text`` (+ optional label).

Returns a flat list of nodes (functions, classes, imports, etc.)
each with a ``scope`` path and line numbers. Auto-registered by
``register_builtin_tools``.
"""
from __future__ import annotations

import ast
from pathlib import Path

from .registry import (
    Tool,
    ToolContext,
    ToolFailed,
    ToolFsDenied,
    ToolInvalidArgs,
)


# ── Limits ──────────────────────────────────────────────────────────────


DEFAULT_AST_MAX_FILE_BYTES = 2 * 1024 * 1024
DEFAULT_AST_MAX_NODES      = 5000
DEFAULT_AST_MAX_DEPTH      = 4
MIN_AST_MAX_DEPTH          = 1
MAX_AST_MAX_DEPTH          = 10

ALLOWED_KINDS = frozenset({
    "function", "async_function", "class",
    "import", "import_from", "assign", "annotation",
})

DEFAULT_INCLUDE = ("function", "async_function", "class",
                    "import", "import_from")


# ── Validation ──────────────────────────────────────────────────────────


def _validate_include(value) -> tuple:
    if value is None:
        return DEFAULT_INCLUDE
    if not isinstance(value, list):
        raise ToolInvalidArgs("'include' must be a list of strings")
    out = []
    for v in value:
        if not isinstance(v, str) or v not in ALLOWED_KINDS:
            raise ToolInvalidArgs(
                f"include kind {v!r} not in {sorted(ALLOWED_KINDS)}",
            )
        out.append(v)
    return tuple(out)


def _validate_max_depth(value) -> int:
    if value is None:
        return DEFAULT_AST_MAX_DEPTH
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolInvalidArgs("'max_depth' must be int")
    if value < MIN_AST_MAX_DEPTH or value > MAX_AST_MAX_DEPTH:
        raise ToolInvalidArgs(
            f"'max_depth' must be in "
            f"[{MIN_AST_MAX_DEPTH}, {MAX_AST_MAX_DEPTH}], got {value}",
        )
    return value


def _validate_label(value, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value:
        raise ToolInvalidArgs("'label' must be non-empty str")
    if "\n" in value or "\r" in value or len(value) > 256:
        raise ToolInvalidArgs("'label' invalid")
    return value


def _read_py_file(path_str: str) -> tuple[str, int]:
    p = Path(path_str)
    if not p.exists():
        raise ToolFailed(f"path not found: {path_str!r}")
    if not p.is_file():
        raise ToolFailed(f"path is not a file: {path_str!r}")
    if p.suffix != ".py":
        raise ToolInvalidArgs(
            f"AST tool only handles .py files, got {p.suffix!r}",
        )
    try:
        size = p.stat().st_size
    except OSError as e:
        raise ToolFailed(f"stat failed: {e}") from e
    if size > DEFAULT_AST_MAX_FILE_BYTES:
        raise ToolFailed(
            f"file too large for AST: {size} > "
            f"{DEFAULT_AST_MAX_FILE_BYTES}",
        )
    try:
        return p.read_text(encoding="utf-8", errors="replace"), size
    except OSError as e:
        raise ToolFailed(f"read failed: {e}") from e


# ── AST walking ────────────────────────────────────────────────────────


def _arg_list(args: ast.arguments) -> list[str]:
    out = []
    for a in args.args:
        out.append(a.arg)
    for a in args.kwonlyargs:
        out.append(a.arg)
    if args.vararg:
        out.append("*" + args.vararg.arg)
    if args.kwarg:
        out.append("**" + args.kwarg.arg)
    return out


def _decorator_names(deco_list) -> list[str]:
    names = []
    for d in deco_list:
        names.append(_node_name(d) or "?")
    return names


def _node_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _node_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _node_name(node.func)
    return ""


def _walk(
    node, scope: list[str], depth: int,
    out: list, include: tuple, max_depth: int,
) -> bool:
    """Walk a node's body. Returns False once max-nodes cap hit."""
    if depth > max_depth:
        return True
    body = getattr(node, "body", None)
    if not body:
        return True
    for child in body:
        if len(out) >= DEFAULT_AST_MAX_NODES:
            return False
        if isinstance(child, ast.FunctionDef):
            if "function" in include:
                out.append({
                    "kind":       "function",
                    "name":       child.name,
                    "lineno":     child.lineno,
                    "end_lineno": getattr(child, "end_lineno", child.lineno),
                    "args":       _arg_list(child.args),
                    "decorators": _decorator_names(child.decorator_list),
                    "scope":      list(scope),
                })
            if not _walk(child, scope + [child.name], depth + 1,
                         out, include, max_depth):
                return False
        elif isinstance(child, ast.AsyncFunctionDef):
            if "async_function" in include:
                out.append({
                    "kind":       "async_function",
                    "name":       child.name,
                    "lineno":     child.lineno,
                    "end_lineno": getattr(child, "end_lineno", child.lineno),
                    "args":       _arg_list(child.args),
                    "decorators": _decorator_names(child.decorator_list),
                    "scope":      list(scope),
                })
            if not _walk(child, scope + [child.name], depth + 1,
                         out, include, max_depth):
                return False
        elif isinstance(child, ast.ClassDef):
            if "class" in include:
                out.append({
                    "kind":       "class",
                    "name":       child.name,
                    "lineno":     child.lineno,
                    "end_lineno": getattr(child, "end_lineno", child.lineno),
                    "bases":      [_node_name(b) for b in child.bases],
                    "decorators": _decorator_names(child.decorator_list),
                    "scope":      list(scope),
                })
            if not _walk(child, scope + [child.name], depth + 1,
                         out, include, max_depth):
                return False
        elif isinstance(child, ast.Import):
            if "import" in include:
                out.append({
                    "kind":   "import",
                    "names":  [n.name for n in child.names],
                    "lineno": child.lineno,
                    "scope":  list(scope),
                })
        elif isinstance(child, ast.ImportFrom):
            if "import_from" in include:
                out.append({
                    "kind":   "import_from",
                    "module": child.module or "",
                    "names":  [n.name for n in child.names],
                    "level":  child.level,
                    "lineno": child.lineno,
                    "scope":  list(scope),
                })
        elif isinstance(child, ast.Assign):
            if "assign" in include:
                targets = [_node_name(t) for t in child.targets]
                out.append({
                    "kind":    "assign",
                    "targets": [t for t in targets if t],
                    "lineno":  child.lineno,
                    "scope":   list(scope),
                })
        elif isinstance(child, ast.AnnAssign):
            if "annotation" in include:
                out.append({
                    "kind":   "annotation",
                    "name":   _node_name(child.target),
                    "lineno": child.lineno,
                    "scope":  list(scope),
                })
    return True


# ── Handler ────────────────────────────────────────────────────────────


def ast_handler(args: dict, ctx: ToolContext) -> dict:
    path = args.get("path")
    text = args.get("text")
    if path is not None and text is not None:
        raise ToolInvalidArgs("cannot provide both 'path' and 'text'")
    if path is None and text is None:
        raise ToolInvalidArgs("must provide either 'path' or 'text'")

    include   = _validate_include(args.get("include"))
    max_depth = _validate_max_depth(args.get("max_depth"))

    if path is not None:
        if not isinstance(path, str) or not path:
            raise ToolInvalidArgs("'path' must be non-empty string")
        if ctx.kernel is not None:
            if not ctx.kernel.cap.check_fs(ctx.pid, path, "r"):
                raise ToolFsDenied(
                    f"agent {ctx.pid} not granted 'r' on {path!r}",
                )
        source, _size = _read_py_file(path)
        label = path
    else:
        if not isinstance(text, str):
            raise ToolInvalidArgs("'text' must be a string")
        source = text
        label = _validate_label(args.get("label"), "<text>")

    line_count = source.count("\n") + (
        0 if source.endswith("\n") or not source else 1
    )

    syntax_error = None
    nodes: list = []
    truncated = False
    try:
        tree = ast.parse(source, filename=label)
    except SyntaxError as e:
        syntax_error = {
            "message": e.msg or "",
            "lineno":  e.lineno or 0,
            "offset":  e.offset or 0,
        }
    else:
        ok = _walk(tree, [], 1, nodes, include, max_depth)
        truncated = not ok

    return {
        "path":         label,
        "nodes":        nodes,
        "syntax_error": syntax_error,
        "line_count":   line_count,
        "truncated":    truncated,
        "include":      list(include),
        "max_depth":    max_depth,
    }


AST_TOOL = Tool(
    name="AST",
    description=(
        "Parse a Python file or text snippet and return a structured "
        "list of definitions (functions, classes, imports, etc.). "
        "Path mode requires 'r' fs_grants. Pure stdlib (no third-party "
        "deps); .py only; up to 5000 nodes."
    ),
    handler=ast_handler,
    requires_capability=True,
    requires_fs=(),     # handler does its own fs check.
)


__all__ = [
    "AST_TOOL", "ast_handler",
    "DEFAULT_AST_MAX_FILE_BYTES", "DEFAULT_AST_MAX_NODES",
    "ALLOWED_KINDS",
]
