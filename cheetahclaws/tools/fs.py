"""tools_fs.py — File-system tool implementations: Read, Write, Edit, Glob."""
from __future__ import annotations

import difflib
from pathlib import Path


def _read_preserving_newlines(p: Path) -> str:
    """Read a text file without newline translation.

    Path.read_text gained a `newline=` parameter only in Python 3.14; the
    project supports 3.10+, so we use open() which has accepted `newline=`
    since the pathlib API was introduced.
    """
    with p.open(encoding="utf-8", errors="replace", newline="") as f:
        return f.read()


# ── Diff helpers ──────────────────────────────────────────────────────────

def generate_unified_diff(old: str, new: str, filename: str,
                           context_lines: int = 3) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}",
        n=context_lines,
    )
    return "".join(diff)


def maybe_truncate_diff(diff_text: str, max_lines: int = 80) -> str:
    lines = diff_text.splitlines()
    if len(lines) <= max_lines:
        return diff_text
    shown     = lines[:max_lines]
    remaining = len(lines) - max_lines
    return "\n".join(shown) + f"\n\n[... {remaining} more lines ...]"


# ── Read ─────────────────────────────────────────────────────────────────

def _read(file_path: str, limit: int = None, offset: int = None) -> str:
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    if p.is_dir():
        return f"Error: {file_path} is a directory"
    try:
        lines = _read_preserving_newlines(p).splitlines(keepends=True)
        start = offset or 0
        chunk = lines[start:start + limit] if limit else lines[start:]
        if not chunk:
            return "(empty file)"
        return "".join(f"{start + i + 1:6}\t{l}" for i, l in enumerate(chunk))
    except Exception as e:
        return f"Error: {e}"


# ── Write ─────────────────────────────────────────────────────────────────

def _write(file_path: str, content: str) -> str:
    p = Path(file_path)
    try:
        is_new      = not p.exists()
        old_content = "" if is_new else _read_preserving_newlines(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="")
        if is_new:
            lc = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Created {file_path} ({lc} lines)"
        diff = generate_unified_diff(old_content, content, p.name)
        if not diff:
            return f"No changes in {file_path}"
        return f"File updated — {file_path}:\n\n{maybe_truncate_diff(diff)}"
    except Exception as e:
        return f"Error: {e}"


# ── Edit ──────────────────────────────────────────────────────────────────

def _edit(file_path: str, old_string: str, new_string: str,
          replace_all: bool = False) -> str:
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    try:
        content = _read_preserving_newlines(p)

        crlf_count = content.count("\r\n")
        lf_count   = content.count("\n")
        is_pure_crlf = crlf_count > 0 and crlf_count == lf_count

        content_norm = content.replace("\r\n", "\n")
        old_norm     = old_string.replace("\r\n", "\n")
        new_norm     = new_string.replace("\r\n", "\n")

        count = content_norm.count(old_norm)
        if count == 0:
            return ("Error: old_string not found in file. Please ensure EXACT match, "
                    "including all exact leading spaces/indentation and trailing newlines.")
        if count > 1 and not replace_all:
            return (f"Error: old_string appears {count} times. "
                    "Provide more context to make it unique, or use replace_all=true.")

        if replace_all:
            new_content_norm = content_norm.replace(old_norm, new_norm)
        else:
            new_content_norm = content_norm.replace(old_norm, new_norm, 1)

        if is_pure_crlf:
            final_content    = new_content_norm.replace("\n", "\r\n")
            old_content_final = content
        else:
            final_content    = new_content_norm
            old_content_final = content_norm

        p.write_text(final_content, encoding="utf-8", newline="")
        diff = generate_unified_diff(old_content_final, final_content, p.name)
        return f"Changes applied to {p.name}:\n\n{diff}"
    except Exception as e:
        return f"Error: {e}"


# ── Glob ──────────────────────────────────────────────────────────────────

def _glob(pattern: str, path: str = None, cwd: str = None) -> str:
    base = Path(path) if path else (Path(cwd) if cwd else Path.cwd())
    try:
        matches = sorted(base.glob(pattern))
        if not matches:
            return "No files matched"
        return "\n".join(str(m) for m in matches[:500])
    except Exception as e:
        return f"Error: {e}"
