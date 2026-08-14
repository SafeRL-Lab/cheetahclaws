"""tools_fs.py — File-system tool implementations: Read, Write, Edit, Glob."""
from __future__ import annotations

import difflib
from pathlib import Path


# A Read call should never materialize an arbitrarily large file (or even an
# arbitrarily long single line) before the agent can apply its output cap.
_LINE_SCAN_BYTES = 8 * 1024
_DEFAULT_READ_MAX_BYTES = 256 * 1024
_DEFAULT_READ_SCAN_MAX_BYTES = 2 * 1024 * 1024
_DEFAULT_READ_MAX_OUTPUT_CHARS = 50_000


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

def _read_logical_line(handle, capture_bytes: int | None, scan_remaining: int):
    """Read one newline-preserving binary line with strict byte budgets.

    Reading bytes rather than ``TextIOWrapper.readline(size)`` prevents a
    UTF-8-heavy line from turning a character limit into a several-times-larger
    byte read.  Any bytes after a line ending are rewound, so a CRLF split at a
    chunk boundary is never exposed as a spurious blank line.
    """
    chunks: list[bytes] = []
    captured = 0
    scanned = 0

    while True:
        remaining = scan_remaining - scanned
        if remaining <= 0:
            return b"".join(chunks), False, False, scanned, False, True

        request_bytes = min(_LINE_SCAN_BYTES, remaining)
        if capture_bytes is not None:
            capture_remaining = capture_bytes - captured
            if capture_remaining <= 0:
                return b"".join(chunks), False, False, scanned, True, False
            request_bytes = min(request_bytes, capture_remaining)
        piece = handle.read(request_bytes)
        if not piece:
            return b"".join(chunks), bool(chunks), True, scanned, False, False

        cr_index = piece.find(b"\r")
        lf_index = piece.find(b"\n")
        newline_index = min(
            (idx for idx in (cr_index, lf_index) if idx >= 0),
            default=-1,
        )
        take = len(piece) if newline_index < 0 else newline_index + 1
        is_cr = newline_index >= 0 and piece[newline_index:newline_index + 1] == b"\r"
        if is_cr and take < len(piece) and piece[take:take + 1] == b"\n":
            take += 1

        tail = piece[take:]
        if tail:
            # ``handle`` is a regular binary file, so this only moves its
            # cursor back over bytes already read in the small bounded chunk.
            handle.seek(-len(tail), 1)
        selected = piece[:take]
        if capture_bytes is not None:
            chunks.append(selected)
            captured += len(selected)
        scanned += len(selected)

        if newline_index >= 0:
            if is_cr and take == len(piece) and piece[take - 1:take] == b"\r":
                # A *bare* CR at the end of a chunk may start a CRLF whose LF
                # was split into the next read. Probe only in that case — a
                # complete CRLF already consumed its LF (last byte is b"\n"),
                # so probing there would swallow a following blank line and
                # shift every later line number.  Probe only when the caller
                # still has budget to retain that final LF.
                can_probe = (
                    scanned < scan_remaining
                    and (capture_bytes is None or captured < capture_bytes)
                )
                if can_probe:
                    next_byte = handle.read(1)
                    if next_byte == b"\n":
                        if capture_bytes is not None:
                            chunks.append(next_byte)
                        scanned += 1
                        captured += 1
                    elif next_byte:
                        handle.seek(-1, 1)
                else:
                    # The caller stopped exactly at CR. Treat this as an
                    # incomplete line rather than later splitting CRLF into
                    # a blank logical line.
                    return (
                        b"".join(chunks), False, False, scanned,
                        capture_bytes is not None and captured >= capture_bytes,
                        scanned >= scan_remaining,
                    )
            return b"".join(chunks), True, False, scanned, False, False

        if capture_bytes is not None and captured >= capture_bytes:
            return b"".join(chunks), False, False, scanned, True, False
        if scanned >= scan_remaining:
            return b"".join(chunks), False, False, scanned, False, True


def _with_stop_marker(rendered: list[str], marker: str, output_limit: int) -> str:
    """Add a stop marker without letting it exceed the configured output cap."""
    marker = marker[:output_limit]
    if not rendered:
        return marker
    visible = "".join(rendered)
    return visible[:max(0, output_limit - len(marker))] + marker


def _read(
    file_path: str,
    limit: int = None,
    offset: int = None,
    max_bytes: int = _DEFAULT_READ_MAX_BYTES,
    scan_max_bytes: int = _DEFAULT_READ_SCAN_MAX_BYTES,
    max_output_chars: int = _DEFAULT_READ_MAX_OUTPUT_CHARS,
) -> str:
    """Stream a numbered text slice with bounded I/O and rendered output."""
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    if p.is_dir():
        return f"Error: {file_path} is a directory"
    try:
        start = max(0, int(offset or 0))
        line_limit = int(limit) if limit else None
        byte_limit = max(1, int(max_bytes or _DEFAULT_READ_MAX_BYTES))
        scan_limit = max(1, int(scan_max_bytes or _DEFAULT_READ_SCAN_MAX_BYTES))
        output_limit = max(1, int(max_output_chars or _DEFAULT_READ_MAX_OUTPUT_CHARS))
        rendered: list[str] = []
        rendered_lines = 0
        source_bytes = 0
        scanned_bytes = 0
        rendered_chars = 0
        line_no = 0
        source_budget_hit = False
        scan_budget_hit = False
        output_budget_hit = False

        file_size = p.stat().st_size
        with p.open("rb") as handle:
            while True:
                if line_limit is not None and rendered_lines >= line_limit:
                    break
                if source_bytes >= byte_limit:
                    # A size check avoids an extra unbounded text-buffer read
                    # solely to distinguish exact EOF from a longer file.
                    source_budget_hit = handle.tell() < file_size
                    break
                if scanned_bytes >= scan_limit:
                    scan_budget_hit = True
                    break

                if line_no < start:
                    _, ended, eof, consumed, _, scan_hit = _read_logical_line(
                        handle, None, scan_limit - scanned_bytes,
                    )
                    scanned_bytes += consumed
                    if not ended:
                        if scan_hit:
                            scan_budget_hit = True
                        if eof or scan_budget_hit:
                            break
                    else:
                        line_no += 1
                    if eof:
                        break
                    continue

                prefix = f"{line_no + 1:6}\t"
                output_room = output_limit - rendered_chars
                if output_room <= len(prefix):
                    output_budget_hit = True
                    break
                source_remaining = byte_limit - source_bytes
                output_content_room = output_room - len(prefix)
                capture_bytes = min(source_remaining, output_content_room)
                source_constrained = source_remaining <= output_content_room
                raw, ended, eof, consumed, capture_hit, scan_hit = _read_logical_line(
                    handle, max(1, capture_bytes), scan_limit - scanned_bytes,
                )
                scanned_bytes += consumed
                if not raw and eof:
                    break
                # A cap can split a multi-byte codepoint. Dropping only the
                # incomplete tail preserves valid UTF-8 and the byte ceiling.
                text = raw.decode(
                    "utf-8", errors="ignore" if capture_hit or scan_hit else "replace",
                )
                line_no += 1
                source_bytes += len(raw)
                formatted = prefix + text
                rendered.append(formatted)
                rendered_chars += len(formatted)
                rendered_lines += 1
                if capture_hit:
                    if source_constrained:
                        source_budget_hit = handle.tell() < file_size
                    else:
                        output_budget_hit = True
                    break
                if scan_hit:
                    scan_budget_hit = True
                    break
                if not ended:
                    # EOF after a final line with no terminator is still a
                    # valid logical line; any other incomplete line hit a cap.
                    if not eof:
                        scan_budget_hit = True
                    break

        if scan_budget_hit:
            marker = (
                f"[... Read stopped after scanning {scan_limit:,} bytes; "
                "use a smaller offset or a narrower file ...]\n"
            )
            return _with_stop_marker(rendered, "\n" + marker if rendered else marker, output_limit)
        elif source_budget_hit:
            marker = (
                f"[... Read stopped after {byte_limit:,} source bytes; use "
                "offset and limit to request another line range ...]\n"
            )
            return _with_stop_marker(rendered, "\n" + marker if rendered else marker, output_limit)
        elif output_budget_hit:
            marker = (
                f"[... Read output capped at {output_limit:,} characters to "
                "keep memory and model context bounded ...]\n"
            )
            return _with_stop_marker(rendered, "\n" + marker if rendered else marker, output_limit)
        if not rendered:
            return "(empty file)"
        return "".join(rendered)
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
