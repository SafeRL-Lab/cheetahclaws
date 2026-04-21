"""Model-driven context garbage collection for conversation history.

Lets the LLM trash consumed tool results, keep relevant snippets,
and persist notes across turns to manage its context window.

Flat-file port of bouzecode's context_gc/ package — combines:
  state (GCState, process_gc_call, note_save, note_read)
  apply (apply_gc, snippet handling)
  notes (inject_notes)
  audit (build_verbatim_audit_note, prepend_verbatim_audit)
  stubs (strip_trashed_stubs, _is_stub, _is_auto_trashed_stub)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field


# ── Constants ──────────────────────────────────────────────────────────────

METHODOLOGY_NOTE = "methodology"


# ── Stub detection ─────────────────────────────────────────────────────────

_ELIDED_RE = re.compile(r'\s*<tool_use_elided[^/]*/>')

# Only matches stubs produced by apply_gc (model-driven trash).
# Does NOT match <tool_use_elided/> breadcrumbs from compact_tool_history —
# those must survive so the model retains a trace of prior tool calls.
_TRASHED_STUB_RE = re.compile(r'^\[.{1,60} -- (?:trashed by model|auto-trashed)\]$')
_AUTO_TRASHED_RE = re.compile(r'^\[.{1,60} -- auto-trashed\]$')


def _is_stub(content: str) -> bool:
    """Return True for any GC stub (model-trashed OR auto-trashed).

    Used by audit to skip all stubs in the verbatim audit note.
    """
    if not content or len(content) > 200:
        return False
    stripped = content.strip()
    if not stripped:
        return False
    return bool(_TRASHED_STUB_RE.match(stripped))


def _is_auto_trashed_stub(content: str) -> bool:
    """Return True only for auto-trashed stubs (ContextGC's own results)."""
    if not content or len(content) > 200:
        return False
    stripped = content.strip()
    if not stripped:
        return False
    return bool(_AUTO_TRASHED_RE.match(stripped))


# ── GCState ────────────────────────────────────────────────────────────────

@dataclass
class GCState:
    trashed_ids: set = field(default_factory=set)
    snippets: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    compact_xml: bool = False


# ── Process ContextGC tool call ────────────────────────────────────────────

def process_gc_call(params: dict, config: dict) -> str:
    gc_state: GCState = config.get("_gc_state")
    if gc_state is None:
        return "Error: no GC state available"

    trashed = params.get("trash") or []
    snippets = params.get("keep_snippets") or []
    notes = params.get("notes") or []
    trash_notes = params.get("trash_notes") or []

    notes_before = dict(gc_state.notes)

    for tid in trashed:
        gc_state.trashed_ids.add(tid)
        gc_state.snippets.pop(tid, None)

    for snippet in snippets:
        sid = snippet.get("id")
        if sid and sid not in gc_state.trashed_ids:
            gc_state.snippets[sid] = snippet

    for note in notes:
        name = note.get("name")
        content = note.get("content", "")
        if name:
            gc_state.notes[name] = content

    methodology_protected = False
    for name in trash_notes:
        if name == METHODOLOGY_NOTE:
            methodology_protected = True
            continue
        gc_state.notes.pop(name, None)

    if params.get("compact_xml"):
        gc_state.compact_xml = True

    # Track notes timeline
    added = [k for k in gc_state.notes if k not in notes_before]
    updated = [k for k in gc_state.notes if k in notes_before and gc_state.notes[k] != notes_before[k]]
    removed = [k for k in notes_before if k not in gc_state.notes]
    if added or updated or removed:
        state = config.get("_state")
        if state is not None and hasattr(state, "notes_timeline"):
            state.notes_timeline.append({
                "turn": getattr(state, "turn_count", 0),
                "timestamp": time.time(),
                "notes": dict(gc_state.notes),
                "delta": {"added": added, "updated": updated, "removed": removed},
            })

    parts = []
    if trashed:
        parts.append(f"trashed {len(trashed)} results")
    if snippets:
        parts.append(f"kept snippets for {len(snippets)} results")
    if notes:
        parts.append(f"{len(notes)} notes saved")
    if trash_notes:
        trashed_count = len(trash_notes) - (1 if methodology_protected else 0)
        if trashed_count:
            parts.append(f"{trashed_count} notes removed")
    if methodology_protected:
        parts.append(f"note '{METHODOLOGY_NOTE}' protected from trash")
    if params.get("compact_xml"):
        parts.append("XML compaction enabled")
    parts.append(f"{len(gc_state.notes)} active notes, {len(gc_state.trashed_ids)} total trashed")
    return "GC applied: " + ", ".join(parts)


# ── NoteSave / NoteRead ───────────────────────────────────────────────────

def note_save(params: dict, config: dict) -> str:
    gc_state: GCState = config.get("_gc_state")
    if gc_state is None:
        return "Error: no GC state available"

    name = params.get("name", "")
    content = params.get("content", "")
    if not name:
        return "Error: 'name' is required"

    notes_before = dict(gc_state.notes)
    gc_state.notes[name] = content

    is_new = name not in notes_before
    changed = not is_new and notes_before[name] != content
    if is_new or changed:
        state = config.get("_state")
        if state is not None and hasattr(state, "notes_timeline"):
            state.notes_timeline.append({
                "turn": getattr(state, "turn_count", 0),
                "timestamp": time.time(),
                "notes": dict(gc_state.notes),
                "delta": {
                    "added": [name] if is_new else [],
                    "updated": [name] if changed else [],
                    "removed": [],
                },
            })

    action = "created" if is_new else ("updated" if changed else "unchanged")
    return f"Note '{name}' {action}. {len(gc_state.notes)} active notes."


def note_read(params: dict, config: dict) -> str:
    gc_state: GCState = config.get("_gc_state")
    if gc_state is None:
        return "Error: no GC state available"

    name = params.get("name")
    if name:
        content = gc_state.notes.get(name)
        if content is None:
            available = ", ".join(sorted(gc_state.notes)) or "(none)"
            return f"Note '{name}' not found. Active notes: {available}"
        return f"## {name}\n{content}"

    if not gc_state.notes:
        return "No active notes."
    parts = []
    for n, c in gc_state.notes.items():
        parts.append(f"## {n}\n{c}")
    return "\n\n".join(parts)


# ── Apply GC (transform messages before API call) ─────────────────────────

def apply_gc(messages: list, gc_state: GCState) -> list:
    if not gc_state.trashed_ids and not gc_state.snippets and not gc_state.compact_xml:
        return messages

    _compact_all = None
    _compact_selective = None
    last_asst_idx = -1

    if gc_state.compact_xml:
        from followup_compaction import compact_assistant_xml
        _compact_all = compact_assistant_xml
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_asst_idx = i
                break

    if gc_state.trashed_ids:
        from followup_compaction import compact_assistant_xml_selective
        _compact_selective = compact_assistant_xml_selective

    result = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            if _compact_all and idx != last_asst_idx:
                stubbed = dict(msg)
                stubbed["content"] = _compact_all(msg["content"], msg["tool_calls"])
                result.append(stubbed)
                continue
            if _compact_selective:
                tc_ids = {tc.get("id") for tc in msg["tool_calls"]}
                targeted = tc_ids & gc_state.trashed_ids
                if targeted:
                    stubbed = dict(msg)
                    stubbed["content"] = _compact_selective(
                        msg["content"], msg["tool_calls"], targeted,
                    )
                    result.append(stubbed)
                    continue
            result.append(msg)
            continue
        if role != "tool":
            result.append(msg)
            continue
        tc_id = msg.get("tool_call_id", "")
        if tc_id in gc_state.trashed_ids:
            stubbed = dict(msg)
            name = msg.get("name", "tool")
            stubbed["content"] = f"[{name} result -- trashed by model]"
            result.append(stubbed)
        elif tc_id in gc_state.snippets:
            transformed = dict(msg)
            transformed["content"] = _apply_snippet(msg["content"], gc_state.snippets[tc_id])
            result.append(transformed)
        else:
            result.append(msg)
    return result


def _apply_snippet(content: str, snippet: dict) -> str:
    if not content:
        return content
    lines = content.split("\n")

    if "keep_after" in snippet:
        anchor = snippet["keep_after"]
        idx = _find_anchor_line(lines, anchor)
        if idx is None:
            return content + f"\n[GC warning: anchor {anchor!r} not found, kept full result]"
        kept = lines[idx:]
        trimmed = len(lines) - len(kept)
        return f"[{trimmed} lines trimmed, kept after {anchor!r}]\n" + "\n".join(kept)

    if "keep_before" in snippet:
        anchor = snippet["keep_before"]
        idx = _find_anchor_line(lines, anchor)
        if idx is None:
            return content + f"\n[GC warning: anchor {anchor!r} not found, kept full result]"
        kept = lines[:idx]
        trimmed = len(lines) - len(kept)
        return "\n".join(kept) + f"\n[{trimmed} lines trimmed at {anchor!r}]"

    if "keep_between" in snippet:
        anchors = snippet["keep_between"]
        if len(anchors) != 2:
            return content + "\n[GC warning: keep_between needs exactly 2 anchors]"
        start_anchor, end_anchor = anchors
        start_idx = _find_anchor_line(lines, start_anchor)
        if start_idx is None:
            return content + f"\n[GC warning: start anchor {start_anchor!r} not found]"
        end_idx = _find_anchor_line(lines, end_anchor, start_from=start_idx)
        if end_idx is None:
            return content + f"\n[GC warning: end anchor {end_anchor!r} not found]"
        kept = lines[start_idx:end_idx + 1]
        before = start_idx
        after = len(lines) - end_idx - 1
        header = f"[{before} lines trimmed before {start_anchor!r}]"
        footer = f"[{after} lines trimmed after {end_anchor!r}]"
        return header + "\n" + "\n".join(kept) + "\n" + footer

    return content


def _find_anchor_line(lines: list, text: str, start_from: int = 0) -> int | None:
    for i in range(start_from, len(lines)):
        if text in lines[i]:
            return i
    return None


# ── Notes injection ───────────────────────────────────────────────────────

def inject_notes(messages: list, notes: dict) -> list:
    if not notes:
        return messages
    parts = []
    for name, content in notes.items():
        parts.append(f"## {name}\n{content}")
    notes_block = "[Your working memory notes]\n" + "\n\n".join(parts) + "\n[/Notes]"
    result = list(messages)
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "user":
            result[i] = dict(result[i])
            result[i]["content"] = notes_block + "\n\n" + result[i]["content"]
            break
    return result


# ── Verbatim audit ────────────────────────────────────────────────────────

_ARGS_PREFERRED_KEY = {
    "Read": "file_path", "Edit": "file_path", "Write": "file_path",
    "NotebookEdit": "notebook_path",
    "Glob": "pattern", "Grep": "pattern",
    "Bash": "command",
    "WebFetch": "url", "WebSearch": "query",
}


def _summarize_args(tool_name: str, input_dict: dict, max_len: int = 60) -> str:
    if not input_dict:
        return ""
    val = input_dict.get(_ARGS_PREFERRED_KEY.get(tool_name, ""))
    if val is None:
        for v in input_dict.values():
            if isinstance(v, str) and v:
                val = v
                break
    if val is None:
        return ""
    val = str(val).replace("\n", " ")
    if len(val) > max_len:
        val = val[: max_len - 3] + "..."
    return val


def build_verbatim_audit_note(messages: list) -> str:
    """List every tool_result still kept verbatim with its token size.

    Each entry includes the tool's key arg (file_path, pattern, command...)
    so the model can correlate notes with results already in context.
    """
    from compaction import estimate_tokens
    args_by_id: dict[str, dict] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tc in message.get("tool_calls") or []:
            tc_id = tc.get("id")
            if tc_id:
                args_by_id[tc_id] = tc.get("input") or {}
    lines = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        if _is_stub(content):
            continue
        tool_call_id = message.get("tool_call_id", "?")
        tool_name = message.get("name", "?")
        size = estimate_tokens([{"content": content}])
        args = _summarize_args(tool_name, args_by_id.get(tool_call_id, {}))
        suffix = f" {args}" if args else ""
        lines.append(f"- {tool_call_id} ({tool_name}{suffix}): {size} tk")
    if not lines:
        return ""
    return (
        "[Verbatim tool_results still in your context -- trash any you've already consumed]\n"
        + "\n".join(lines)
        + "\n[/Verbatim audit]"
    )


def prepend_verbatim_audit(messages: list) -> list:
    """Prepend the verbatim audit note to the last user message."""
    note = build_verbatim_audit_note(messages)
    if not note:
        return messages
    result = list(messages)
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "user":
            result[i] = dict(result[i])
            result[i]["content"] = note + "\n\n" + result[i]["content"]
            break
    return result


# ── Strip auto-trashed stubs ─────────────────────────────────────────────

def strip_trashed_stubs(messages: list) -> list:
    """Remove auto-trashed tool messages and their tool_call entries entirely."""
    stubbed_ids = set()
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if _is_auto_trashed_stub(content):
                tc_id = msg.get("tool_call_id", "")
                if tc_id:
                    stubbed_ids.add(tc_id)
    if not stubbed_ids:
        return messages
    result = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool" and msg.get("tool_call_id", "") in stubbed_ids:
            continue
        if role == "assistant" and msg.get("tool_calls"):
            original_tcs = msg["tool_calls"]
            remaining = [tc for tc in original_tcs if tc.get("id") not in stubbed_ids]
            if len(remaining) == len(original_tcs):
                result.append(msg)
                continue
            cleaned = dict(msg)
            content = cleaned.get("content", "") or ""
            if not remaining:
                content = _ELIDED_RE.sub("", content).strip()
                cleaned.pop("tool_calls", None)
            else:
                cleaned["tool_calls"] = remaining
            cleaned["content"] = content
            result.append(cleaned)
            continue
        result.append(msg)
    return result
