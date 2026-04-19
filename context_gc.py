"""Model-driven context garbage collection for conversation history.

Lets the LLM trash consumed tool results, keep relevant snippets,
and persist notes across turns to manage its context window.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GCState:
    trashed_ids: set = field(default_factory=set)
    snippets: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)


def process_gc_call(params: dict, config: dict) -> str:
    gc_state: GCState = config.get("_gc_state")
    if gc_state is None:
        return "Error: no GC state available"

    trashed = params.get("trash") or []
    snippets = params.get("keep_snippets") or []
    notes = params.get("notes") or []
    trash_notes = params.get("trash_notes") or []

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

    for name in trash_notes:
        gc_state.notes.pop(name, None)

    parts = []
    if trashed:
        parts.append(f"trashed {len(trashed)} results")
    if snippets:
        parts.append(f"kept snippets for {len(snippets)} results")
    if notes:
        parts.append(f"{len(notes)} notes saved")
    if trash_notes:
        parts.append(f"{len(trash_notes)} notes removed")
    parts.append(f"{len(gc_state.notes)} active notes, {len(gc_state.trashed_ids)} total trashed")
    return "GC applied: " + ", ".join(parts)


def apply_gc(messages: list, gc_state: GCState) -> list:
    """Return a new message list with trashed tool_results stubbed and kept snippets trimmed.

    Non-destructive: original messages are preserved in state.messages so /save + /load
    can restore the full conversation. Only the message list sent to the API is reshaped.
    """
    if not gc_state.trashed_ids and not gc_state.snippets:
        return messages
    return [_apply_gc_to_message(msg, gc_state) for msg in messages]


def _apply_gc_to_message(msg: dict, gc_state: GCState) -> dict:
    """Apply GC rules to a single message; returns original msg untouched if no rule matches."""
    if msg.get("role") != "tool":
        return msg
    tool_call_id = msg.get("tool_call_id", "")
    if tool_call_id in gc_state.trashed_ids:
        return _stub_trashed_tool_result(msg)
    if tool_call_id in gc_state.snippets:
        return _apply_snippet_to_message(msg, gc_state.snippets[tool_call_id])
    return msg


def _stub_trashed_tool_result(msg: dict) -> dict:
    """Replace a tool_result's content with a short stub the model can recognise."""
    stubbed = dict(msg)
    name = msg.get("name", "tool")
    stubbed["content"] = f"[{name} result -- trashed by model]"
    return stubbed


def _apply_snippet_to_message(msg: dict, snippet: dict) -> dict:
    """Apply a keep_{after,before,between} snippet rule to a tool_result message."""
    transformed = dict(msg)
    transformed["content"] = _apply_snippet(msg.get("content", ""), snippet)
    return transformed


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


def build_verbatim_audit_note(messages: list) -> str:
    from compaction import estimate_tokens
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
        if "<tool_use_elided" in content or "trashed by model" in content:
            continue
        tool_call_id = message.get("tool_call_id", "?")
        tool_name = message.get("name", "?")
        size = estimate_tokens([{"content": content}])
        lines.append(f"- {tool_call_id} ({tool_name}): {size} tk")
    if not lines:
        return ""
    return (
        "[Verbatim tool_results still in your context -- trash any you've already consumed]\n"
        + "\n".join(lines)
        + "\n[/Verbatim audit]"
    )


def prepend_verbatim_audit(messages: list) -> list:
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
