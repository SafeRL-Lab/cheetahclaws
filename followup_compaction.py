"""Follow-up compaction: stub past-turn tool_results before each API call.

Non-destructive: produces a new message list, leaves `state.messages` intact
so persistence and resume keep the full history.
"""
from __future__ import annotations

import html
import json
import re
import time
from typing import Iterable

DEFAULT_EXEMPT_TOOLS = frozenset({"Edit", "Write", "TodoWrite"})


def compact_tool_history(
    messages: list,
    keep_last_n_turns: int = 0,
    exempt_tools: Iterable[str] = DEFAULT_EXEMPT_TOOLS,
) -> list:
    """Return a NEW list where past-turn tool_result contents are replaced by stubs.

    A "turn" begins at a role='user' message. The current turn (from the last
    user message onward) is always kept intact.
    """
    exempt = frozenset(exempt_tools)
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_indices) <= keep_last_n_turns + 1:
        return list(messages)

    cutoff = user_indices[-(keep_last_n_turns + 1)]
    tool_call_lookup = _build_tool_call_lookup(messages)

    compacted = []
    for index, message in enumerate(messages):
        if index >= cutoff:
            compacted.append(message)
            continue
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            stubbed = dict(message)
            stubbed["content"] = compact_assistant_xml(
                message["content"], message.get("tool_calls")
            )
            compacted.append(stubbed)
            continue
        if role != "tool" or message.get("name") in exempt:
            compacted.append(message)
            continue
        tool_call_id = message.get("tool_call_id", "")
        name, inp = tool_call_lookup.get(
            tool_call_id, (message.get("name", "tool"), {})
        )
        stubbed = dict(message)
        stubbed["content"] = _build_stub(name, inp)
        compacted.append(stubbed)
    return compacted


def _build_tool_call_lookup(messages: list) -> dict:
    lookup: dict = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            lookup[tool_call.get("id", "")] = (
                tool_call.get("name", ""),
                tool_call.get("input") or {},
            )
    return lookup


def _escape_xml_attr(value: str) -> str:
    return html.escape(value, quote=False).replace('"', '&quot;')


def _build_stub(name: str, input_dict: dict) -> str:
    brief = _input_brief(name, input_dict)
    return f'<tool_use_elided name="{_escape_xml_attr(name)}" brief="{_escape_xml_attr(brief)}"/>'


def _input_brief(name: str, inp: dict) -> str:
    if name == "Read":
        path = inp.get("file_path", "?")
        parts = [f"file_path={path}"]
        if "offset" in inp:
            parts.append(f"offset={inp['offset']}")
        if "limit" in inp:
            parts.append(f"limit={inp['limit']}")
        return ", ".join(parts)
    if name == "Bash":
        cmd = (inp.get("command") or "").replace("\n", " ")
        if len(cmd) > 100:
            cmd = cmd[:97] + "..."
        return f"command={cmd!r}"
    if name == "Grep":
        parts = [f"pattern={inp.get('pattern', '?')!r}"]
        if "path" in inp:
            parts.append(f"path={inp['path']}")
        return ", ".join(parts)
    if name == "Glob":
        return f"pattern={inp.get('pattern', '?')!r}"
    try:
        rendered = json.dumps(inp, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = str(inp)
    if len(rendered) > 120:
        rendered = rendered[:117] + "..."
    return rendered


def _build_tc_lookup(tool_calls: list | None) -> dict:
    lookup: dict = {}
    for tc in tool_calls or []:
        tid = tc.get("id", "")
        if tid:
            lookup[tid] = (tc.get("name", "tool"), tc.get("input") or {})
    return lookup


def _xml_replacer(tc_lookup: dict, target_ids: set | None = None):
    def _replacer(match):
        name, tid = match.group(1), match.group(2)
        if target_ids is not None and tid not in target_ids:
            return match.group(0)
        tc_name, tc_input = tc_lookup.get(tid, (name, {}))
        brief = _input_brief(tc_name, tc_input)
        return f'<tool_use_elided name="{_escape_xml_attr(tc_name)}" brief="{_escape_xml_attr(brief)}"/>'
    return _replacer


_TOOL_USE_RE = re.compile(
    r'<tool_use\s+name="([^"]+)"\s+id="([^"]+)"[^>]*>.*?</tool_use>',
    re.DOTALL,
)


def compact_assistant_xml(content: str, tool_calls: list | None = None) -> str:
    """Replace ALL inline XML tool_use blocks with one-line summaries."""
    if not content or "<tool_use" not in content:
        return content
    return _TOOL_USE_RE.sub(
        _xml_replacer(_build_tc_lookup(tool_calls)), content,
    )


def compact_assistant_xml_selective(
    content: str, tool_calls: list | None, target_ids: set,
) -> str:
    """Replace only XML blocks whose id is in target_ids, leaving others intact."""
    if not content or "<tool_use" not in content or not target_ids:
        return content
    return _TOOL_USE_RE.sub(
        _xml_replacer(_build_tc_lookup(tool_calls), target_ids), content,
    )


def build_messages_for_api(state, config: dict) -> list:
    """Apply follow-up compaction + model-driven GC, then inject working memory notes."""
    if not config.get("followup_compaction_enabled", True):
        compacted = list(state.messages)
    else:
        keep = config.get("followup_keep_last_n_turns", 0)
        exempt = config.get("followup_exempt_tools", DEFAULT_EXEMPT_TOOLS)
        compacted = compact_tool_history(state.messages, keep_last_n_turns=keep, exempt_tools=exempt)

        from compaction import estimate_tokens
        tokens_before = estimate_tokens(state.messages)
        tokens_after = estimate_tokens(compacted)
        if tokens_before != tokens_after:
            state.compaction_log.append({
                "event": "followup_compact",
                "timestamp": time.time(),
                "turn": getattr(state, "turn_count", 0),
                "tokens_est_before": tokens_before,
                "tokens_est_after": tokens_after,
                "tokens_est_saved": tokens_before - tokens_after,
            })

    return _apply_context_gc(compacted, state)


def _apply_context_gc(messages: list, state) -> list:
    """Apply model-driven GC decisions and inject working memory notes."""
    try:
        try:
            from context_gc import apply_gc
        except ImportError:
            return messages  # context_gc not available yet, skip, inject_notes, prepend_verbatim_audit
    except ImportError:
        return messages
    gc_state = getattr(state, 'gc_state', None)
    if not gc_state:
        return prepend_verbatim_audit(messages)
    if not gc_state.trashed_ids and not gc_state.snippets and not gc_state.notes:
        return prepend_verbatim_audit(messages)

    from compaction import estimate_tokens
    tokens_before = estimate_tokens(messages)
    result = apply_gc(messages, gc_state)
    result = inject_notes(result, gc_state.notes)
    tokens_after = estimate_tokens(result)
    if tokens_before != tokens_after:
        state.compaction_log.append({
            "event": "context_gc",
            "timestamp": time.time(),
            "turn": getattr(state, "turn_count", 0),
            "trashed_count": len(gc_state.trashed_ids),
            "snippet_count": len(gc_state.snippets),
            "notes_count": len(gc_state.notes),
            "tokens_est_saved": tokens_before - tokens_after,
        })
    return prepend_verbatim_audit(result)
