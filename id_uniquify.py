"""Prevent tool_call_id collisions across turns.

LLMs pick short ids (e.g. ``r1``, ``w2``) and reuse them freely across turns.
When a previous id is reused, it can cause API errors or wrong tool results
being associated with wrong calls.

The fix: on ingest, rewrite any id that already exists in the message history
to ``t{turn}_{original}`` (with a numeric suffix if that still collides).
Same-turn ``depends_on`` references are rewritten in lockstep so the DAG
resolves correctly.
"""
from __future__ import annotations


def _collect_used_ids(state) -> set[str]:
    """Gather all tool_call ids already present in state.messages."""
    used: set[str] = set()
    for msg in state.messages:
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                tid = tc.get("id")
                if tid:
                    used.add(tid)
        elif role == "tool":
            tid = msg.get("tool_call_id")
            if tid:
                used.add(tid)
    return used


def _pick_fresh_id(original: str, turn: int, used: set[str]) -> str:
    candidate = f"t{turn}_{original}"
    if candidate not in used:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in used:
        suffix += 1
    return f"{candidate}_{suffix}"


def uniquify_tool_call_ids(tool_calls: list, state) -> dict[str, str]:
    """Rewrite colliding tool_call ids in-place and rewrite depends_on refs.

    Only ids that already exist in state.messages are remapped.  New ids
    pass through unchanged, preserving behaviour for simple sessions.

    Returns:
        Mapping of ``{original_id: new_id}`` for any ids that were remapped.
    """
    if not tool_calls:
        return {}

    used = _collect_used_ids(state)
    remap: dict[str, str] = {}

    for tc in tool_calls:
        original = tc.get("id")
        if not original or original not in used:
            if original:
                used.add(original)
            continue
        fresh = _pick_fresh_id(original, state.turn_count, used)
        remap[original] = fresh
        tc["id"] = fresh
        used.add(fresh)

    if not remap:
        return {}

    # Rewrite depends_on references inside same-turn tool calls
    for tc in tool_calls:
        params = tc.get("input") or {}
        deps = params.get("depends_on")
        if deps:
            params["depends_on"] = [remap.get(d, d) for d in deps]

    return remap
