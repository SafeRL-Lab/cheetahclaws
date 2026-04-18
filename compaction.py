"""Context window management: two-layer compression for long conversations."""
from __future__ import annotations

import providers


# ── Token estimation ──────────────────────────────────────────────────────

def _count_str_chars(obj) -> int:
    """Recursively count total characters across all string values in a nested structure."""
    if isinstance(obj, str):
        return len(obj)
    if isinstance(obj, dict):
        return sum(_count_str_chars(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_str_chars(item) for item in obj)
    return 0


def estimate_tokens(messages: list) -> int:
    """Estimate token count. Uses chars/2.8 (conservative for code-heavy content).

    The old chars/3.5 divisor underestimated real token counts for code-heavy
    conversations because: (1) code tokens are ~2.5-3 chars each, not 3.5,
    (2) tool schemas, JSON keys, and special chars take more tokens than plain
    text, (3) per-message framing overhead (~4 tokens/msg) is not counted.
    This caused compaction to skip when it should have triggered, leading to
    context overflow crashes.

    Args:
        messages: list of message dicts with "content" field (str or list of dicts)
    Returns:
        approximate token count, int
    """
    total_chars = 0
    msg_count = 0
    for m in messages:
        msg_count += 1
        content = m.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for v in block.values():
                        if isinstance(v, str):
                            total_chars += len(v)
        for tc in m.get("tool_calls", []):
            # Recursively count all string values, including nested input dicts
            # (e.g. {"id": "c1", "name": "Bash", "input": {"command": "..."}})
            total_chars += _count_str_chars(tc)
    # chars/2.8 for content + 4 tokens/msg framing overhead + 10% buffer
    content_tokens = int(total_chars / 2.8)
    framing_tokens = msg_count * 4
    return int((content_tokens + framing_tokens) * 1.1)


def get_context_limit(model: str) -> int:
    """Look up context window size for a model.

    Args:
        model: model string (e.g. "claude-opus-4-6", "ollama/llama3.3")
    Returns:
        context limit in tokens
    """
    provider_name = providers.detect_provider(model)
    prov = providers.PROVIDERS.get(provider_name, {})
    return prov.get("context_limit", 128000)


# ── Layer 1: Snip old tool results ────────────────────────────────────────

def snip_old_tool_results(
    messages: list,
    max_chars: int = 2000,
    preserve_last_n_turns: int = 6,
) -> list:
    """Truncate tool-role messages older than preserve_last_n_turns from end.

    For old tool messages whose content exceeds max_chars, keep the first half
    and last quarter, inserting '[... N chars snipped ...]' in between.
    Mutates in place and returns the same list.

    Args:
        messages: list of message dicts (mutated in place)
        max_chars: maximum character length before truncation
        preserve_last_n_turns: number of messages from end to preserve
    Returns:
        the same messages list (mutated)
    """
    cutoff = max(0, len(messages) - preserve_last_n_turns)
    for i in range(cutoff):
        m = messages[i]
        if m.get("role") != "tool":
            continue
        content = m.get("content", "")
        if not isinstance(content, str) or len(content) <= max_chars:
            continue
        first_half = content[: max_chars // 2]
        last_quarter = content[-(max_chars // 4):]
        snipped = len(content) - len(first_half) - len(last_quarter)
        m["content"] = f"{first_half}\n[... {snipped} chars snipped ...]\n{last_quarter}"
    return messages


# ── Layer 2: Auto-compact ─────────────────────────────────────────────────

def _respect_tool_pairs(messages: list, split: int) -> int:
    """Advance split so it never falls inside a tool_calls → tool-response block.

    OpenAI-compatible APIs (DeepSeek, etc.) reject any 'tool' message that is
    not preceded by an 'assistant' with matching tool_calls. If the split lands
    between an assistant(tool_calls) and its tool responses, the recent half
    would contain orphan tool messages after compaction.
    """
    n = len(messages)
    if split <= 0 or split >= n:
        return split
    prev = messages[split - 1]
    if prev.get("role") == "assistant" and (prev.get("tool_calls") or []):
        j = split
        while j < n and messages[j].get("role") == "tool":
            j += 1
        split = j
    while split < n and messages[split].get("role") == "tool":
        split += 1
    return split


def find_split_point(messages: list, keep_ratio: float = 0.3) -> int:
    """Find index that splits messages so ~keep_ratio of tokens are in the recent portion.

    Walks backwards from end, accumulating token estimates, and returns the
    index where the recent portion reaches ~keep_ratio of total tokens. The
    index is then adjusted so it never cuts a tool-call response block.

    Args:
        messages: list of message dicts
        keep_ratio: fraction of tokens to keep in the recent portion
    Returns:
        split index (messages[:idx] = old, messages[idx:] = recent).
        Returns 0 if no safe split exists (caller should skip compaction).
    """
    if not messages:
        return 0
    keep_ratio = max(0.0, min(1.0, keep_ratio))
    total = estimate_tokens(messages)
    target = int(total * keep_ratio)
    running = 0
    raw = 0
    for i in range(len(messages) - 1, -1, -1):
        running += estimate_tokens([messages[i]])
        if running >= target:
            raw = i
            break
    adjusted = _respect_tool_pairs(messages, raw)
    if adjusted >= len(messages):
        return 0
    return adjusted


def sanitize_history(messages: list) -> list:
    """Enforce the tool-calls ↔ tool-response invariant required by OpenAI-compatible APIs.

    Walks the list in order maintaining a set of pending tool_call_ids from the
    most recent assistant(tool_calls). Drops any 'tool' message whose
    tool_call_id is not in that set (orphan). When a non-tool message arrives
    with pending ids still open, strips those unanswered tool_calls from the
    preceding assistant message (so DeepSeek won't reject it).

    Returns a new list; the input is not mutated.
    """
    cleaned: list = []
    pending: set[str] = set()

    def _strip_unanswered():
        if not pending:
            return
        # Walk back past any trailing tool messages to reach the assistant that owns them.
        target = None
        for k in range(len(cleaned) - 1, -1, -1):
            role_k = cleaned[k].get("role")
            if role_k == "tool":
                continue
            if role_k == "assistant":
                target = k
            break
        if target is None:
            return
        prev = cleaned[target]
        tcs = prev.get("tool_calls") or []
        kept = [tc for tc in tcs if tc.get("id") not in pending]
        if len(kept) == len(tcs):
            return
        new_prev = dict(prev)
        if kept:
            new_prev["tool_calls"] = kept
        else:
            new_prev.pop("tool_calls", None)
            if new_prev.get("content") in (None, ""):
                new_prev["content"] = ""
        cleaned[target] = new_prev

    for m in messages:
        role = m.get("role")
        if role == "tool":
            tid = m.get("tool_call_id")
            if tid in pending:
                cleaned.append(m)
                pending.discard(tid)
            continue
        _strip_unanswered()
        pending = set()
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                pending = {tc["id"] for tc in tcs if tc.get("id")}
        cleaned.append(m)

    _strip_unanswered()
    return cleaned


def compact_messages(messages: list, config: dict, focus: str = "") -> list:
    """Compress old messages into a summary via LLM call.

    Splits at find_split_point, summarizes old portion, returns
    [summary_msg, ack_msg, *recent_messages].

    Args:
        messages: full message list
        config: agent config dict (must contain "model")
        focus: optional focus instructions for the summarizer
    Returns:
        new compacted message list
    """
    split = find_split_point(messages)
    if split <= 0:
        return messages

    old = messages[:split]
    recent = messages[split:]

    # Build summary request
    old_text = ""
    for m in old:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str):
            old_text += f"[{role}]: {content[:500]}\n"
        elif isinstance(content, list):
            old_text += f"[{role}]: (structured content)\n"

    summary_prompt = (
        "Summarize the following conversation history concisely. "
        "Preserve key decisions, file paths, tool results, and context "
        "needed to continue the conversation."
    )
    if focus:
        summary_prompt += f"\n\nFocus especially on: {focus}"
    summary_prompt += "\n\n" + old_text

    # Call auxiliary (fast/cheap) model for summary instead of the primary model
    from auxiliary import stream_auxiliary
    summary_text = stream_auxiliary(
        system="You are a concise summarizer.",
        messages=[{"role": "user", "content": summary_prompt}],
        config=config,
    )

    summary_msg = {
        "role": "user",
        "content": f"[Previous conversation summary]\n{summary_text}",
    }
    ack_msg = {
        "role": "assistant",
        "content": "Understood. I have the context from the previous conversation. Let's continue.",
    }
    return [summary_msg, ack_msg, *recent]


# ── Main entry ────────────────────────────────────────────────────────────

def maybe_compact(state, config: dict) -> bool:
    """Check if context window is getting full and compress if needed.

    Runs snip_old_tool_results first, then auto-compact if still over threshold.

    Args:
        state: AgentState with .messages list
        config: agent config dict (must contain "model")
    Returns:
        True if compaction was performed
    """
    model = config.get("model", "")
    limit = get_context_limit(model)
    threshold = limit * 0.7

    if estimate_tokens(state.messages) <= threshold:
        return False

    # Layer 1: snip old tool results
    snip_old_tool_results(state.messages)

    if estimate_tokens(state.messages) <= threshold:
        return True

    # Layer 2: auto-compact
    state.messages = compact_messages(state.messages, config)
    state.messages.extend(_restore_plan_context(config))
    return True


# ── Plan context restoration ─────────────────────────────────────────────

def _restore_plan_context(config: dict) -> list:
    """If in plan mode, return messages that restore plan file context."""
    from pathlib import Path
    import runtime
    plan_file = runtime.get_ctx(config).plan_file or ""
    if not plan_file or config.get("permission_mode") != "plan":
        return []
    p = Path(plan_file)
    if not p.exists():
        return []
    content = p.read_text(encoding="utf-8").strip()
    if not content:
        return []
    return [
        {"role": "user", "content": f"[Plan file restored after compaction: {plan_file}]\n\n{content}"},
        {"role": "assistant", "content": "I have the plan context. Let's continue."},
    ]


# ── Manual compact ───────────────────────────────────────────────────────

def manual_compact(state, config: dict, focus: str = "") -> tuple[bool, str]:
    """User-triggered compaction via /compact. Not gated by threshold.

    Returns (success, info_message).
    """
    if len(state.messages) < 4:
        return False, "Not enough messages to compact."

    before = estimate_tokens(state.messages)
    snip_old_tool_results(state.messages)
    state.messages = compact_messages(state.messages, config, focus=focus)
    state.messages.extend(_restore_plan_context(config))
    after = estimate_tokens(state.messages)
    saved = before - after
    return True, f"Compacted: ~{before} → ~{after} tokens (~{saved} saved)"
