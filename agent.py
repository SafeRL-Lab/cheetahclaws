"""Core agent loop: neutral message format, multi-provider streaming."""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Generator

from tool_registry import get_tool_schemas
from tools import execute_tool
import tools as _tools_init  # ensure built-in tools are registered on import
from providers import stream, AssistantTurn, TextChunk, ThinkingChunk, detect_provider
from compaction import maybe_compact, estimate_tokens, get_context_limit, compact_messages, sanitize_history
import logging_utils as _log
import quota as _quota
from circuit_breaker import CircuitOpenError as _CircuitOpenError
import runtime

# ── Re-export event types (used by cheetahclaws.py) ────────────────────────
__all__ = [
    "AgentState", "run",
    "TextChunk", "ThinkingChunk",
    "ToolStart", "ToolEnd", "TurnDone", "PermissionRequest",
]


@dataclass
class AgentState:
    """Mutable session state. messages use the neutral provider-independent format."""
    messages: list = field(default_factory=list)
    total_input_tokens:  int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens:  int = 0
    total_cache_write_tokens: int = 0
    turn_count: int = 0


@dataclass
class ToolStart:
    name:   str
    inputs: dict

@dataclass
class ToolEnd:
    name:      str
    result:    str
    permitted: bool = True

@dataclass
class TurnDone:
    input_tokens:  int
    output_tokens: int

@dataclass
class PermissionRequest:
    description: str
    granted: bool = False


# ── Agent loop ─────────────────────────────────────────────────────────────

def run(
    user_message: str,
    state: AgentState,
    config: dict,
    system_prompt: str,
    depth: int = 0,
    cancel_check=None,
) -> Generator:
    """
    Multi-turn agent loop (generator).
    Yields: TextChunk | ThinkingChunk | ToolStart | ToolEnd |
            PermissionRequest | TurnDone

    Args:
        depth: sub-agent nesting depth, 0 for top-level
        cancel_check: callable returning True to abort the loop early
    """
    # Append user turn in neutral format
    user_msg = {"role": "user", "content": user_message}
    # Attach pending image from /image command if present
    sctx = runtime.get_ctx(config)
    pending_img = sctx.pending_image
    sctx.pending_image = None
    if pending_img:
        user_msg["images"] = [pending_img]
    state.messages.append(user_msg)

    # Inject runtime metadata into config so tools (e.g. Agent) can access it
    config = {**config, "_depth": depth, "_system_prompt": system_prompt}
    session_id = config.get("_session_id", "default")

    # Wire up structured logging from config (idempotent, cheap)
    _log.configure_from_config(config)

    # Loop guard: defends against models that retry failing tool calls
    # indefinitely (e.g. Gemma 4 31B looping on WebSearch+Bash whose
    # args got eaten by the native tool-call parser). Two thresholds:
    #   - same (name+args) repeated → break after `_LOOP_REPEAT_LIMIT`
    #   - any tool returning Error/Denied N consecutive times → break
    #     after `_LOOP_ERROR_LIMIT`, even across different tool names
    _loop_last_call_sig: tuple | None = None
    _loop_repeat_count = 0
    _loop_consecutive_errors = 0
    _LOOP_REPEAT_LIMIT = 3
    _LOOP_ERROR_LIMIT  = 5

    # Auto-nudge: weaker models (qwen2.5, kimi, smaller llamas, …) often
    # reply with prose like "please give me the file name" when handed an
    # absolute path that they could have explored themselves. We give them
    # exactly one transparent "try again with tools" reminder when this
    # happens. Bounded to one shot per user message to prevent any loop.
    _nudges_remaining = 1 if _looks_like_investigation(user_message) else 0

    while True:
        if cancel_check and cancel_check():
            return
        state.turn_count += 1
        assistant_turn: AssistantTurn | None = None

        # Compact context if approaching window limit
        try:
            maybe_compact(state, config)
        except Exception as _compact_err:
            _log.warn("compact_failed", error=str(_compact_err))

        # Enforce tool_calls ↔ tool-response pairing before every API call.
        # Defends against compaction artifacts, crashed tool execs, or any
        # other source of orphan 'tool' messages that OpenAI-compatible
        # providers (DeepSeek et al.) reject with a 400.
        _before_len = len(state.messages)
        state.messages = sanitize_history(state.messages)
        if len(state.messages) != _before_len:
            _log.warn("history_sanitized",
                      session_id=session_id,
                      removed=_before_len - len(state.messages))

        # ── Quota check — before spending tokens ──────────────────────────
        try:
            _quota.check_quota(session_id, config)
        except _quota.QuotaExceeded as qe:
            _log.warn("quota_exceeded", session_id=session_id, reason=qe.reason)
            yield TextChunk(f"\n[Quota exceeded — {qe.reason}]\n")
            break

        # Stream from provider — retry on ANY error (never crash the session)
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                for event in stream(
                    model=config["model"],
                    system=system_prompt,
                    messages=state.messages,
                    tool_schemas=get_tool_schemas(),
                    config=config,
                ):
                    if isinstance(event, (TextChunk, ThinkingChunk)):
                        yield event
                    elif isinstance(event, AssistantTurn):
                        assistant_turn = event
                        # Record usage for quota tracking
                        _quota.record_usage(
                            session_id, config["model"],
                            event.in_tokens, event.out_tokens,
                        )
                break  # success — exit retry loop

            except _CircuitOpenError as e:
                _log.warn("circuit_open_skip", session_id=session_id,
                          error=str(e)[:200])
                yield TextChunk(f"\n[{e}]\n")
                return  # circuit manages its own cooldown — don't retry

            except Exception as e:
                from error_classifier import classify as _classify_err
                cerr = _classify_err(e)

                if attempt >= max_retries or not cerr.retryable:
                    _log.error("api_failed", session_id=session_id,
                               error_type=type(e).__name__,
                               category=cerr.category.value,
                               error=_truncate_err(str(e)))
                    hint = f" Hint: {cerr.hint}" if cerr.hint else ""
                    yield TextChunk(f"\n[Failed — {type(e).__name__}: {_truncate_err(str(e))}.{hint}]\n")
                    break

                if cerr.should_compress:
                    _force_compact(state, config)
                    yield TextChunk(f"\n[Context too long — compacted and retrying (attempt {attempt+1}/{max_retries})]\n")
                    continue

                backoff = int(2 ** (attempt + 1) * cerr.backoff_multiplier)
                backoff = min(backoff, 30)
                _log.warn("api_retry", session_id=session_id,
                          attempt=attempt + 1, max_retries=max_retries,
                          category=cerr.category.value,
                          error_type=type(e).__name__,
                          error=_truncate_err(str(e)),
                          backoff_s=backoff)
                yield TextChunk(f"\n[Retry {attempt+1}/{max_retries} after {backoff}s — {cerr.category.value}: {_truncate_err(str(e))}]\n")
                time.sleep(backoff)

        if assistant_turn is None:
            break

        # Record assistant turn in neutral format
        _assistant_msg = {
            "role":       "assistant",
            "content":    assistant_turn.text,
            "tool_calls": assistant_turn.tool_calls,
        }
        # DeepSeek v4 requires reasoning_content to be echoed back on
        # subsequent requests when the turn contains tool_calls.  Storing it
        # on the neutral history lets messages_to_openai pass it through.
        _rc = getattr(assistant_turn, "reasoning_content", "")
        if _rc and assistant_turn.tool_calls:
            _assistant_msg["reasoning_content"] = _rc
        state.messages.append(_assistant_msg)

        state.total_input_tokens  += assistant_turn.in_tokens
        state.total_output_tokens += assistant_turn.out_tokens
        state.total_cache_read_tokens  += getattr(assistant_turn, 'cache_read_tokens', 0)
        state.total_cache_write_tokens += getattr(assistant_turn, 'cache_write_tokens', 0)
        yield TurnDone(assistant_turn.in_tokens, assistant_turn.out_tokens)

        if not assistant_turn.tool_calls:
            # Auto-nudge: text-only reply when the user clearly wanted
            # investigation (their message contained an absolute path).
            # One shot only — see `_nudges_remaining` init above.
            if _nudges_remaining > 0 and get_tool_schemas():
                _nudges_remaining -= 1
                _nudge_msg = (
                    "[system reminder] You replied with text and no tool "
                    "calls, but the user's request includes a concrete path "
                    "or file reference. Do NOT ask the user to clarify what "
                    "they already provided. Instead: list the path with Bash "
                    "`ls` (or Glob `**/*` for recursive), Read the relevant "
                    "files, then answer. Try again now."
                )
                state.messages.append({"role": "user", "content": _nudge_msg})
                _log.info("auto_nudge_text_only",
                           session_id=session_id,
                           reason="user_provided_path_but_assistant_text_only")
                continue   # retry the loop with the nudge in history
            break   # No tools → conversation turn complete

        # ── Execute tools (parallel when safe) ────────────────────────────
        tool_calls = assistant_turn.tool_calls

        # Loop guard: same tool_calls signature repeated N times → break.
        # The model is stuck retrying without progress (typically because
        # a tool result it can't parse came back, or its tool-call arg
        # parser keeps emitting the same malformed payload).
        import hashlib as _h_loop
        import json as _j_loop
        _sig = tuple(
            (tc.get("name", ""),
             _h_loop.md5(
                 _j_loop.dumps(tc.get("input", {}) or {},
                                sort_keys=True, default=str).encode(
                     "utf-8", "ignore"),
             ).hexdigest())
            for tc in tool_calls
        )
        if _sig == _loop_last_call_sig:
            _loop_repeat_count += 1
        else:
            _loop_last_call_sig = _sig
            _loop_repeat_count = 1
        if _loop_repeat_count >= _LOOP_REPEAT_LIMIT:
            _names = ", ".join(sorted({tc.get("name", "?")
                                         for tc in tool_calls}))
            _loop_msg = (
                f"\n[Loop guard] Same tool call repeated "
                f"{_LOOP_REPEAT_LIMIT} times — stopping to prevent a "
                f"runaway loop. The model kept calling {_names} with "
                f"identical args without making progress. Try /clear "
                f"and rephrase your request, or switch to a more "
                f"capable model.\n"
            )
            _log.warn("loop_guard_triggered",
                       session_id=session_id,
                       tools=_names,
                       repeats=_loop_repeat_count)
            yield TextChunk(_loop_msg)
            state.messages.append({
                "role": "assistant", "content": _loop_msg.strip(),
            })
            break

        # Check permissions first (must be sequential — may prompt user)
        permissions: dict[str, bool] = {}
        for tc in tool_calls:
            permitted = _check_permission(tc, config)
            if not permitted:
                if config.get("permission_mode") == "plan":
                    permitted = False
                else:
                    req = PermissionRequest(description=_permission_desc(tc))
                    yield req
                    permitted = req.granted
            permissions[tc["id"]] = permitted

        # Determine which tools can run in parallel
        from tool_registry import get_tool as _get_tool
        parallel_batch = []
        sequential_batch = []
        for tc in tool_calls:
            if not permissions[tc["id"]]:
                sequential_batch.append(tc)
                continue
            tdef = _get_tool(tc["name"])
            if tdef and tdef.concurrent_safe and len(tool_calls) > 1:
                parallel_batch.append(tc)
            else:
                sequential_batch.append(tc)

        def _exec_one(tc):
            """Execute a single tool call, return (tc, result, permitted)."""
            tid = tc["id"]
            permitted = permissions[tid]
            if not permitted:
                if config.get("permission_mode") == "plan":
                    plan_file = runtime.get_ctx(config).plan_file or ""
                    result = (
                        f"[Plan mode] Write operations are blocked except to the plan file: {plan_file}\n"
                        "Finish your analysis and write the plan to the plan file. "
                        "The user will run /plan done to exit plan mode and begin implementation."
                    )
                else:
                    result = "Denied: user rejected this operation"
            else:
                result = execute_tool(
                    tc["name"], tc["input"],
                    permission_mode="accept-all",
                    config=config,
                )
            return tc, result, permitted

        results_ordered = []

        # Run parallel batch concurrently
        if parallel_batch:
            from concurrent.futures import ThreadPoolExecutor
            for tc in parallel_batch:
                yield ToolStart(tc["name"], tc["input"])
            with ThreadPoolExecutor(max_workers=min(len(parallel_batch), 8)) as pool:
                futures = {pool.submit(_exec_one, tc): tc for tc in parallel_batch}
                for future in futures:
                    tc, result, permitted = future.result()
                    _log.debug("tool_end", session_id=session_id,
                               tool=tc["name"], permitted=permitted,
                               result_len=len(result))
                    results_ordered.append((tc, result, permitted))

        # Run sequential batch one by one
        for tc in sequential_batch:
            yield ToolStart(tc["name"], tc["input"])
            _log.debug("tool_start", session_id=session_id,
                       tool=tc["name"], input_keys=list(tc["input"].keys()))
            tc, result, permitted = _exec_one(tc)
            _log.debug("tool_end", session_id=session_id,
                       tool=tc["name"], permitted=permitted,
                       result_len=len(result))
            results_ordered.append((tc, result, permitted))

        # Yield results and append to state in original order
        _all_errors = bool(results_ordered)
        for tc, result, permitted in results_ordered:
            yield ToolEnd(tc["name"], result, permitted)
            state.messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "name":         tc["name"],
                "content":      result,
            })
            # Loop guard: track whether this batch was all errors.
            res_str = result if isinstance(result, str) else str(result)
            res_low = res_str.lstrip()[:24].lower()
            if not (res_low.startswith("error")
                    or res_low.startswith("denied")
                    or "keyerror" in res_low):
                _all_errors = False

        # Loop guard: cross-tool consecutive-error counter — break if
        # the model keeps invoking tools that all fail (e.g. cycling
        # between empty-args WebSearch and empty-args Bash).
        if _all_errors:
            _loop_consecutive_errors += len(results_ordered)
        else:
            _loop_consecutive_errors = 0
        if _loop_consecutive_errors >= _LOOP_ERROR_LIMIT:
            _err_msg = (
                f"\n[Loop guard] {_loop_consecutive_errors} consecutive "
                f"tool calls returned errors — stopping to prevent a "
                f"runaway loop. Likely cause: the model is emitting "
                f"tool calls without valid args (Gemma 4 + vLLM "
                f"hermes parser is a known offender). Try /clear and "
                f"rephrase, or switch to a model with native "
                f"function-calling support (claude-*, gpt-*, "
                f"deepseek-*).\n"
            )
            _log.warn("loop_guard_consecutive_errors_triggered",
                       session_id=session_id,
                       count=_loop_consecutive_errors)
            yield TextChunk(_err_msg)
            state.messages.append({
                "role": "assistant", "content": _err_msg.strip(),
            })
            break


# ── Helpers ───────────────────────────────────────────────────────────────

def _check_permission(tc: dict, config: dict) -> bool:
    """Return True if operation is auto-approved (no need to ask user)."""
    perm_mode = config.get("permission_mode", "auto")
    name = tc["name"]

    # Plan mode tools are always auto-approved
    if name in ("EnterPlanMode", "ExitPlanMode", "AskUserQuestion"):
        return True

    if perm_mode == "accept-all":
        return True
    if perm_mode == "manual":
        return False   # always ask

    if perm_mode == "plan":
        # Allow writes ONLY to the plan file
        if name in ("Write", "Edit"):
            plan_file = runtime.get_ctx(config).plan_file or ""
            target = tc["input"].get("file_path", "")
            if plan_file and target and \
               os.path.normpath(target) == os.path.normpath(plan_file):
                return True
            return False
        if name == "NotebookEdit":
            return False
        if name == "Bash":
            from tools import _is_safe_bash
            return _is_safe_bash(tc["input"].get("command", ""))
        return True  # reads are fine

    # "auto" mode: only ask for writes and non-safe bash
    if name in ("Read", "Glob", "Grep", "WebFetch", "WebSearch"):
        return True
    if name == "Bash":
        from tools import _is_safe_bash
        return _is_safe_bash(tc["input"].get("command", ""))
    return False   # Write, Edit → ask


def _permission_desc(tc: dict) -> str:
    name = tc["name"]
    inp  = tc["input"]
    if name == "Bash":   return f"Run: {inp.get('command', '')}"
    if name == "Write":  return f"Write to: {inp.get('file_path', '')}"
    if name == "Edit":   return f"Edit: {inp.get('file_path', '')}"
    return f"{name}({list(inp.values())[:1]})"


def _force_compact(state: AgentState, config: dict) -> bool:
    """Force compaction regardless of threshold. Used when API rejects for context too long."""
    limit = get_context_limit(config.get("model", ""))
    before = estimate_tokens(state.messages)
    if before <= 0:
        return False
    from compaction import snip_old_tool_results
    snip_old_tool_results(state.messages, max_chars=1000, preserve_last_n_turns=3)
    if estimate_tokens(state.messages) < limit * 0.9:
        return True
    state.messages = compact_messages(state.messages, config)
    from compaction import _restore_plan_context
    state.messages.extend(_restore_plan_context(config))
    after = estimate_tokens(state.messages)
    return after < before


def _truncate_err(s: str, max_len: int = 120) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."


# Matches an absolute-path-like token: starts with '/', has at least two
# segments, segment chars are word/dot/dash. Rejects bare '/' or '//'.
# The leading lookbehind keeps URL paths (https://host/...) out of the match.
import re as _re_invest
_PATH_RE = _re_invest.compile(
    r"(?:(?<=^)|(?<=[\s,;:'\"`(<\[]))/[A-Za-z0-9_.][\w./-]*/[\w.][\w./-]*"
)


def _looks_like_investigation(text: str) -> bool:
    """Heuristic: does the user message hand the agent a path/file to look at?

    Only the highest-precision signal is used — an absolute path token —
    because the auto-nudge that consumes this signal must not fire on
    benign greetings. URLs are stripped first so 'https://x/y' does not
    count as a filesystem path.
    """
    if not text:
        return False
    # Strip URLs so http(s)://host/path doesn't masquerade as a fs path.
    no_urls = _re_invest.sub(r"https?://\S+", " ", text)
    return bool(_PATH_RE.search(no_urls))
