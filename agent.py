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
from providers import stream, AssistantTurn, TextChunk, ThinkingChunk, detect_provider, nim_next_model
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

    # Read-only dedup: when the model fires the same Read/Glob/Grep/WebFetch/
    # WebSearch call with identical args twice in this run(), short-circuit
    # the second one. We still append a synthetic tool_result to history
    # (the OpenAI/Anthropic format requires tool_calls ↔ tool_response
    # pairing) but the result is a brief reminder telling the model the
    # content is already in its context — and we suppress the UI yields
    # so the user doesn't see `⚙ Read(…)` printed twice for the same file.
    # Catches the qwen2.5 pattern where the same file gets Read in two
    # consecutive turns, then the master plan gets echoed as text twice.
    _readonly_sigs_seen: set[str] = set()
    _READ_ONLY_TOOLS = {"Read", "Glob", "Grep", "WebFetch", "WebSearch"}

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

        # NIM-only: when build.nvidia.com rate-limits a model, cycle to
        # the next free-tier model before consuming a regular retry. Capped
        # at _NIM_FALLBACK_LIMIT total swaps per turn so a fully-throttled
        # tier can't cause a busy loop.
        _nim_fallbacks_used = 0
        _NIM_FALLBACK_LIMIT = 3

        # Bounded to ONE per turn so a genuine overflow (prompt itself
        # too big) eventually surfaces instead of looping. See
        # `_try_reduce_output_cap_from_error` for the parser.
        _output_cap_reduced_this_turn = False

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
                from error_classifier import classify as _classify_err, ErrorCategory as _ErrCat
                cerr = _classify_err(e)

                # NIM 429 cascade: swap to the next free-tier model before
                # spending a retry slot. Doesn't increment `attempt` so a
                # transient global throttle still gets the regular backoff
                # path after _NIM_FALLBACK_LIMIT swaps.
                if (cerr.category == _ErrCat.RATE_LIMIT
                        and detect_provider(config.get("model", "")) == "nim"
                        and config.get("nim_auto_fallback", True)
                        and _nim_fallbacks_used < _NIM_FALLBACK_LIMIT):
                    _old = config["model"]
                    _new = nim_next_model(_old)
                    if _new and _new != _old:
                        config = {**config, "model": _new}
                        _nim_fallbacks_used += 1
                        _log.warn("nim_fallback",
                                   session_id=session_id,
                                   from_model=_old, to_model=_new,
                                   used=_nim_fallbacks_used,
                                   limit=_NIM_FALLBACK_LIMIT)
                        yield TextChunk(
                            f"\n[NIM rate-limited on {_old} — switching to "
                            f"{_new} ({_nim_fallbacks_used}/"
                            f"{_NIM_FALLBACK_LIMIT})]\n"
                        )
                        continue   # retry without consuming attempt budget

                if attempt >= max_retries or not cerr.retryable:
                    _log.error("api_failed", session_id=session_id,
                               error_type=type(e).__name__,
                               category=cerr.category.value,
                               error=_truncate_err(str(e)))
                    hint = f" Hint: {cerr.hint}" if cerr.hint else ""
                    yield TextChunk(f"\n[Failed — {type(e).__name__}: {_truncate_err(str(e))}.{hint}]\n")
                    break

                if cerr.should_compress:
                    # Before compacting, try to PARSE the error message
                    # for explicit token counts. Many providers return:
                    #   "max context 32768. requested 8192 output tokens
                    #    and your prompt contains 24577 input tokens..."
                    # When the prompt itself fits but `requested_output`
                    # pushes total over the limit, we can fix this by
                    # lowering max_tokens — no compaction needed (it
                    # wouldn't help anyway when the user's input is the
                    # huge thing, e.g. a PDF read).
                    # Bounded to ONE auto-reduction per turn so a true
                    # overflow eventually surfaces.
                    if not _output_cap_reduced_this_turn:
                        new_cap = _try_reduce_output_cap_from_error(str(e), config)
                        if new_cap and new_cap >= 256:
                            _output_cap_reduced_this_turn = True
                            old_cap = config.get("max_tokens")
                            config = {**config, "max_tokens": new_cap}
                            _log.info("output_cap_auto_reduced",
                                       session_id=session_id,
                                       from_cap=old_cap, to_cap=new_cap)
                            yield TextChunk(
                                f"\n[Context overflow — reducing output cap "
                                f"{old_cap}→{new_cap} and retrying (attempt "
                                f"{attempt+1}/{max_retries})]\n"
                            )
                            continue
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

        # Read-only dedup: walk the batch first, mark any read-only call
        # whose (name, args) signature already fired in this run() as a
        # short-circuit. The actual execution + UI yields will skip these,
        # but a synthetic tool_result still gets appended to history so the
        # OpenAI/Anthropic tool_calls ↔ tool_response pairing stays valid.
        _redundant_tcs: dict[str, str] = {}   # tool_call id → reminder text
        for tc in tool_calls:
            if tc.get("name") not in _READ_ONLY_TOOLS:
                continue
            try:
                _args_blob = _j_loop.dumps(tc.get("input", {}) or {},
                                            sort_keys=True, default=str)
            except Exception:
                continue
            _ro_sig = f"{tc['name']}:{_h_loop.md5(_args_blob.encode('utf-8','ignore')).hexdigest()}"
            if _ro_sig in _readonly_sigs_seen:
                _arg_summary = _args_blob[:120]
                _redundant_tcs[tc["id"]] = (
                    f"[deduped] You already called {tc['name']} with these "
                    f"args earlier in this turn ({_arg_summary}). The result "
                    f"is identical to your previous tool result — do not "
                    f"re-call read-only tools, use the content already in "
                    f"your context."
                )
                _log.info("readonly_dedup",
                           session_id=session_id,
                           tool=tc["name"],
                           sig=_ro_sig)
            else:
                _readonly_sigs_seen.add(_ro_sig)

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

        # Determine which tools can run in parallel — but treat redundant
        # read-only calls as "sequential" (and short-circuit during exec)
        # so the dedup path always lands on a single, predictable code path.
        from tool_registry import get_tool as _get_tool
        parallel_batch = []
        sequential_batch = []
        for tc in tool_calls:
            if not permissions[tc["id"]] or tc["id"] in _redundant_tcs:
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
            # Read-only dedup short-circuit: skip the actual execute_tool
            # call, return the synthetic reminder as the tool result. Marked
            # `permitted=True` so downstream loop-error counters don't treat
            # it as a denial.
            if tid in _redundant_tcs:
                return tc, _redundant_tcs[tid], True
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
            if tc["id"] not in _redundant_tcs:
                yield ToolStart(tc["name"], tc["input"])
                _log.debug("tool_start", session_id=session_id,
                           tool=tc["name"], input_keys=list(tc["input"].keys()))
            else:
                # Tell the user *something* happened, but tersely — don't
                # repeat the full ⚙ Read(<long path>) line.
                yield TextChunk(f"\n[deduped {tc['name']}: already in context]\n")
            tc, result, permitted = _exec_one(tc)
            _log.debug("tool_end", session_id=session_id,
                       tool=tc["name"], permitted=permitted,
                       result_len=len(result))
            results_ordered.append((tc, result, permitted))

        # Yield results and append to state in original order
        _all_errors = bool(results_ordered)
        for tc, result, permitted in results_ordered:
            # Suppress the visible ToolEnd for deduped calls — the brief
            # `[deduped ...]` line above is enough. The tool_result still
            # gets appended to state.messages so the next API request has
            # a valid tool_calls ↔ tool_response pairing.
            if tc["id"] not in _redundant_tcs:
                yield ToolEnd(tc["name"], result, permitted)
            # Auto-fanout: when a single tool result is too big to fit in the
            # active model's context window, split it across parallel sub-LLM
            # summaries instead of letting the next API call overflow.  Only
            # fires for permitted, oversize, non-error results — denials and
            # error strings are tiny and would just waste sub-calls.
            if permitted and isinstance(result, str):
                _res_low = result.lstrip()[:24].lower()
                _is_err = (_res_low.startswith("error")
                           or _res_low.startswith("denied"))
                if not _is_err:
                    try:
                        from multi_agent.fanout import (
                            should_fanout, fanout_summarize, make_llm_caller,
                            fanout_notice,
                        )
                        from compaction import get_context_limit
                        _ctx = get_context_limit(config.get("model", ""), config)
                        if should_fanout(tc["name"], result, _ctx, config):
                            # Find last user message for query focus
                            _user_q = ""
                            for _m in reversed(state.messages):
                                if _m.get("role") == "user":
                                    _c = _m.get("content", "")
                                    if isinstance(_c, str):
                                        _user_q = _c
                                    break
                            _max_sub = int(config.get("auto_fanout_max_subagents", 5) or 5)
                            yield TextChunk(
                                "\n" + fanout_notice(tc["name"], len(result),
                                                     _max_sub, _ctx) + "\n"
                            )
                            _log.info("auto_fanout_triggered",
                                       session_id=session_id,
                                       tool=tc["name"],
                                       original_chars=len(result),
                                       ctx_window=_ctx,
                                       max_subagents=_max_sub)
                            result = fanout_summarize(
                                text=result, user_question=_user_q,
                                config=config, llm_call=make_llm_caller(config),
                                ctx_window=_ctx, max_subagents=_max_sub,
                            )
                    except Exception as _fanout_err:
                        # Fanout is opportunistic — never block the tool result
                        # path on a fanout failure.  Log + fall through with
                        # the original result; downstream compaction / dynamic
                        # cap can still try.
                        _log.warn("auto_fanout_failed",
                                   session_id=session_id,
                                   tool=tc["name"],
                                   error_type=type(_fanout_err).__name__,
                                   error=_truncate_err(str(_fanout_err)))
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
    limit = get_context_limit(config.get("model", ""), config)
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


def _try_reduce_output_cap_from_error(error_str: str, config: dict) -> int | None:
    """Parse an OpenAI-style context-overflow error and compute a safe
    new max_tokens cap that fits within the model's window.

    Most providers return the message in this shape:

        "This model's maximum context length is 32768 tokens. However,
         you requested 8192 output tokens and your prompt contains at
         least 24577 input tokens, for a total of 32769 tokens..."

    From those numbers we can compute the largest output cap that fits:
        new_cap = model_max - prompt_tokens - SAFETY_BUFFER

    Returns:
        Suggested new max_tokens (>=1), or None if numbers couldn't be
        parsed or the new cap would be too small (<256) to be useful —
        in which case the caller falls back to compaction.
    """
    if not error_str:
        return None
    import re as _re_cap
    # Three numbers, in order: max-context, requested-output, prompt-tokens.
    # All providers we've seen use these patterns; tolerant on phrasing.
    m_max = _re_cap.search(
        r"(?:maximum\s+context\s+length|context\s+window|max(?:imum)?\s+tokens)\s+"
        r"(?:is\s+|of\s+)?(\d+)",
        error_str, _re_cap.IGNORECASE,
    )
    m_prompt = _re_cap.search(
        r"prompt\s+contains\s+(?:at\s+least\s+)?(\d+)",
        error_str, _re_cap.IGNORECASE,
    )
    if not (m_max and m_prompt):
        return None
    try:
        model_max = int(m_max.group(1))
        prompt_tokens = int(m_prompt.group(1))
    except ValueError:
        return None
    # Buffer must absorb provider-side prompt-token-count variance
    # between attempts. Observed in the wild on vLLM-served qwen2.5-72b:
    # the prompt grows by ~+1000 tokens between the original attempt
    # and the retry (vLLM appears to reserve decoder priming budget
    # that's not counted in the initial validation message). 2500
    # (~7.6% of 32K, ~1.25% of 200K) gives real headroom for that
    # behavior across providers we've seen.
    SAFETY_BUFFER = 2500
    new_cap = model_max - prompt_tokens - SAFETY_BUFFER
    # Don't return a cap that's even smaller than what's currently set
    # — that would be a no-op or a regression.
    current_cap = config.get("max_tokens") or 0
    if current_cap and new_cap >= current_cap:
        return None
    if new_cap < 256:
        return None
    return new_cap


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
