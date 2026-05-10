"""Configuration management for CheetahClaws (multi-provider)."""
import os
import json
from pathlib import Path

CONFIG_DIR        = Path.home() / ".cheetahclaws"
CONFIG_FILE       = CONFIG_DIR  / "config.json"
HISTORY_FILE      = CONFIG_DIR  / "input_history.txt"
SESSIONS_DIR      = CONFIG_DIR  / "sessions"
DAILY_DIR         = SESSIONS_DIR / "daily"       # daily/YYYY-MM-DD/session_*.json
SESSION_HIST_FILE = SESSIONS_DIR / "history.json" # master: all sessions ever

# kept for backward-compat (/resume still reads from here)
MR_SESSION_DIR = SESSIONS_DIR / "mr_sessions"

DEFAULTS = {
    "model":            "ollama/gemma4:e4b",
    "max_tokens":       40000,
    "permission_mode":  "auto",   # auto | accept-all | manual
    "verbose":          False,
    # Tri-state: None = unset (use provider default), True = ON, False = explicit OFF.
    # The explicit-OFF state matters for DeepSeek v4 where the server default
    # is ON; providers.py only injects the disable toggle when value is False.
    "thinking":         None,
    "thinking_budget":  10000,
    "custom_base_url":  "",       # for "custom" provider
    "max_tool_output":  32000,
    "max_agent_depth":  3,
    "max_concurrent_agents": 3,
    "session_daily_limit":   10000,    # max sessions kept per day in daily/
    "session_history_limit": 100000,  # max sessions kept in history.json
    # ── Security settings ──────────────────────────────────────────────────
    # allowed_root: restrict file operations (Read/Write/Edit/Glob/Grep) to this
    # directory tree.  null = unrestricted (CLI default).  Set to the project
    # root in production deployments to prevent path traversal.
    "allowed_root": None,
    # shell_policy: controls Bash tool execution.
    #   "allow"   — execute freely (CLI default)
    #   "log"     — execute but write every command to stderr with session_id
    #   "deny"    — block all Bash execution
    "shell_policy": "allow",
    # ── Structured logging ─────────────────────────────────────────────────
    # log_level: "off" | "error" | "warn" | "info" | "debug"
    #   Default "warn" keeps the interactive CLI quiet; set to "info" on
    #   production servers to capture every API call, retry, and quota event.
    "log_level": "warn",
    # log_file: absolute path or null.  null → stderr (only warn/error visible
    #   at default level).  Point to a file in production for persistent logs.
    "log_file": None,
    # ── Circuit breaker ────────────────────────────────────────────────────
    # circuit_failure_threshold: consecutive failures (in window) to trip open.
    "circuit_failure_threshold": 5,
    # circuit_window_seconds: rolling window for failure counting.
    "circuit_window_seconds": 60,
    # circuit_cooldown_seconds: how long to stay OPEN before probing again.
    "circuit_cooldown_seconds": 120,
    # ── Quota / budget control ─────────────────────────────────────────────
    # All limits are null (unlimited) by default.  Set to enforce hard caps.
    "session_token_budget": None,  # max tokens (in+out) per session
    "session_cost_budget":  None,  # max USD per session
    "daily_token_budget":   None,  # max tokens today (all sessions)
    "daily_cost_budget":    None,  # max USD today (all sessions)
    # ── Auto-fanout for oversize tool outputs ─────────────────────────────
    # When a single tool output (e.g. Read on a 6.6 MB PDF) is bigger than
    # auto_fanout_threshold * model_context_window, instead of letting the
    # next API call overflow, split it into chunks and dispatch parallel
    # sub-LLM calls to summarize each chunk, then merge. Critical for small
    # context windows (32k local models) reading large source material.
    "auto_fanout_enabled":                  True,
    "auto_fanout_threshold":                0.4,   # fraction of ctx_window
    "auto_fanout_max_subagents":            5,
    "auto_fanout_chunk_overlap_tokens":     200,
    # ── Autonomous-agent stagnation detection ─────────────────────────────
    # Stop the iteration loop in agent_runner if the model emits the same
    # summary text N iterations in a row (e.g. "task complete" repeated).
    # 0 disables. 3 catches the common "model declares done; template
    # politely asks again" case without false-positives on slowly-progressing
    # multi-day work where consecutive iterations may produce similar status
    # updates.
    "auto_agent_dup_summary_limit":         3,
    # Per-provider API keys (optional; env vars take priority)
    # "anthropic_api_key": "sk-ant-..."
    # "openai_api_key":    "sk-..."
    # "gemini_api_key":    "..."
    # "kimi_api_key":      "..."
    # "qwen_api_key":      "..."
    # "zhipu_api_key":     "..."
    # "deepseek_api_key":  "..."
    # ── WeChat smart-reply (off by default) ────────────────────────────────
    # When enabled, inbound messages from whitelisted contacts no longer
    # auto-reply via the agent.  Instead the auxiliary cheap model drafts
    # 3 candidate replies and pushes them to the user's `filehelper`
    # (文件传输助手) chat for approval.  See bridges/wechat_smart_reply.py.
    "wechat_smart_reply":                  False,
    "wechat_smart_reply_whitelist":        [],     # list of from_user_id strings
    "wechat_smart_reply_groups":           False,  # also draft for group messages
    "wechat_smart_reply_groups_at_only":   False,  # in groups, only when @<self>
    "wechat_smart_reply_timeout_s":        300,    # panel expiry seconds
    # WeChat self nickname — needed for groups_at_only matching.  Not set
    # automatically; user provides via config or `/wechat self <nickname>`.
    "wechat_self_nickname":                "",
    # WeChat self uid — bridge inbound from this uid is YOUR OWN message
    # to the bot. smart-reply ignores it unconditionally so your normal
    # messages still reach the agent, even if you accidentally put your
    # own uid in wechat_smart_reply_whitelist (which is intended for OTHER
    # contacts whose messages you want the bot to draft replies for).
    # Auto-recorded by the wechat bridge on first inbound (see _wx_poll_loop).
    "wechat_self_uid":                     "",
}


def load_config() -> dict:
    CONFIG_DIR.mkdir(exist_ok=True)
    SESSIONS_DIR.mkdir(exist_ok=True)
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text()))
        except Exception:
            pass
    # Backward-compat: legacy single api_key → anthropic_api_key
    if cfg.get("api_key") and not cfg.get("anthropic_api_key"):
        cfg["anthropic_api_key"] = cfg.pop("api_key")
    # Also accept ANTHROPIC_API_KEY env for backward-compat
    if not cfg.get("anthropic_api_key"):
        cfg["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
    # Env var always wins over persisted value so .env changes take effect immediately
    if os.environ.get("ANTHROPIC_ENDPOINT"):
        cfg["anthropic_endpoint"] = os.environ["ANTHROPIC_ENDPOINT"]
    elif not cfg.get("anthropic_endpoint"):
        cfg["anthropic_endpoint"] = "https://api.anthropic.com"
    return cfg


def save_config(cfg: dict):
    CONFIG_DIR.mkdir(exist_ok=True)
    # Strip internal runtime keys (e.g. _run_query_callback) before saving
    data = {k: v for k, v in cfg.items() if not k.startswith("_")}
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


def current_provider(cfg: dict) -> str:
    from providers import detect_provider
    return detect_provider(cfg.get("model", "claude-opus-4-6"))


def has_api_key(cfg: dict) -> bool:
    """Check whether the active provider has an API key configured."""
    from providers import get_api_key
    pname = current_provider(cfg)
    key = get_api_key(pname, cfg)
    return bool(key)


def calc_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    from providers import calc_cost as _cc
    return _cc(model, in_tokens, out_tokens)
