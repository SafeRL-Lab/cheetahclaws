"""Bridge modules for external messaging platforms."""
from __future__ import annotations

import os


def resolve_bridge_token(env_var: str, config_key: str, repl_arg: str,
                         config: dict) -> tuple[str, str]:
    """Resolve a bridge bot token with this precedence:

      1. Environment variable (recommended — never enters readline history).
      2. REPL argument (deprecated — surfaces token in readline history).
      3. Persisted config (config.json on disk).

    Returns (token, source) where source is "env", "repl", "config", or "none".
    """
    env_tok = (os.environ.get(env_var) or "").strip()
    if env_tok:
        return env_tok, "env"
    if repl_arg:
        return repl_arg, "repl"
    cfg_tok = (config.get(config_key) or "").strip() if config else ""
    if cfg_tok:
        return cfg_tok, "config"
    return "", "none"


def scrub_token_from_history(token: str) -> None:
    """Remove any readline history entries that contain the given token.

    Bot tokens land in readline history when a user types
    ``/telegram <token> <chat_id>`` at the REPL prompt. Once we know the
    actual token value, walk the history backwards and delete every
    entry that embeds it.
    """
    if not token or len(token) < 8:
        return
    try:
        import readline
    except ImportError:
        return
    try:
        n = readline.get_current_history_length()
    except Exception:
        return
    for idx in range(n, 0, -1):
        try:
            line = readline.get_history_item(idx)
        except Exception:
            continue
        if line and token in line:
            try:
                readline.remove_history_item(idx - 1)
            except Exception:
                pass
