"""
logging_utils.py — Structured JSON logging for CheetahClaws.

Writes newline-delimited JSON records to stderr (default) or a log file.
Thread-safe. Zero external dependencies.

Each log record:
  {"ts": "...", "level": "info", "event": "api_call_done",
   "session_id": "abc123", "provider": "anthropic", ...}

Config keys (read by configure_from_config):
  log_level : "off" | "error" | "warn" | "info" | "debug"  (default "warn")
  log_file  : absolute path string or null/empty → stderr
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

# Level map: higher number = more verbose
_LEVELS = {"off": 0, "error": 1, "warn": 2, "info": 3, "debug": 4}
_DEFAULT_LEVEL = "warn"

_lock              = threading.Lock()
_level: int        = _LEVELS[_DEFAULT_LEVEL]
_log_fh            = None        # open file handle, or None → stderr
_log_file_path: str | None = None
# Fast-path: track last configured values to skip redundant configure() calls
_cfg_key: tuple    = (_DEFAULT_LEVEL, None)


def configure(log_level: str = "warn", log_file: str | None = None) -> None:
    """Configure log level and output file. Thread-safe. Idempotent."""
    global _level, _log_fh, _log_file_path, _cfg_key

    lv  = (log_level  or "warn").lower()
    lf  = log_file or None
    key = (lv, lf)

    with _lock:
        if key == _cfg_key:
            return  # nothing changed — fast path

        _level = _LEVELS.get(lv, _LEVELS["warn"])

        # Re-open file only if path changed
        if lf != _log_file_path:
            if _log_fh is not None:
                try:
                    _log_fh.close()
                except Exception:
                    pass
                _log_fh = None
            if lf:
                try:
                    _log_fh = open(lf, "a", encoding="utf-8", buffering=1)
                except Exception:
                    _log_fh = None  # fallback to stderr silently
            _log_file_path = lf

        _cfg_key = key


def configure_from_config(config: dict) -> None:
    """Convenience wrapper: read log_level / log_file from a config dict."""
    configure(
        log_level=config.get("log_level", "warn"),
        log_file=config.get("log_file") or None,
    )


def _emit(level_name: str, event: str, **fields: Any) -> None:
    level_num = _LEVELS.get(level_name, _LEVELS["info"])
    if level_num > _level:
        return
    record: dict[str, Any] = {
        "ts":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": level_name,
        "event": event,
    }
    record.update(fields)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _lock:
        dest = _log_fh
        if dest is None:
            # In web terminal mode, suppress stderr JSON logs to avoid
            # polluting the terminal output shown in the browser.
            if _is_web_terminal:
                return
            dest = sys.stderr
        try:
            dest.write(line + "\n")
            dest.flush()
        except Exception:
            pass


# Detect web terminal mode: set by web/server.py via env var
_is_web_terminal = os.environ.get("CHEETAHCLAWS_WEB_TERMINAL") == "1"


# ── Public helpers ─────────────────────────────────────────────────────────

def error(event: str, **fields: Any) -> None:
    _emit("error", event, **fields)

def warn(event: str, **fields: Any) -> None:
    _emit("warn", event, **fields)

def info(event: str, **fields: Any) -> None:
    _emit("info", event, **fields)

def debug(event: str, **fields: Any) -> None:
    _emit("debug", event, **fields)
