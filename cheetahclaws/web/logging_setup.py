"""Structured JSON logging for the web UI.

Loggers used elsewhere in the package:
    web.server   — connection/request lifecycle, access logs
    web.api      — chat session operations
    web.db       — DB init, migration, errors
    web.auth     — login/register/logout (never the password itself)

Configure at process start by calling `setup_logging()`. Idempotent.

Output goes to stderr by default so it doesn't get tangled with the program's
stdout (which the chat UI/agent treats as content). One JSON record per line —
trivial to ship into Loki / CloudWatch / journald.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import threading
from typing import Any

_LEVEL_ENV = "CHEETAHCLAWS_LOG_LEVEL"

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record. Extra kwargs become top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Promote anything passed via `extra={...}` to top-level keys
        for k, v in record.__dict__.items():
            if k in _RESERVED or k.startswith("_"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_setup_lock = threading.Lock()
_setup_done = False


def setup_logging(level: str | None = None) -> None:
    """Install the JSON formatter on the `web.*` namespace. Idempotent."""
    global _setup_done
    with _setup_lock:
        if _setup_done:
            return
        lvl_str = (level or os.environ.get(_LEVEL_ENV, "INFO")).upper()
        lvl = getattr(logging, lvl_str, logging.INFO)
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(JsonFormatter())
        # Apply to the `web` parent logger so all child loggers inherit it.
        root_web = logging.getLogger("web")
        root_web.setLevel(lvl)
        # Don't propagate to the root logger (which may have other handlers
        # set by the host application — would double-log).
        root_web.propagate = False
        # Replace any existing handlers (idempotency guard for re-setup)
        for h in list(root_web.handlers):
            root_web.removeHandler(h)
        root_web.addHandler(handler)
        _setup_done = True


def get_logger(name: str) -> logging.Logger:
    """Get a `web.<name>` logger, ensuring setup has run."""
    setup_logging()
    return logging.getLogger(f"web.{name}")


# ── Lightweight in-process metrics (no Prometheus client dependency) ──────
# Counters here back the /metrics endpoint and the request access log.

_metrics_lock = threading.Lock()
_counters: dict[str, int] = {
    "requests_total": 0,
    "requests_4xx": 0,
    "requests_5xx": 0,
    "ws_connections_total": 0,
    "auth_logins_total": 0,
    "auth_logins_failed": 0,
    "auth_registrations_total": 0,
}
_started_at = time.time()


def incr(name: str, by: int = 1) -> None:
    with _metrics_lock:
        _counters[name] = _counters.get(name, 0) + by


def snapshot() -> dict[str, int]:
    with _metrics_lock:
        return dict(_counters)


def uptime_seconds() -> float:
    return time.time() - _started_at
