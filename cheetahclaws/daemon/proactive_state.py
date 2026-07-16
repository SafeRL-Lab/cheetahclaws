"""proactive_state.py — durable proactive-watcher settings (RFC 0002 F-5).

Stores the three pieces of proactive-watcher state in ``schema_meta`` so
the daemon-side scheduler can survive REPL exits and daemon restarts:

  ``proactive.enabled``       — "0" or "1"
  ``proactive.interval_s``    — idle threshold in seconds (integer string)
  ``proactive.last_tick_at``  — UNIX timestamp the scheduler last fired

A separate table would have been overkill — there's exactly one
session-wide setting, no fan-out. ``schema_meta`` already exists from
F-2 with the right shape (KV string), and the read/write traffic here
is tiny.

The module exposes a small, typed surface:

  :func:`get_state`     — returns ``(enabled: bool, interval_s: int, last_tick_at: float)``
  :func:`set_state`     — writes ``enabled`` and ``interval_s`` together
  :func:`tickle`        — bumps ``last_tick_at`` to ``time.time()``
  :func:`record_tick`   — same as ``tickle`` but takes an explicit timestamp,
                          used by the scheduler so the published event and
                          the persisted row share a clock reading.

All writes commit immediately; reads do a single SELECT per call. The
volume is one round-trip per scheduler tick (configurably every 1 s in
F-5, so ~3600 reads/hour — negligible for SQLite).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Tuple

from . import schema


# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_INTERVAL_S = 300  # 5 min — matches the historical REPL value.

_KEY_ENABLED      = "proactive.enabled"
_KEY_INTERVAL_S   = "proactive.interval_s"
_KEY_LAST_TICK_AT = "proactive.last_tick_at"


# ── Helpers ────────────────────────────────────────────────────────────────


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(conn, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row else None


def _write(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO schema_meta (key, value, updated_at) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
        "updated_at=excluded.updated_at",
        (key, value, _iso_now()),
    )


# ── Public API ─────────────────────────────────────────────────────────────


def get_state() -> Tuple[bool, int, float]:
    """Return ``(enabled, interval_s, last_tick_at)``.

    Missing rows return safe defaults (``False`` / ``DEFAULT_INTERVAL_S``
    / ``0.0``) — first-time callers don't have to special-case "table
    populated yet?".
    """
    conn = schema.get_conn()
    enabled  = _read(conn, _KEY_ENABLED) or "0"
    interval = _read(conn, _KEY_INTERVAL_S) or str(DEFAULT_INTERVAL_S)
    last_at  = _read(conn, _KEY_LAST_TICK_AT) or "0"
    try:
        return (
            enabled == "1",
            max(1, int(interval)),
            float(last_at),
        )
    except (TypeError, ValueError):
        # Corrupt row — fall back to defaults rather than crashing the
        # scheduler thread. A warn-level log could go here but the
        # daemon's structured logger isn't always reachable; keep silent
        # and let the next set_state() heal the bad row.
        return (False, DEFAULT_INTERVAL_S, 0.0)


def set_state(*, enabled: bool, interval_s: int) -> None:
    """Persist ``enabled`` + ``interval_s`` atomically.

    Resets ``last_tick_at`` to "now" so the user gets the full configured
    interval of grace before the next tick, regardless of how stale the
    prior reading was. The REPL's pre-F-5 watcher does the same.

    Raises:
        ValueError when ``interval_s`` is < 1 (a 0-second interval would
        spin the scheduler at its tick rate and is almost certainly a
        bug in the caller).
    """
    if interval_s < 1:
        raise ValueError(f"interval_s must be >= 1, got {interval_s}")
    conn = schema.get_conn()
    _write(conn, _KEY_ENABLED, "1" if enabled else "0")
    _write(conn, _KEY_INTERVAL_S, str(int(interval_s)))
    _write(conn, _KEY_LAST_TICK_AT, repr(time.time()))
    conn.commit()


def disable() -> None:
    """Flip enabled off without touching the interval — the next
    ``proactive.set`` round-trip keeps the user's previous cadence."""
    conn = schema.get_conn()
    _write(conn, _KEY_ENABLED, "0")
    conn.commit()


def tickle() -> None:
    """Mark "now" as the most recent user interaction so the watcher's
    idle counter restarts. Used by REPL and bridges when they receive
    input — keeps the watcher from firing during active conversations."""
    record_tick(time.time())


def record_tick(ts: float) -> None:
    """Persist a specific tick timestamp. Used by the scheduler so the
    SSE event and the SQLite row share the same clock reading."""
    conn = schema.get_conn()
    _write(conn, _KEY_LAST_TICK_AT, repr(ts))
    conn.commit()


__all__ = [
    "DEFAULT_INTERVAL_S",
    "get_state",
    "set_state",
    "disable",
    "tickle",
    "record_tick",
]
