"""proactive_scheduler.py — daemon-owned proactive watcher (RFC 0002 F-5).

Runs a single background thread inside the daemon. The thread wakes
every ``TICK_INTERVAL_S`` (1 s by default — same cadence as the
historical in-REPL ``_proactive_watcher_loop``), reads the persisted
``proactive_state``, and:

  * if disabled → no-op.
  * if enabled and idle (``now - last_tick_at >= interval_s``) → publish
    a ``proactive_tick`` event on the EventBus and reset
    ``last_tick_at`` so the counter restarts.

Mirrors :mod:`monitor.scheduler` (F-3): module-level singleton thread,
interruptible ``Event.wait`` so daemon shutdown doesn't have to stall a
full tick, ``owned_by_daemon`` global so REPL callers can step aside
without the daemon's own scheduler thinking *itself* is the foreign one.

The event payload is intentionally minimal:

    {
        "interval_s":  300,
        "last_tick_at": 1715520012.345,
        "fired_at":     1715520312.789,
    }

Subscribers (REPL, bridges, future agents) decide what to do with it —
typically inject a "review previous messages" prompt. The scheduler
itself doesn't reach into agent / bridge state; that coupling lives in
the consumer, where it belongs.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from . import proactive_state


TICK_INTERVAL_S = 1.0

_scheduler_thread: Optional[threading.Thread] = None
_scheduler_stop = threading.Event()
_owned_by_daemon: bool = False


def _foreign_daemon_running() -> bool:
    """True when a daemon other than this process is registered as owner.

    Used by the REPL-side caller (when this module is imported from a
    REPL that started its own scheduler) to step aside. The daemon's
    own scheduler thread sets ``_owned_by_daemon=True`` so the same code
    path doesn't make the daemon defer to itself.
    """
    if _owned_by_daemon:
        return False
    try:
        import os
        from . import discovery
        info = discovery.locate()
        if info is None:
            return False
        peer_pid = info.get("pid")
        return isinstance(peer_pid, int) and peer_pid != os.getpid()
    except Exception:
        return False


def _scheduler_loop() -> None:
    """Body of the scheduler thread. One wake-up per ``TICK_INTERVAL_S``;
    publishes ``proactive_tick`` when idle threshold is crossed."""
    while True:
        if _scheduler_stop.wait(timeout=TICK_INTERVAL_S):
            return
        if _foreign_daemon_running():
            # A REPL-side scheduler accidentally inherited a foreign
            # discovery file — back off rather than double-fire.
            continue
        try:
            enabled, interval_s, last_at = proactive_state.get_state()
        except Exception:
            # DB read failure: skip this tick rather than crash the
            # thread. Next tick will retry.
            continue
        if not enabled:
            continue
        now = time.time()
        # last_at == 0 means "just enabled or just reset" — set_state()
        # writes a fresh tick on every enable so this normally only
        # matters for tests / very first run after migration.
        if last_at <= 0:
            try:
                proactive_state.record_tick(now)
            except Exception:
                pass
            continue
        if now - last_at < interval_s:
            continue
        # Idle threshold crossed — fire the event and reset the counter
        # using a single ``now`` reading so the persisted row and the
        # event agree on the clock.
        try:
            proactive_state.record_tick(now)
        except Exception:
            pass
        _publish_tick(interval_s=interval_s, last_tick_at=last_at,
                      fired_at=now)


def _publish_tick(*, interval_s: int, last_tick_at: float,
                  fired_at: float) -> None:
    """Best-effort SSE publish — never raises, so a broken bus can't
    take the scheduler down."""
    try:
        from . import events
        events.get_bus().publish(
            "proactive_tick",
            {
                "interval_s":   int(interval_s),
                "last_tick_at": float(last_tick_at),
                "fired_at":     float(fired_at),
            },
        )
    except Exception:
        pass


def start(*, owned_by_daemon: bool = False) -> bool:
    """Start the background scheduler. Returns False if already running.

    ``owned_by_daemon`` is the equivalent of F-3's flag: the daemon
    process sets it so the loop's own ``_foreign_daemon_running()``
    check doesn't fire false positives against its own discovery file.
    """
    global _scheduler_thread, _owned_by_daemon
    if _scheduler_thread and _scheduler_thread.is_alive():
        return False
    _owned_by_daemon = bool(owned_by_daemon)
    _scheduler_stop.clear()
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        daemon=True,
        name="proactive-scheduler",
    )
    _scheduler_thread.start()
    return True


def stop() -> bool:
    """Signal the scheduler thread to exit and join with a short cap.
    Safe to call when not running (returns False)."""
    global _scheduler_thread
    if not _scheduler_thread or not _scheduler_thread.is_alive():
        return False
    _scheduler_stop.set()
    _scheduler_thread.join(timeout=5)
    _scheduler_thread = None
    return True


def is_running() -> bool:
    return bool(_scheduler_thread and _scheduler_thread.is_alive())


__all__ = [
    "TICK_INTERVAL_S",
    "start",
    "stop",
    "is_running",
]
