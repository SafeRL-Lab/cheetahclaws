"""
monitor/scheduler.py — Background scheduler for subscription monitoring.

Runs subscriptions on their configured schedule in a daemon thread.
Each subscription is checked against its schedule; if due, it fetches,
summarizes, and delivers via configured channels.

Schedule values:
  "30m"    — every 30 minutes
  "1h"     — every hour
  "6h"     — every 6 hours
  "12h"    — every 12 hours
  "daily"  — once per day (24h)
  "weekly" — once per week
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Callable

from monitor.store import (
    list_subscriptions, update_last_run, save_report,
)
from monitor.fetchers import fetch
from monitor.summarizer import summarize
from monitor.notifier import deliver, auto_channels

_scheduler_thread: threading.Thread | None = None
_scheduler_stop = threading.Event()
_current_config: dict = {}

# Maps schedule strings to seconds
_SCHEDULE_SECONDS = {
    "15m":     15 * 60,
    "30m":     30 * 60,
    "1h":      60 * 60,
    "2h":      2  * 60 * 60,
    "6h":      6  * 60 * 60,
    "12h":     12 * 60 * 60,
    "daily":   24 * 60 * 60,
    "weekly":  7  * 24 * 60 * 60,
}


def _parse_schedule(s: str) -> int:
    """Convert schedule string to seconds. Default 6h."""
    s = (s or "6h").lower().strip()
    if s in _SCHEDULE_SECONDS:
        return _SCHEDULE_SECONDS[s]
    # Parse "Nh" / "Nm" patterns
    if s.endswith("h"):
        try:
            return int(s[:-1]) * 3600
        except ValueError:
            pass
    if s.endswith("m"):
        try:
            return int(s[:-1]) * 60
        except ValueError:
            pass
    return _SCHEDULE_SECONDS["6h"]


def _is_due(sub: dict) -> bool:
    """Return True if this subscription should run now."""
    last_run = sub.get("last_run")
    if not last_run:
        return True
    try:
        last_dt = datetime.fromisoformat(last_run)
    except Exception:
        return True
    interval = _parse_schedule(sub.get("schedule", "6h"))
    return (datetime.now() - last_dt).total_seconds() >= interval


def run_one(topic: str, config: dict, force: bool = False) -> str:
    """Fetch + summarize + deliver one subscription. Returns the report."""
    subs = {s["topic"]: s for s in list_subscriptions()}
    sub = subs.get(topic)
    if not sub and not force:
        return f"No subscription found for topic: {topic}"

    raw = fetch(topic)
    report = summarize(raw, config)

    channels = []
    if sub:
        channels = sub.get("channels") or []
    if not channels:
        channels = auto_channels(config)
    # Always at least console
    if not channels:
        channels = ["console"]

    results = deliver(report, channels, config)
    failed = [f"{ch}: {e}" for ch, e in results.items() if e]
    if failed:
        report += "\n\n[Delivery errors: " + "; ".join(failed) + "]"

    update_last_run(topic, report)
    # F-3: persist the full report into monitor_reports + emit a
    # monitor_report event so SSE clients (Web UI / future bridges) see
    # the new digest as it lands.  Both calls are best-effort — a
    # failure in either path must not lose the in-process return value
    # that REPL `/monitor run` is showing the user right now.
    sent_to = [ch for ch, err in results.items() if not err]
    report_id = ""
    try:
        report_id = save_report(topic, report, sent_to=sent_to)
    except Exception:
        pass
    try:
        from cc_daemon import events as _events
        _events.get_bus().publish(
            "monitor_report",
            {
                "topic":     topic,
                "report_id": report_id,
                "body":      report,
                "sent_to":   sent_to,
                "errors":    failed,
            },
        )
    except Exception:
        # If the daemon isn't running (REPL-only mode), no bus is set up
        # — we already saved the report to monitor_reports above so the
        # next daemon-bound subscriber will see it via list_reports.
        pass
    return report


def _scheduler_loop(config: dict, on_report: Callable | None) -> None:
    """Background loop: check every minute, run due subscriptions."""
    while not _scheduler_stop.is_set():
        try:
            for sub in list_subscriptions():
                if _scheduler_stop.is_set():
                    break
                if _is_due(sub):
                    report = run_one(sub["topic"], config)
                    if on_report:
                        on_report(sub["topic"], report)
        except Exception:
            pass
        # Interruptible 60 s wait — Event.wait returns immediately when
        # _scheduler_stop is set, so daemon shutdown doesn't have to
        # stall up to 30 s for the scheduler thread to wake up.
        if _scheduler_stop.wait(timeout=60):
            return


def start(config: dict, on_report: Callable | None = None) -> bool:
    """Start background scheduler. Returns False if already running."""
    global _scheduler_thread, _current_config
    if _scheduler_thread and _scheduler_thread.is_alive():
        return False
    _current_config = config
    _scheduler_stop.clear()
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        args=(config, on_report),
        daemon=True,
        name="monitor-scheduler",
    )
    _scheduler_thread.start()
    return True


def stop() -> bool:
    """Stop background scheduler. Returns False if not running."""
    global _scheduler_thread
    if not _scheduler_thread or not _scheduler_thread.is_alive():
        return False
    _scheduler_stop.set()
    _scheduler_thread.join(timeout=5)
    return True


def is_running() -> bool:
    return bool(_scheduler_thread and _scheduler_thread.is_alive())
