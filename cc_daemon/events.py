"""events.py — SQLite-backed event log + SSE pub/sub for the daemon.

F-2 swap of F-1's in-memory ring for the ``daemon_events`` SQLite table.
The wire surface (``EventBus.publish`` / ``replay_since`` / ``subscribe`` /
``unsubscribe`` / ``subscriber_count`` / ``format_sse`` / ``heartbeat_frame``
/ ``get_bus`` / ``reset_bus_for_tests``) is unchanged so the spike's tests
(``tests/test_daemon_spike.py``) keep passing without edits.

Behaviour deltas vs F-1:

* ``publish`` writes a row to ``daemon_events`` (id is the SQLite
  ``AUTOINCREMENT`` rowid → strictly monotonic across daemon restarts and
  across pruning).  Subscribers still get the in-process fanout for live
  tail.
* ``replay_since`` reads from SQLite, so a client that disconnects and
  reconnects after the daemon restarts still sees what it missed (subject
  to retention).
* Retention enforces both an age cap (default 24 h) and a row cap (default
  100 K) — chauncygu's #74 review §7 default.  Pruning runs opportunistically
  every N publishes so steady-state load stays inside one INSERT + one
  COMMIT per event.
* Gap detection: ``replay_since(N)`` yields a synthetic ``gap`` event when
  ``N + 1`` is older than ``MIN(id)`` in the table (i.e. retention has
  evicted what the client wanted), so SSE clients can resync.

Concurrency: each thread that hits the bus gets its own SQLite connection
through :func:`cc_daemon.schema.get_conn` (mirrors ``session_store``'s
pattern); WAL + 5 s busy timeout is set there.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

# Re-exported for spike compatibility — server.py and tests import these.
RING_CAP = 1000
HEARTBEAT_INTERVAL_S = 15.0

# Default retention policy (chauncygu #74 review §7).
DEFAULT_RETENTION_HOURS = 24
DEFAULT_RETENTION_ROWS = 100_000

# How often (in publishes) we run the prune sweep.  Cheap heuristic that
# avoids running an O(N) DELETE on every publish.
PRUNE_EVERY_N_PUBLISHES = 100

# Per-subscriber queue depth.  Same as the spike default.
SUBSCRIBER_QUEUE_DEPTH = 4096


# ── Helpers ────────────────────────────────────────────────────────────────

def _epoch_to_iso(ts: float) -> str:
    """ISO 8601 with microsecond precision, ``Z`` suffix.

    Microseconds matter for retention: at high publish rates two events
    can otherwise end up with identical second-precision timestamps and
    the time-based prune would treat them as a single bucket.  String
    comparison still sorts correctly because all timestamps share the
    same width.
    """
    return (datetime.fromtimestamp(ts, tz=timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S.%fZ"))


def _iso_to_epoch(ts: str) -> float:
    """Best-effort ISO 8601 (with trailing ``Z``) → unix timestamp."""
    if not ts:
        return time.time()
    try:
        cleaned = ts.rstrip("Z")
        dt = datetime.fromisoformat(cleaned).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return time.time()


# ── EventBus ──────────────────────────────────────────────────────────────

class EventBus:
    """Append-and-fanout bus backed by ``daemon_events``.

    *ring_cap* is accepted for backward compat with the spike constructor but
    no longer drives capacity — retention is governed by
    ``retention_hours`` / ``retention_rows`` instead.
    """

    def __init__(self, ring_cap: int = RING_CAP, *,
                 retention_hours: float = DEFAULT_RETENTION_HOURS,
                 retention_rows: int = DEFAULT_RETENTION_ROWS,
                 prune_every_n: int = PRUNE_EVERY_N_PUBLISHES) -> None:
        # ring_cap is preserved on the instance for any caller that
        # introspects it (none in the tree today, but the spike publicised
        # the kw).
        self._ring_cap = ring_cap
        self._retention_hours = retention_hours
        self._retention_rows = retention_rows
        self._prune_every_n = prune_every_n
        self._subscribers: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self._publishes_since_prune = 0

    # ── publish ────────────────────────────────────────────────────────────

    def publish(self, ev_type: str, data: dict, *,
                originator: Optional[dict] = None) -> int:
        from .schema import get_conn

        ts_epoch = time.time()
        payload = {"data": data}
        if originator is not None:
            payload["originator"] = originator
        payload_json = json.dumps(payload, separators=(",", ":"))

        conn = get_conn()
        cur = conn.execute(
            "INSERT INTO daemon_events (ts, kind, payload_json) VALUES (?, ?, ?)",
            (_epoch_to_iso(ts_epoch), ev_type, payload_json),
        )
        ev_id = int(cur.lastrowid)
        conn.commit()

        evt: dict = {
            "id":   ev_id,
            "ts":   ts_epoch,
            "type": ev_type,
            "data": data,
        }
        if originator is not None:
            evt["originator"] = originator

        with self._lock:
            subs = list(self._subscribers)
            self._publishes_since_prune += 1
            should_prune = self._publishes_since_prune >= self._prune_every_n
            if should_prune:
                self._publishes_since_prune = 0

        for q in subs:
            try:
                q.put_nowait(evt)
            except queue.Full:
                pass

        if should_prune:
            try:
                self._prune_old_events()
            except Exception:
                # Pruning failure must not affect publish; janitor will
                # retry on the next cycle.
                pass

        return ev_id

    # ── replay ─────────────────────────────────────────────────────────────

    def replay_since(self, since: int) -> Iterable[dict]:
        """Yield events with id > since.

        If retention has evicted events the caller would expect, a synthetic
        ``gap`` event is yielded first so SSE clients know to resync.
        """
        from .schema import get_conn
        conn = get_conn()

        oldest_row = conn.execute(
            "SELECT MIN(id) FROM daemon_events"
        ).fetchone()
        oldest = oldest_row[0] if oldest_row and oldest_row[0] is not None else None

        if since > 0 and oldest is not None and since + 1 < oldest:
            yield {
                "id":   oldest - 1,
                "ts":   time.time(),
                "type": "gap",
                "data": {
                    "missed_from": since + 1,
                    "missed_to":   oldest - 1,
                    "reason":      "retention_prune",
                },
            }

        rows = conn.execute(
            "SELECT id, ts, kind, payload_json FROM daemon_events "
            "WHERE id > ? ORDER BY id",
            (since,),
        ).fetchall()

        for row in rows:
            ev_id, ts_iso, kind, payload_json = (
                row["id"], row["ts"], row["kind"], row["payload_json"]
            )
            try:
                payload = json.loads(payload_json) if payload_json else {}
            except (TypeError, ValueError):
                payload = {}
            evt: dict = {
                "id":   int(ev_id),
                "ts":   _iso_to_epoch(ts_iso),
                "type": str(kind),
                "data": payload.get("data", {}),
            }
            if "originator" in payload:
                evt["originator"] = payload["originator"]
            yield evt

    # ── subscribe / unsubscribe ────────────────────────────────────────────

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_QUEUE_DEPTH)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    # ── internal: retention ────────────────────────────────────────────────

    def _prune_old_events(self) -> None:
        from .schema import get_conn
        conn = get_conn()

        # Time-based prune.
        cutoff = time.time() - self._retention_hours * 3600
        conn.execute(
            "DELETE FROM daemon_events WHERE ts < ?",
            (_epoch_to_iso(cutoff),),
        )
        # Row-count cap — keep the newest N.
        excess = conn.execute(
            "SELECT COUNT(*) FROM daemon_events"
        ).fetchone()[0] - self._retention_rows
        if excess > 0:
            conn.execute(
                "DELETE FROM daemon_events WHERE id IN ("
                "  SELECT id FROM daemon_events ORDER BY id LIMIT ?"
                ")",
                (int(excess),),
            )
        conn.commit()


# ── Module-level singleton ────────────────────────────────────────────────

_BUS: Optional[EventBus] = None
_BUS_LOCK = threading.Lock()


def get_bus() -> EventBus:
    global _BUS
    with _BUS_LOCK:
        if _BUS is None:
            _BUS = EventBus()
        return _BUS


def reset_bus_for_tests() -> None:
    """Drop the singleton AND truncate the daemon_events table.

    Tests rely on ``id`` starting from 1 after this — we therefore also
    clear ``sqlite_sequence`` so the AUTOINCREMENT counter resets.  Spike
    tests that call ``events.reset_bus_for_tests()`` keep passing because
    the next publish produces id == 1 just as the in-memory implementation
    used to.
    """
    global _BUS
    with _BUS_LOCK:
        _BUS = EventBus()
    try:
        from .schema import get_conn
        conn = get_conn()
        conn.execute("DELETE FROM daemon_events")
        # AUTOINCREMENT counter lives in sqlite_sequence; drop the row so
        # next insert restarts at 1.  Best-effort; absent if no inserts yet.
        try:
            conn.execute("DELETE FROM sqlite_sequence WHERE name='daemon_events'")
        except Exception:
            pass
        conn.commit()
    except Exception:
        # Test that didn't init schema yet (e.g. monkeypatched DB path):
        # nothing to truncate.  Ignore.
        pass


# ── SSE wire format ───────────────────────────────────────────────────────

def format_sse(evt: dict) -> bytes:
    """Render an event as one SSE message frame."""
    return (
        f"id: {evt['id']}\n"
        f"event: {evt['type']}\n"
        f"data: {json.dumps(evt, separators=(',', ':'))}\n\n"
    ).encode("utf-8")


def heartbeat_frame() -> bytes:
    return b":\n\n"
