"""WeChat smart-reply persistent storage.

Three concerns share a single SQLite file at
``~/.cheetahclaws/wx_smart_reply.db`` so a bridge restart doesn't drop
panels mid-conversation, and so style mimicking has past replies to
draw from after a daemon recycle:

- ``wx_panels``           — pending PermissionPanels (panel_id, target,
                             candidates, expires_at).  Replaces the
                             in-memory ring; janitor sweeps on access.
- ``wx_reply_history``    — every confirmed send; smart-reply prompt
                             reads the last N rows as style examples.
- ``wx_contacts``         — per-uid relationship/notes; mirrored from
                             ``~/.cheetahclaws/wx_contacts.json`` (JSON
                             stays the source of truth because it's
                             user-edited; the SQLite mirror is just a
                             cache for fast lookup).

Schema is additive and idempotent.  No migrations needed for v1.

Threading: SQLite ``check_same_thread=False`` plus a connection lock so
the poll loop, janitor, and any future RPC handler can share one
connection.  Each method is short — no long transactions — so the lock
contention is negligible.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = Path.home() / ".cheetahclaws" / "wx_smart_reply.db"
DEFAULT_CONTACTS_JSON = Path.home() / ".cheetahclaws" / "wx_contacts.json"
DEFAULT_TIMEOUT_S = 5 * 60
JANITOR_TICK_S = 30.0
HISTORY_KEEP_DAYS = 30  # rows older than this are pruned by the janitor

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS wx_panels (
        panel_id     TEXT PRIMARY KEY,
        target_uid   TEXT NOT NULL,
        target_label TEXT NOT NULL,
        message      TEXT NOT NULL,
        candidates   TEXT NOT NULL,
        created_at   REAL NOT NULL,
        expires_at   REAL NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_wx_panels_expires ON wx_panels(expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_wx_panels_target ON wx_panels(target_uid)",
    """CREATE TABLE IF NOT EXISTS wx_reply_history (
        ts        REAL NOT NULL,
        to_uid    TEXT NOT NULL,
        to_label  TEXT,
        text      TEXT NOT NULL,
        source    TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_wx_history_ts ON wx_reply_history(ts)",
    """CREATE TABLE IF NOT EXISTS wx_id_counter (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        n  INTEGER NOT NULL
    )""",
    "INSERT OR IGNORE INTO wx_id_counter (id, n) VALUES (1, 0)",
]


# ── Domain types ───────────────────────────────────────────────────────────


@dataclass
class PendingPanel:
    panel_id: str            # 2-letter monotonic ID, e.g. "AA"
    target_uid: str
    target_label: str
    message: str
    candidates: list[str]
    created_at: float
    expires_at: float


@dataclass(frozen=True)
class ReplyHistoryEntry:
    ts: float
    to_uid: str
    to_label: Optional[str]
    text: str
    source: Optional[str]


@dataclass(frozen=True)
class Contact:
    uid: str
    label: Optional[str] = None
    relationship: Optional[str] = None
    notes: Optional[str] = None


# ── Panel-ID assignment (AA..ZZ rolling) ───────────────────────────────────

def n_to_id(n: int) -> str:
    """Map a non-negative integer to a 2-letter base-26 ID (AA..ZZ).

    Wraps every 676 panels.  In practice users never queue >100
    simultaneously, so collisions across active panels are not a concern;
    if one ever happened, the SQLite primary key would refuse the insert
    and the bridge would log a warning and assign the next id.
    """
    n = n % (26 * 26)
    a, b = divmod(n, 26)
    return chr(ord("A") + a) + chr(ord("A") + b)


# ── SQLite-backed store ────────────────────────────────────────────────────


class SqliteStore:
    """Panel + reply-history persistence backed by SQLite.

    Public surface mirrors the in-memory ``InMemoryStore`` so the bridge
    can swap implementations without further changes.
    """

    def __init__(self, db_path: Optional[Path] = None,
                 *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._timeout_s = timeout_s
        # check_same_thread=False because the poll loop and janitor are
        # different threads.  We protect with our own lock.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()
        self._stop = threading.Event()
        self._janitor: Optional[threading.Thread] = None

    def _init_schema(self) -> None:
        with self._txn() as cur:
            for stmt in _SCHEMA:
                cur.execute(stmt)

    @contextmanager
    def _txn(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    # ── Janitor lifecycle ─────────────────────────────────────────────

    def start_janitor(self) -> None:
        if self._janitor is not None:
            return
        self._janitor = threading.Thread(
            target=self._janitor_loop, name="wx-smart-reply-janitor", daemon=True,
        )
        self._janitor.start()

    def stop(self) -> None:
        self._stop.set()
        if self._janitor is not None:
            self._janitor.join(timeout=2.0)
        try:
            self._conn.close()
        except Exception:
            pass

    def _janitor_loop(self) -> None:
        while not self._stop.wait(JANITOR_TICK_S):
            try:
                self.sweep_expired()
                self.prune_history(older_than_days=HISTORY_KEEP_DAYS)
            except Exception:
                # Janitor failures are non-fatal; the next tick will retry.
                pass

    # ── Panel ID generation ───────────────────────────────────────────

    def assign_next_id(self) -> str:
        with self._txn() as cur:
            cur.execute("UPDATE wx_id_counter SET n = n + 1 WHERE id = 1")
            cur.execute("SELECT n FROM wx_id_counter WHERE id = 1")
            n = cur.fetchone()["n"]
        return n_to_id(n - 1)

    # ── Panel operations ──────────────────────────────────────────────

    def put(self, panel: PendingPanel) -> None:
        with self._txn() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO wx_panels
                   (panel_id, target_uid, target_label, message,
                    candidates, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (panel.panel_id, panel.target_uid, panel.target_label,
                 panel.message, json.dumps(panel.candidates, ensure_ascii=False),
                 panel.created_at, panel.expires_at),
            )

    def take_active(self) -> Optional[PendingPanel]:
        """Return the most-recently-created non-expired panel, or None."""
        now = time.time()
        with self._txn() as cur:
            cur.execute(
                """SELECT * FROM wx_panels
                   WHERE expires_at > ?
                   ORDER BY created_at DESC LIMIT 1""",
                (now,),
            )
            row = cur.fetchone()
        return _row_to_panel(row) if row else None

    def get_by_id(self, panel_id: str) -> Optional[PendingPanel]:
        now = time.time()
        with self._txn() as cur:
            cur.execute(
                """SELECT * FROM wx_panels
                   WHERE panel_id = ? AND expires_at > ?""",
                (panel_id, now),
            )
            row = cur.fetchone()
        return _row_to_panel(row) if row else None

    def list_active(self) -> list[PendingPanel]:
        now = time.time()
        with self._txn() as cur:
            cur.execute(
                """SELECT * FROM wx_panels
                   WHERE expires_at > ?
                   ORDER BY created_at ASC""",
                (now,),
            )
            rows = cur.fetchall()
        return [_row_to_panel(r) for r in rows]

    def consume(self, target_uid: str) -> Optional[PendingPanel]:
        """Remove and return the active panel for this target_uid, if any."""
        now = time.time()
        with self._txn() as cur:
            cur.execute(
                """SELECT * FROM wx_panels
                   WHERE target_uid = ? AND expires_at > ?
                   ORDER BY created_at DESC LIMIT 1""",
                (target_uid, now),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute("DELETE FROM wx_panels WHERE panel_id = ?",
                        (row["panel_id"],))
        return _row_to_panel(row)

    def consume_by_id(self, panel_id: str) -> Optional[PendingPanel]:
        with self._txn() as cur:
            cur.execute(
                "SELECT * FROM wx_panels WHERE panel_id = ?",
                (panel_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute("DELETE FROM wx_panels WHERE panel_id = ?", (panel_id,))
        return _row_to_panel(row)

    def sweep_expired(self) -> int:
        now = time.time()
        with self._txn() as cur:
            cur.execute("DELETE FROM wx_panels WHERE expires_at <= ?", (now,))
            return cur.rowcount

    def __len__(self) -> int:
        return len(self.list_active())

    # ── Reply history ─────────────────────────────────────────────────

    def write_reply(self, *, to_uid: str, to_label: Optional[str],
                    text: str, source: Optional[str] = None,
                    ts: Optional[float] = None) -> None:
        with self._txn() as cur:
            cur.execute(
                """INSERT INTO wx_reply_history (ts, to_uid, to_label, text, source)
                   VALUES (?, ?, ?, ?, ?)""",
                (ts if ts is not None else time.time(),
                 to_uid, to_label, text, source),
            )

    def recent_replies(self, n: int = 20,
                       *, exclude_uid: Optional[str] = None) -> list[ReplyHistoryEntry]:
        """Return up to ``n`` most-recent replies, newest first.

        ``exclude_uid`` skips replies that went to a specific contact —
        used by candidate generation to avoid leaking the *current*
        thread's drafts back as "style examples" for itself.
        """
        with self._txn() as cur:
            if exclude_uid:
                cur.execute(
                    """SELECT * FROM wx_reply_history
                       WHERE to_uid != ?
                       ORDER BY ts DESC LIMIT ?""",
                    (exclude_uid, n),
                )
            else:
                cur.execute(
                    """SELECT * FROM wx_reply_history
                       ORDER BY ts DESC LIMIT ?""",
                    (n,),
                )
            rows = cur.fetchall()
        return [
            ReplyHistoryEntry(
                ts=r["ts"], to_uid=r["to_uid"],
                to_label=r["to_label"], text=r["text"], source=r["source"],
            )
            for r in rows
        ]

    def prune_history(self, *, older_than_days: int = HISTORY_KEEP_DAYS) -> int:
        cutoff = time.time() - older_than_days * 86400
        with self._txn() as cur:
            cur.execute("DELETE FROM wx_reply_history WHERE ts < ?", (cutoff,))
            return cur.rowcount


# ── In-memory fallback (no SQLite) ────────────────────────────────────────


class InMemoryStore:
    """Thread-safe in-memory store with the same shape as :class:`SqliteStore`.

    Used when SQLite init fails (read-only filesystem, broken db file,
    permissions).  All methods match the persistent store's signatures.
    """

    def __init__(self, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._panels: dict[str, PendingPanel] = {}    # panel_id → panel
        self._history: list[ReplyHistoryEntry] = []
        self._lock = threading.Lock()
        self._timeout_s = timeout_s
        self._stop = threading.Event()
        self._janitor: Optional[threading.Thread] = None
        self._counter = 0

    def start_janitor(self) -> None:
        if self._janitor is not None:
            return
        self._janitor = threading.Thread(
            target=self._janitor_loop, name="wx-smart-reply-janitor-mem",
            daemon=True,
        )
        self._janitor.start()

    def stop(self) -> None:
        self._stop.set()
        if self._janitor is not None:
            self._janitor.join(timeout=2.0)

    def _janitor_loop(self) -> None:
        while not self._stop.wait(JANITOR_TICK_S):
            self.sweep_expired()

    # Panel ID generation
    def assign_next_id(self) -> str:
        with self._lock:
            cid = n_to_id(self._counter)
            self._counter += 1
            return cid

    # Panel operations
    def put(self, panel: PendingPanel) -> None:
        with self._lock:
            self._panels[panel.panel_id] = panel

    def take_active(self) -> Optional[PendingPanel]:
        now = time.time()
        with self._lock:
            active = [p for p in self._panels.values() if p.expires_at > now]
            if not active:
                return None
            active.sort(key=lambda p: p.created_at, reverse=True)
            return active[0]

    def get_by_id(self, panel_id: str) -> Optional[PendingPanel]:
        with self._lock:
            p = self._panels.get(panel_id)
            return p if (p and p.expires_at > time.time()) else None

    def list_active(self) -> list[PendingPanel]:
        now = time.time()
        with self._lock:
            return sorted([p for p in self._panels.values() if p.expires_at > now],
                          key=lambda p: p.created_at)

    def consume(self, target_uid: str) -> Optional[PendingPanel]:
        now = time.time()
        with self._lock:
            cands = [p for p in self._panels.values()
                     if p.target_uid == target_uid and p.expires_at > now]
            if not cands:
                return None
            cands.sort(key=lambda p: p.created_at, reverse=True)
            best = cands[0]
            del self._panels[best.panel_id]
            return best

    def consume_by_id(self, panel_id: str) -> Optional[PendingPanel]:
        with self._lock:
            return self._panels.pop(panel_id, None)

    def sweep_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired = [pid for pid, p in self._panels.items()
                       if p.expires_at <= now]
            for pid in expired:
                del self._panels[pid]
            return len(expired)

    def __len__(self) -> int:
        return len(self.list_active())

    def write_reply(self, *, to_uid: str, to_label: Optional[str],
                    text: str, source: Optional[str] = None,
                    ts: Optional[float] = None) -> None:
        with self._lock:
            self._history.append(ReplyHistoryEntry(
                ts=ts if ts is not None else time.time(),
                to_uid=to_uid, to_label=to_label, text=text, source=source,
            ))
            # Cap to last 1000 to bound memory
            if len(self._history) > 1000:
                self._history = self._history[-1000:]

    def recent_replies(self, n: int = 20,
                       *, exclude_uid: Optional[str] = None) -> list[ReplyHistoryEntry]:
        with self._lock:
            rows = self._history
            if exclude_uid:
                rows = [r for r in rows if r.to_uid != exclude_uid]
            return list(reversed(rows))[:n]

    def prune_history(self, *, older_than_days: int = HISTORY_KEEP_DAYS) -> int:
        cutoff = time.time() - older_than_days * 86400
        with self._lock:
            before = len(self._history)
            self._history = [r for r in self._history if r.ts >= cutoff]
            return before - len(self._history)


# ── Store factory ──────────────────────────────────────────────────────────


def make_store(*, db_path: Optional[Path] = None,
               timeout_s: float = DEFAULT_TIMEOUT_S,
               prefer_sqlite: bool = True):
    """Build a store: SQLite by default, fall back to in-memory on failure."""
    if not prefer_sqlite:
        return InMemoryStore(timeout_s=timeout_s)
    try:
        return SqliteStore(db_path, timeout_s=timeout_s)
    except (sqlite3.Error, OSError):
        return InMemoryStore(timeout_s=timeout_s)


# ── Helpers ───────────────────────────────────────────────────────────────


def _row_to_panel(row) -> PendingPanel:
    return PendingPanel(
        panel_id=row["panel_id"],
        target_uid=row["target_uid"],
        target_label=row["target_label"],
        message=row["message"],
        candidates=json.loads(row["candidates"]),
        created_at=row["created_at"],
        expires_at=row["expires_at"],
    )


# ── Contacts (JSON-backed; user-edited source of truth) ───────────────────


class ContactsStore:
    """Thin loader/saver for ``~/.cheetahclaws/wx_contacts.json``.

    Schema::

        {
          "wxid_alice": {
            "label":        "Alice (大学同学)",
            "relationship": "close friend",
            "notes":        "她最近在找工作。语气随便，喜欢用 emoji。"
          },
          ...
        }

    Missing or unreadable file → empty store; lookups return None so
    callers can short-circuit cleanly.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_CONTACTS_JSON
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        self._mtime: float = 0.0
        self._load()

    def _load(self) -> None:
        try:
            stat = self.path.stat()
            mtime = stat.st_mtime
        except (FileNotFoundError, OSError):
            self._data = {}
            self._mtime = 0.0
            return
        if mtime == self._mtime:
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self._data, dict):
                self._data = {}
        except (json.JSONDecodeError, OSError):
            self._data = {}
        self._mtime = mtime

    def get(self, uid: str) -> Optional[Contact]:
        """Return the contact for ``uid``, reloading the file if mtime changed."""
        with self._lock:
            self._load()
            entry = self._data.get(uid)
            if not entry:
                return None
            return Contact(
                uid=uid,
                label=entry.get("label"),
                relationship=entry.get("relationship"),
                notes=entry.get("notes"),
            )

    def all(self) -> dict[str, Contact]:
        with self._lock:
            self._load()
            return {
                uid: Contact(
                    uid=uid,
                    label=v.get("label"),
                    relationship=v.get("relationship"),
                    notes=v.get("notes"),
                )
                for uid, v in self._data.items()
            }

    def set(self, contact: Contact) -> None:
        with self._lock:
            self._load()
            self._data[contact.uid] = {
                k: v for k, v in {
                    "label":        contact.label,
                    "relationship": contact.relationship,
                    "notes":        contact.notes,
                }.items() if v is not None
            }
            self._save()

    def delete(self, uid: str) -> bool:
        with self._lock:
            self._load()
            existed = self._data.pop(uid, None) is not None
            if existed:
                self._save()
            return existed

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.path)
        try:
            stat = self.path.stat()
            self._mtime = stat.st_mtime
        except OSError:
            pass
