"""SQLite-backed result cache with 24h TTL.

Cache key = (source_name, normalized_query, limit). Cache value = JSON blob
of the serialized Result list. Missing or expired entries return None.

Lives at ~/.cheetahclaws/research_cache.db. No-op if the directory is
read-only (e.g. sandboxed CI). Every lookup is wrapped in try/except so a
corrupt DB never breaks research runs — the worst that happens is a
full re-fetch.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from .types import Result

DEFAULT_TTL_SECONDS = 24 * 3600


def _db_path() -> Path:
    return Path.home() / ".cheetahclaws" / "research_cache.db"


def _connect() -> sqlite3.Connection | None:
    try:
        p = _db_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), timeout=2.0)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            " key TEXT PRIMARY KEY,"
            " ts  INTEGER NOT NULL,"
            " blob TEXT NOT NULL"
            ")"
        )
        conn.commit()
        return conn
    except (sqlite3.Error, OSError):
        return None


def _key(source: str, query: str, limit: int) -> str:
    s = f"{source}\x1f{query.strip().lower()}\x1f{limit}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def get(source: str, query: str, limit: int,
        ttl_seconds: int = DEFAULT_TTL_SECONDS) -> list[Result] | None:
    conn = _connect()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT ts, blob FROM cache WHERE key = ?", (_key(source, query, limit),)
        ).fetchone()
        if not row:
            return None
        ts, blob = row
        if ttl_seconds <= 0 or int(time.time()) - ts > ttl_seconds:
            return None
        raw = json.loads(blob)
        return [Result(**r) for r in raw]
    except (sqlite3.Error, json.JSONDecodeError, TypeError):
        return None
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def put(source: str, query: str, limit: int, results: list[Result]) -> None:
    conn = _connect()
    if conn is None:
        return
    try:
        blob = json.dumps([dataclasses.asdict(r) for r in results], ensure_ascii=False)
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, ts, blob) VALUES (?, ?, ?)",
            (_key(source, query, limit), int(time.time()), blob),
        )
        conn.commit()
    except (sqlite3.Error, TypeError, ValueError):
        pass
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def purge_expired(ttl_seconds: int = DEFAULT_TTL_SECONDS) -> int:
    """Best-effort cleanup of stale rows. Returns number deleted."""
    conn = _connect()
    if conn is None:
        return 0
    try:
        cutoff = int(time.time()) - ttl_seconds
        cur = conn.execute("DELETE FROM cache WHERE ts < ?", (cutoff,))
        conn.commit()
        return cur.rowcount or 0
    except sqlite3.Error:
        return 0
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
