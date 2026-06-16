"""Per-uid cache of /draft candidates for digit-reply selection.

After the user runs ``/draft <message>`` from a bridge channel (WeChat /
Telegram / Slack), the 3 candidates are stashed here keyed by their uid.
The bridge inbound handler can then resolve a follow-up digit message
(``1`` / ``2`` / ``3``) to the chosen candidate text and echo just that
one line back — no agent invocation, no smart-reply panel, one-shot.

Design notes:
  * In-memory only — these are ephemeral display artifacts, not history.
  * One pending draft per uid (a new ``/draft`` from the same uid replaces
    any previous one).
  * TTL defaults to 10 min; expired entries are pruned on access.
  * One-shot: a successful ``take()`` deletes the entry immediately.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

_DEFAULT_TTL_S = 600.0  # 10 min


@dataclass
class _Entry:
    candidates: list[str]
    expires_at: float


_lock = threading.Lock()
_cache: dict[str, _Entry] = {}


def put(uid: str, candidates: list[str], *, ttl_s: float = _DEFAULT_TTL_S) -> None:
    """Stash ``candidates`` under ``uid``; replaces any existing entry."""
    if not uid or not candidates:
        return
    with _lock:
        _cache[uid] = _Entry(list(candidates), time.time() + ttl_s)


def take(uid: str, choice: int) -> Optional[str]:
    """Return candidate at 1-based ``choice`` for ``uid`` and remove the
    entry. Returns None if no draft is pending, the entry expired, or
    ``choice`` is out of range. The pop happens only on a successful pick.
    """
    if not uid or choice < 1:
        return None
    now = time.time()
    with _lock:
        entry = _cache.get(uid)
        if entry is None:
            return None
        if entry.expires_at < now:
            _cache.pop(uid, None)
            return None
        if choice > len(entry.candidates):
            return None
        text = entry.candidates[choice - 1]
        _cache.pop(uid, None)
        return text


def peek(uid: str) -> Optional[list[str]]:
    """Return a copy of pending candidates for ``uid`` without consuming.

    Returns None if absent or expired (and prunes if expired).
    """
    now = time.time()
    with _lock:
        entry = _cache.get(uid)
        if entry is None:
            return None
        if entry.expires_at < now:
            _cache.pop(uid, None)
            return None
        return list(entry.candidates)


def clear(uid: str) -> None:
    """Drop any pending draft for ``uid`` (called on explicit cancel)."""
    with _lock:
        _cache.pop(uid, None)
