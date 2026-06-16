"""originator.py — client_id minting, persistence, and resume.

The RFC §2.5 says: "originator disconnects mid-request — held until timeout.
On reconnect, the originator gets the request back via SSE replay scoped to
its own pending requests."  That requires a stable client identity that
survives client-process restarts.

Model:
- Daemon mints a client_id (32 hex) on first connect for a given client kind.
- Client persists it at ~/.cheetahclaws/clients/<kind>.id (mode 0600).
- Subsequent connects present the saved id via X-Client-Id header.
- Daemon validates: known id → reuse; unknown id from a peer that already
  has a different active id → reject. Otherwise register.

The store here is in-memory + a JSON sidecar so tests can survive a daemon
restart in the same temp dir.
"""
from __future__ import annotations

import json
import secrets
import threading
from pathlib import Path
from typing import Optional

CLIENT_ID_HEADER = "X-Client-Id"
CLIENT_KIND_HEADER = "X-Client-Kind"


class OriginatorStore:
    """Tracks known client_ids → kind. Persisted to a single JSON file so
    a daemon restart in the same data dir keeps continuity."""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "originators.json"
        self._lock = threading.Lock()
        self._known: dict[str, str] = {}  # client_id → kind
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._known = json.loads(self._path.read_text())
            except Exception:
                self._known = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._known, indent=2))

    def mint(self, kind: str) -> str:
        """Issue a new client_id for the given kind."""
        cid = secrets.token_hex(16)
        with self._lock:
            self._known[cid] = kind
            self._save()
        return cid

    def resolve(self, presented_id: Optional[str], kind: str) -> tuple[str, bool]:
        """Return (client_id, was_minted). If presented_id is known and
        matches a kind, reuse it; otherwise mint a new one."""
        with self._lock:
            if presented_id and presented_id in self._known:
                # Allow kind to be empty (client doesn't care to announce);
                # reject only on explicit mismatch.
                if kind and self._known[presented_id] != kind:
                    # Mismatched kind → mint new instead of reusing.
                    cid = secrets.token_hex(16)
                    self._known[cid] = kind
                    self._save()
                    return cid, True
                return presented_id, False
        # Mint outside lock to avoid double-acquire path
        return self.mint(kind or "unknown"), True

    def kind_of(self, client_id: str) -> Optional[str]:
        with self._lock:
            return self._known.get(client_id)
