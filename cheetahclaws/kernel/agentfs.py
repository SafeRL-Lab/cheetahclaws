"""agentfs.py — AgentFS virtual filesystem (RFC 0011).

Hierarchical key-value object store with paths like ``/memory/<pid>/...``,
``/checkpoints/<pid>/<ts>``, ``/skills/<name>``. Content is stored as
SQLite BLOB; v1 caps single-object size at 16 MB. Capability and ledger
integration are advisory: the kernel charges fs_w_bytes if a row
exists, but does not auto-suspend; the supervisor is the policy maker.

Strictly additive — nothing else in the codebase imports this yet.
"""
from __future__ import annotations

import base64
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .errors import (
    FsAlreadyExists,
    FsInvalidPath,
    FsNotFound,
    FsQuotaExceeded,
    FsReadOnly,
    UnknownPid,
)

if TYPE_CHECKING:
    from cheetahclaws.daemon.rpc import CallContext, RpcRegistry
    from .ledger import LedgerStore


# Default cap for one object's content. Override per-store.
DEFAULT_MAX_OBJECT_BYTES = 16 * 1024 * 1024
# Path constraints (RFC 0011 §2)
MAX_PATH_BYTES = 1024
QUOTA_DIM = "fs_w_bytes"


# ── Dataclass ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FsObject:
    path:        str
    owner_pid:   int
    size:        int
    mode:        str            # 'rw' | 'ro'
    metadata:    dict
    created_at:  float
    updated_at:  float
    accessed_at: Optional[float]

    def to_dict(self) -> dict:
        return {
            "path":        self.path,
            "owner_pid":   self.owner_pid,
            "size":        self.size,
            "mode":        self.mode,
            "metadata":    self.metadata,
            "created_at":  self.created_at,
            "updated_at":  self.updated_at,
            "accessed_at": self.accessed_at,
        }


def _row_to_object(row: sqlite3.Row) -> FsObject:
    try:
        metadata = json.loads(row["metadata"])
    except (TypeError, json.JSONDecodeError):
        metadata = {}
    return FsObject(
        path        = row["path"],
        owner_pid   = int(row["owner_pid"]),
        size        = int(row["size"]),
        mode        = row["mode"],
        metadata    = metadata,
        created_at  = float(row["created_at"]),
        updated_at  = float(row["updated_at"]),
        accessed_at = (None if row["accessed_at"] is None
                       else float(row["accessed_at"])),
    )


# ── Path validation ────────────────────────────────────────────────────────


def _validate_path(path) -> None:
    if not isinstance(path, str) or not path:
        raise FsInvalidPath("path must be a non-empty string")
    if not path.startswith("/"):
        raise FsInvalidPath(
            "path must start with /", path=path,
        )
    if len(path.encode("utf-8")) > MAX_PATH_BYTES:
        raise FsInvalidPath(
            f"path exceeds {MAX_PATH_BYTES}-byte limit", path=path,
        )
    for ch in path:
        if ch == "\x00" or (ord(ch) < 0x20 and ch not in (" ",)):
            raise FsInvalidPath(
                "path contains NUL or control character", path=path,
            )
    # Reject traversal segments. We require literal "/../" boundaries
    # so a legitimate path like "/agents/file.tar.gz" passes.
    if "/../" in path or path.endswith("/..") or path.startswith("../"):
        raise FsInvalidPath(
            "path contains '..' segment", path=path,
        )


# ── Store ──────────────────────────────────────────────────────────────────


class AgentFSStore:
    """SQLite-backed virtual FS sharing kernel.db conn + write lock.

    The optional ``ledger`` argument enables ``fs_w_bytes`` quota
    charging on every write. When None, writes don't charge — the
    supervisor's policy.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        write_lock: Optional[threading.Lock] = None,
        *,
        ledger: Optional["LedgerStore"] = None,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
    ) -> None:
        self._conn = conn
        self._lock = write_lock or threading.Lock()
        self._ledger = ledger
        self._max_object_bytes = max_object_bytes

    # ── write ─────────────────────────────────────────────────────────

    def write(
        self,
        *,
        pid: int,
        path: str,
        content: bytes,
        mode: str = "rw",
        metadata: Optional[dict] = None,
        if_absent: bool = False,
    ) -> FsObject:
        if not isinstance(pid, int):
            raise FsInvalidPath("pid must be int")
        _validate_path(path)
        if isinstance(content, str):
            content = content.encode("utf-8")
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise FsInvalidPath(
                "content must be bytes (or str, encoded UTF-8)",
                path=path,
            )
        content = bytes(content)
        if len(content) > self._max_object_bytes:
            raise FsInvalidPath(
                f"content size {len(content)} exceeds max "
                f"{self._max_object_bytes}",
                path=path,
            )
        if mode not in ("rw", "ro"):
            raise FsInvalidPath(
                "mode must be 'rw' or 'ro'", path=path,
            )
        if metadata is not None and not isinstance(metadata, dict):
            raise FsInvalidPath("metadata must be an object", path=path)
        meta_json = json.dumps(metadata or {}, sort_keys=True,
                               separators=(",", ":"))
        now = time.time()
        size = len(content)

        with self._lock:
            with self._conn:
                pid_row = self._conn.execute(
                    "SELECT pid FROM agent_processes WHERE pid = ?", (pid,),
                ).fetchone()
                if pid_row is None:
                    raise UnknownPid(pid)
                existing = self._conn.execute(
                    "SELECT mode, owner_pid, created_at "
                    "FROM agent_fs_objects WHERE path = ?", (path,),
                ).fetchone()
                if existing is not None:
                    if if_absent:
                        raise FsAlreadyExists(path)
                    if existing["mode"] == "ro":
                        raise FsReadOnly(path)
                    # Update. Owner stays the original creator.
                    self._conn.execute(
                        "UPDATE agent_fs_objects "
                        "SET content = ?, size = ?, mode = ?, "
                        "    metadata = ?, updated_at = ? "
                        "WHERE path = ?",
                        (content, size, mode, meta_json, now, path),
                    )
                    owner = int(existing["owner_pid"])
                    created_at = float(existing["created_at"])
                else:
                    self._conn.execute(
                        "INSERT INTO agent_fs_objects "
                        "(path, owner_pid, content, size, mode, "
                        " metadata, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (path, pid, content, size, mode, meta_json,
                         now, now),
                    )
                    owner = pid
                    created_at = now

                # Optional ledger charge — same transaction so a quota
                # rollback also rolls back the FS row.
                if self._ledger is not None:
                    led_row = self._conn.execute(
                        "SELECT used, hard_limit FROM agent_ledgers "
                        "WHERE pid = ? AND dim = ?",
                        (pid, QUOTA_DIM),
                    ).fetchone()
                    if led_row is not None:
                        new_used = int(led_row["used"]) + size
                        hard_limit = int(led_row["hard_limit"])
                        if new_used > hard_limit:
                            raise FsQuotaExceeded(
                                pid, QUOTA_DIM, new_used, hard_limit,
                            )
                        self._conn.execute(
                            "UPDATE agent_ledgers "
                            "SET used = ?, updated_at = ? "
                            "WHERE pid = ? AND dim = ?",
                            (new_used, now, pid, QUOTA_DIM),
                        )

        return FsObject(
            path        = path,
            owner_pid   = owner,
            size        = size,
            mode        = mode,
            metadata    = metadata or {},
            created_at  = created_at,
            updated_at  = now,
            accessed_at = None,
        )

    # ── read ──────────────────────────────────────────────────────────

    def read(self, path: str) -> tuple[bytes, FsObject]:
        _validate_path(path)
        now = time.time()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT * FROM agent_fs_objects WHERE path = ?",
                    (path,),
                ).fetchone()
                if row is None:
                    raise FsNotFound(path)
                self._conn.execute(
                    "UPDATE agent_fs_objects SET accessed_at = ? "
                    "WHERE path = ?",
                    (now, path),
                )
                content = bytes(row["content"])
                # Re-read to pick up the updated accessed_at.
                refreshed = self._conn.execute(
                    "SELECT * FROM agent_fs_objects WHERE path = ?",
                    (path,),
                ).fetchone()
        return content, _row_to_object(refreshed)

    # ── stat ──────────────────────────────────────────────────────────

    def stat(self, path: str) -> FsObject:
        _validate_path(path)
        row = self._conn.execute(
            "SELECT path, owner_pid, size, mode, metadata, "
            "       created_at, updated_at, accessed_at "
            "FROM agent_fs_objects WHERE path = ?",
            (path,),
        ).fetchone()
        if row is None:
            raise FsNotFound(path)
        return _row_to_object(row)

    # ── exists ────────────────────────────────────────────────────────

    def exists(self, path: str) -> bool:
        _validate_path(path)
        row = self._conn.execute(
            "SELECT 1 FROM agent_fs_objects WHERE path = ?",
            (path,),
        ).fetchone()
        return row is not None

    # ── list ──────────────────────────────────────────────────────────

    def list(
        self,
        *,
        prefix: Optional[str] = None,
        owner_pid: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[FsObject], int]:
        if prefix is not None:
            if not isinstance(prefix, str):
                raise FsInvalidPath("prefix must be str or null")
            # We don't require leading slash on prefix to support filter
            # like "memory/" as well as "/memory/", but we DO require
            # printable + bounded.
            for ch in prefix:
                if ch == "\x00" or (ord(ch) < 0x20 and ch != " "):
                    raise FsInvalidPath(
                        "prefix contains NUL or control character",
                    )
        if owner_pid is not None and not isinstance(owner_pid, int):
            raise FsInvalidPath("owner_pid must be int or null")
        if not isinstance(limit, int) or limit < 1:
            limit = 100
        elif limit > 10_000:
            limit = 10_000
        if not isinstance(offset, int) or offset < 0:
            offset = 0

        # Build SQL with LIKE escape for safe prefix matching.
        params: list = []
        where: list[str] = []
        if prefix is not None:
            where.append("path LIKE ? ESCAPE '\\'")
            esc = (prefix.replace("\\", "\\\\")
                          .replace("%", "\\%")
                          .replace("_", "\\_"))
            params.append(esc + "%")
        if owner_pid is not None:
            where.append("owner_pid = ?")
            params.append(owner_pid)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        total_row = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM agent_fs_objects{where_sql}",
            params,
        ).fetchone()
        total = (int(total_row["n"]) if isinstance(total_row, sqlite3.Row)
                 else int(total_row[0]))

        rows = self._conn.execute(
            f"SELECT path, owner_pid, size, mode, metadata, "
            f"       created_at, updated_at, accessed_at "
            f"FROM agent_fs_objects{where_sql} "
            f"ORDER BY path ASC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return [_row_to_object(r) for r in rows], total

    # ── delete ────────────────────────────────────────────────────────

    def delete(self, path: str) -> bool:
        _validate_path(path)
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    "DELETE FROM agent_fs_objects WHERE path = ?",
                    (path,),
                )
                return (cur.rowcount or 0) > 0

    # ── set_mode ──────────────────────────────────────────────────────

    def set_mode(self, path: str, mode: str) -> FsObject:
        _validate_path(path)
        if mode not in ("rw", "ro"):
            raise FsInvalidPath("mode must be 'rw' or 'ro'", path=path)
        now = time.time()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT 1 FROM agent_fs_objects WHERE path = ?",
                    (path,),
                ).fetchone()
                if row is None:
                    raise FsNotFound(path)
                self._conn.execute(
                    "UPDATE agent_fs_objects SET mode = ?, updated_at = ? "
                    "WHERE path = ?",
                    (mode, now, path),
                )
        return self.stat(path)

    # ── gc_orphaned ───────────────────────────────────────────────────

    def gc_orphaned(self, pid: int) -> int:
        if not isinstance(pid, int):
            raise FsInvalidPath("pid must be int")
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    "DELETE FROM agent_fs_objects WHERE owner_pid = ?",
                    (pid,),
                )
                return int(cur.rowcount or 0)


# ── RPC registration ───────────────────────────────────────────────────────


def register(registry: "RpcRegistry", store: AgentFSStore) -> None:
    from .errors import KernelError

    def _translate(fn):
        def wrapper(params, ctx):
            try:
                return fn(params, ctx)
            except FsInvalidPath as e:
                raise TypeError(str(e))
            except KernelError as e:
                raise RuntimeError(f"{type(e).__name__}: {e}")
        wrapper.__name__ = fn.__name__
        return wrapper

    @_translate
    def fs_write(params, ctx):
        pid = _req_int(params, "pid")
        path = _req_str(params, "path")
        content_b64 = _req_str(params, "content")
        try:
            content = base64.b64decode(content_b64, validate=True)
        except Exception:
            raise FsInvalidPath("content must be base64-encoded bytes",
                                path=path)
        obj = store.write(
            pid=pid, path=path, content=content,
            mode=str(params.get("mode", "rw")),
            metadata=params.get("metadata"),
            if_absent=bool(params.get("if_absent", False)),
        )
        return {"path": obj.path, "size": obj.size,
                "owner_pid": obj.owner_pid, "mode": obj.mode}

    @_translate
    def fs_read(params, ctx):
        path = _req_str(params, "path")
        content, obj = store.read(path)
        d = obj.to_dict()
        d["content"] = base64.b64encode(content).decode("ascii")
        return d

    @_translate
    def fs_stat(params, ctx):
        path = _req_str(params, "path")
        return store.stat(path).to_dict()

    @_translate
    def fs_exists(params, ctx):
        path = _req_str(params, "path")
        return {"exists": store.exists(path)}

    @_translate
    def fs_list(params, ctx):
        prefix = params.get("prefix")
        owner = params.get("owner_pid")
        limit = int(params.get("limit", 100))
        offset = int(params.get("offset", 0))
        entries, total = store.list(
            prefix=prefix, owner_pid=owner,
            limit=limit, offset=offset,
        )
        return {
            "entries": [e.to_dict() for e in entries],
            "total":   total,
        }

    @_translate
    def fs_delete(params, ctx):
        path = _req_str(params, "path")
        removed = store.delete(path)
        return {"path": path, "removed": removed}

    @_translate
    def fs_set_mode(params, ctx):
        path = _req_str(params, "path")
        mode = _req_str(params, "mode")
        obj = store.set_mode(path, mode)
        return {"path": obj.path, "mode": obj.mode}

    @_translate
    def fs_gc_orphaned(params, ctx):
        pid = _req_int(params, "pid")
        removed = store.gc_orphaned(pid)
        return {"pid": pid, "removed": removed}

    registry.register("kernel.fs.write",       fs_write)
    registry.register("kernel.fs.read",        fs_read)
    registry.register("kernel.fs.stat",        fs_stat)
    registry.register("kernel.fs.exists",      fs_exists)
    registry.register("kernel.fs.list",        fs_list)
    registry.register("kernel.fs.delete",      fs_delete)
    registry.register("kernel.fs.set_mode",    fs_set_mode)
    registry.register("kernel.fs.gc_orphaned", fs_gc_orphaned)


def _req_int(params: dict, key: str) -> int:
    if key not in params:
        raise FsInvalidPath(f"missing {key!r}")
    v = params[key]
    if not isinstance(v, int):
        raise FsInvalidPath(f"{key!r} must be int")
    return v


def _req_str(params: dict, key: str) -> str:
    if key not in params:
        raise FsInvalidPath(f"missing {key!r}")
    v = params[key]
    if not isinstance(v, str):
        raise FsInvalidPath(f"{key!r} must be str")
    return v
