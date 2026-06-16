"""registry.py — AgentRegistry (RFC 0010).

Map names to pids. Names are opaque strings; convention is
``/agents/<role>/<instance>`` but the kernel does not parse paths.
Tags are an optional flat list for filtering.

Strictly additive — nothing else in the codebase imports this yet.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .errors import (
    RegistryInvalidName,
    RegistryNameExists,
    RegistryNotFound,
    UnknownPid,
)

if TYPE_CHECKING:
    from cheetahclaws.daemon.rpc import CallContext, RpcRegistry


# Validation: 256-byte cap, no NUL/control characters. Path syntax
# (leading slash, alnum / _-/) is recommended but not enforced.
NAME_MAX_BYTES = 256
TAG_MAX_COUNT  = 32


# ── Dataclass ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RegistryEntry:
    name:           str
    pid:            int
    tags:           tuple
    metadata:       dict
    registered_at:  float

    def to_dict(self) -> dict:
        return {
            "name":          self.name,
            "pid":           self.pid,
            "tags":          list(self.tags),
            "metadata":      self.metadata,
            "registered_at": self.registered_at,
        }


def _row_to_entry(row: sqlite3.Row) -> RegistryEntry:
    try:
        tags = tuple(json.loads(row["tags"]))
    except (TypeError, json.JSONDecodeError):
        tags = ()
    try:
        metadata = json.loads(row["metadata"])
    except (TypeError, json.JSONDecodeError):
        metadata = {}
    return RegistryEntry(
        name=row["name"],
        pid=int(row["pid"]),
        tags=tags,
        metadata=metadata,
        registered_at=float(row["registered_at"]),
    )


def _validate_name(name) -> None:
    if not isinstance(name, str) or not name:
        raise RegistryInvalidName(
            "name must be a non-empty string", name=str(name) if name else None,
        )
    if len(name.encode("utf-8")) > NAME_MAX_BYTES:
        raise RegistryInvalidName(
            f"name exceeds {NAME_MAX_BYTES}-byte limit",
            name=name,
        )
    for ch in name:
        if ch == "\x00" or (ord(ch) < 0x20 and ch not in (" ",)):
            raise RegistryInvalidName(
                "name contains NUL or control character",
                name=name,
            )


def _coerce_tags(tags) -> list[str]:
    if tags is None:
        return []
    if not isinstance(tags, (list, tuple, set, frozenset)):
        raise RegistryInvalidName(
            "tags must be a list/tuple/set of strings",
        )
    out: list[str] = []
    seen: set = set()
    for t in tags:
        if not isinstance(t, str) or not t:
            raise RegistryInvalidName(
                f"tag entries must be non-empty strings, got {t!r}",
            )
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    if len(out) > TAG_MAX_COUNT:
        raise RegistryInvalidName(
            f"too many tags (max {TAG_MAX_COUNT})",
        )
    return out


# ── Store ──────────────────────────────────────────────────────────────────


class RegistryStore:
    """SQLite-backed registry sharing kernel.db conn + write lock."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        write_lock: Optional[threading.Lock] = None,
    ) -> None:
        self._conn = conn
        self._lock = write_lock or threading.Lock()

    def register(
        self,
        *,
        name: str,
        pid: int,
        tags=None,
        metadata: Optional[dict] = None,
    ) -> RegistryEntry:
        _validate_name(name)
        if not isinstance(pid, int):
            raise RegistryInvalidName("pid must be int")
        coerced_tags = _coerce_tags(tags)
        if metadata is not None and not isinstance(metadata, dict):
            raise RegistryInvalidName("metadata must be an object")
        meta_json = json.dumps(metadata or {}, sort_keys=True,
                               separators=(",", ":"))
        tags_json = json.dumps(coerced_tags, separators=(",", ":"))
        now = time.time()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT pid FROM agent_processes WHERE pid = ?", (pid,),
                ).fetchone()
                if row is None:
                    raise UnknownPid(pid)
                exists = self._conn.execute(
                    "SELECT 1 FROM agent_registry WHERE name = ?", (name,),
                ).fetchone()
                if exists:
                    raise RegistryNameExists(name)
                self._conn.execute(
                    "INSERT INTO agent_registry "
                    "(name, pid, tags, metadata, registered_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (name, pid, tags_json, meta_json, now),
                )
        return RegistryEntry(name=name, pid=pid, tags=tuple(coerced_tags),
                             metadata=metadata or {}, registered_at=now)

    def unregister(self, name: str) -> int:
        _validate_name(name)
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    "DELETE FROM agent_registry WHERE name = ?", (name,),
                )
                return int(cur.rowcount or 0)

    def unregister_pid(self, pid: int) -> int:
        if not isinstance(pid, int):
            raise RegistryInvalidName("pid must be int")
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    "DELETE FROM agent_registry WHERE pid = ?", (pid,),
                )
                return int(cur.rowcount or 0)

    def lookup(self, name: str) -> RegistryEntry:
        _validate_name(name)
        row = self._conn.execute(
            "SELECT * FROM agent_registry WHERE name = ?", (name,),
        ).fetchone()
        if row is None:
            raise RegistryNotFound(name)
        return _row_to_entry(row)

    def resolve_pid(self, name: str) -> int:
        return self.lookup(name).pid

    def list(
        self,
        *,
        prefix: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list, int]:
        if prefix is not None and not isinstance(prefix, str):
            raise RegistryInvalidName("prefix must be string or null")
        if tag is not None and not isinstance(tag, str):
            raise RegistryInvalidName("tag must be string or null")
        if not isinstance(limit, int) or limit < 1:
            limit = 100
        elif limit > 10_000:
            limit = 10_000
        if not isinstance(offset, int) or offset < 0:
            offset = 0

        where = []
        params: list = []
        if prefix is not None:
            where.append("name LIKE ?")
            # Escape SQL LIKE wildcards in prefix.
            esc = (prefix.replace("\\", "\\\\")
                          .replace("%", "\\%")
                          .replace("_", "\\_"))
            params.append(esc + "%")
        # Tag filtering uses JSON contains semantics; SQLite has no
        # JSON1 by default in older builds, so we do a string contains
        # check with JSON quoting. Robust enough for v1.
        if tag is not None:
            where.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        # Count total before pagination.
        count_sql = (f"SELECT COUNT(*) AS n FROM agent_registry"
                     f"{' ESCAPE ' if False else ''}"
                     f"{where_sql}")
        # The trailing trick is unnecessary — but we do need ESCAPE for
        # the LIKE clause when a prefix uses wildcard chars. Use literal
        # in the SQL for clarity.
        if prefix is not None:
            count_sql = (f"SELECT COUNT(*) AS n FROM agent_registry "
                         f"WHERE name LIKE ? ESCAPE '\\'"
                         + (" AND tags LIKE ?" if tag is not None else ""))
        elif tag is not None:
            count_sql = ("SELECT COUNT(*) AS n FROM agent_registry "
                         "WHERE tags LIKE ?")
        else:
            count_sql = "SELECT COUNT(*) AS n FROM agent_registry"

        total_row = self._conn.execute(count_sql, params).fetchone()
        total = int(total_row["n"]) if isinstance(total_row, sqlite3.Row) else int(total_row[0])

        # Build the SELECT explicitly to ensure ESCAPE is correct.
        if prefix is not None and tag is not None:
            sql = ("SELECT * FROM agent_registry "
                   "WHERE name LIKE ? ESCAPE '\\' AND tags LIKE ? "
                   "ORDER BY name ASC LIMIT ? OFFSET ?")
            rows = self._conn.execute(
                sql, (*params, limit, offset),
            ).fetchall()
        elif prefix is not None:
            sql = ("SELECT * FROM agent_registry "
                   "WHERE name LIKE ? ESCAPE '\\' "
                   "ORDER BY name ASC LIMIT ? OFFSET ?")
            rows = self._conn.execute(
                sql, (params[0], limit, offset),
            ).fetchall()
        elif tag is not None:
            sql = ("SELECT * FROM agent_registry WHERE tags LIKE ? "
                   "ORDER BY name ASC LIMIT ? OFFSET ?")
            rows = self._conn.execute(
                sql, (params[0], limit, offset),
            ).fetchall()
        else:
            sql = ("SELECT * FROM agent_registry "
                   "ORDER BY name ASC LIMIT ? OFFSET ?")
            rows = self._conn.execute(sql, (limit, offset)).fetchall()
        return [_row_to_entry(r) for r in rows], total


# ── RPC registration ───────────────────────────────────────────────────────


def register(registry: "RpcRegistry", store: RegistryStore) -> None:
    from .errors import KernelError

    def _translate(fn):
        def wrapper(params, ctx):
            try:
                return fn(params, ctx)
            except RegistryInvalidName as e:
                raise TypeError(str(e))
            except KernelError as e:
                raise RuntimeError(f"{type(e).__name__}: {e}")
        wrapper.__name__ = fn.__name__
        return wrapper

    @_translate
    def reg_register(params, ctx):
        e = store.register(
            name=_req_str(params, "name"),
            pid=_req_int(params, "pid"),
            tags=params.get("tags"),
            metadata=params.get("metadata"),
        )
        return {"name": e.name, "pid": e.pid}

    @_translate
    def reg_unregister(params, ctx):
        name = _req_str(params, "name")
        removed = store.unregister(name)
        return {"name": name, "removed": removed}

    @_translate
    def reg_unregister_pid(params, ctx):
        pid = _req_int(params, "pid")
        removed = store.unregister_pid(pid)
        return {"pid": pid, "removed": removed}

    @_translate
    def reg_lookup(params, ctx):
        e = store.lookup(_req_str(params, "name"))
        return e.to_dict()

    @_translate
    def reg_list(params, ctx):
        entries, total = store.list(
            prefix=params.get("prefix"),
            tag=params.get("tag"),
            limit=int(params.get("limit", 100)),
            offset=int(params.get("offset", 0)),
        )
        return {"entries": [e.to_dict() for e in entries], "total": total}

    registry.register("kernel.registry.register",       reg_register)
    registry.register("kernel.registry.unregister",     reg_unregister)
    registry.register("kernel.registry.unregister_pid", reg_unregister_pid)
    registry.register("kernel.registry.lookup",         reg_lookup)
    registry.register("kernel.registry.list",           reg_list)


def _req_int(params: dict, key: str) -> int:
    if key not in params:
        raise RegistryInvalidName(f"missing {key!r}")
    v = params[key]
    if not isinstance(v, int):
        raise RegistryInvalidName(f"{key!r} must be int")
    return v


def _req_str(params: dict, key: str) -> str:
    if key not in params:
        raise RegistryInvalidName(f"missing {key!r}")
    v = params[key]
    if not isinstance(v, str):
        raise RegistryInvalidName(f"{key!r} must be str")
    return v
