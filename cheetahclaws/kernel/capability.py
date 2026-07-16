"""capability.py — Capability model (RFC 0005).

Stores per-agent capability bags in kernel.db. Provides:

  * ``Capability`` / ``FsGrant`` dataclasses
  * ``CapabilityStore`` — CRUD + derivation + path / glob / tool / model checks
  * ``register(rpc_registry, store)`` — RPC handlers for ``kernel.cap.*``

Enforcement is the supervisor's job; this module provides the
authoritative ``check_*`` primitives that the supervisor calls before
tool dispatch. Default-deny: a pid with no capability row gets
``allowed=False`` from every check.

Strictly additive — nothing else in the codebase imports this yet.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

from .errors import (
    CapabilityDerivationError,
    CapabilityExists,
    CapabilityInvalidGrant,
    CapabilityUnknownPid,
    InvalidPayload,
    UnknownPid,
)

if TYPE_CHECKING:
    from cheetahclaws.daemon.rpc import CallContext, RpcRegistry


# Reserved wildcard tokens (RFC 0005 §2 "Reserved tokens").
WILDCARD_ALL = "*"


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FsGrant:
    prefix: str
    mode:   str   # "r" | "rw"

    def __post_init__(self):
        if not isinstance(self.prefix, str) or not self.prefix.startswith("/"):
            raise CapabilityInvalidGrant(
                f"fs prefix must be an absolute path, got {self.prefix!r}",
                field="fs_grants.prefix",
            )
        if self.mode not in ("r", "rw"):
            raise CapabilityInvalidGrant(
                f"fs mode must be 'r' or 'rw', got {self.mode!r}",
                field="fs_grants.mode",
            )


@dataclass(frozen=True)
class Capability:
    cap_id:        int
    parent_cap_id: Optional[int]
    pid:           int
    tool_grants:   frozenset
    fs_grants:     tuple
    net_grants:    frozenset
    model_grants:  frozenset
    sub_agent:     bool
    created_at:    float

    def to_dict(self) -> dict:
        return {
            "cap_id":        self.cap_id,
            "parent_cap_id": self.parent_cap_id,
            "pid":           self.pid,
            "tool_grants":   sorted(self.tool_grants),
            "fs_grants":     [{"prefix": g.prefix, "mode": g.mode}
                              for g in self.fs_grants],
            "net_grants":    sorted(self.net_grants),
            "model_grants":  sorted(self.model_grants),
            "sub_agent":     self.sub_agent,
            "created_at":    self.created_at,
        }


# ── Glob + path matching primitives (also unit-testable) ──────────────────


def _normalize_glob(pat: str) -> str:
    """Map RFC 0005 §2 wildcard tokens to internal canonical form."""
    if pat == WILDCARD_ALL:
        return "**.*"
    return pat


def host_matches_glob(host: str, glob: str) -> bool:
    """Implement the three forms in RFC 0005 §2 "Net grants — glob format".

    * ``example.com``     — exact equality
    * ``*.example.com``   — single-level subdomain (api.example.com only)
    * ``**.example.com``  — any depth, including example.com itself
    """
    if not isinstance(host, str) or not host:
        return False
    glob = _normalize_glob(glob)
    if glob.startswith("**."):
        suffix = glob[3:]
        if suffix == "*":
            return True
        return host == suffix or host.endswith("." + suffix)
    if glob.startswith("*."):
        suffix = glob[2:]
        if "." not in host:
            return False
        return host.endswith("." + suffix) and host.count(".") == suffix.count(".") + 1
    return host == glob


def fs_grant_matches(grant: FsGrant, path: str, mode: str) -> bool:
    """True iff ``path`` is under ``grant.prefix`` with sufficient mode."""
    if not path.startswith("/"):
        return False
    if not _path_under_prefix(path, grant.prefix):
        return False
    if mode == "r":
        return grant.mode in ("r", "rw")
    if mode == "rw":
        return grant.mode == "rw"
    return False


def _path_under_prefix(path: str, prefix: str) -> bool:
    """Strict prefix check that respects directory boundaries.

    ``/agents/alice/`` matches ``/agents/alice`` and ``/agents/alice/x``,
    but not ``/agents/alicia``. We canonicalise the prefix to end with
    ``/`` for the comparison, then accept either equality with the
    boundary form or with the prefix's bare form.
    """
    if not path.startswith("/"):
        return False
    bare   = prefix.rstrip("/") if prefix != "/" else "/"
    bound  = bare if bare == "/" else bare + "/"
    if path == bare:
        return True
    return path.startswith(bound)


def _glob_subset(child_globs: frozenset, parent_globs: frozenset) -> bool:
    """RFC 0005 §3 — conservative string-equality subset.

    True iff every child glob string is also in parent_globs, OR parent
    has the universal "*" wildcard.
    """
    if WILDCARD_ALL in parent_globs:
        return True
    return child_globs.issubset(parent_globs)


def _tool_or_model_subset(child: frozenset, parent: frozenset) -> bool:
    if WILDCARD_ALL in parent:
        return True
    return child.issubset(parent)


def _fs_subset(child: Sequence, parent: Sequence) -> bool:
    """Each child grant must have a parent grant with a path ⊇ and mode ⊇."""
    for fc in child:
        ok = False
        for fp in parent:
            if not _path_under_prefix(fc.prefix, fp.prefix):
                continue
            if fc.mode == "rw" and fp.mode != "rw":
                continue
            ok = True
            break
        if not ok:
            return False
    return True


# ── Coercion helpers ───────────────────────────────────────────────────────


def _coerce_str_set(value, *, field: str) -> frozenset:
    if value is None:
        return frozenset()
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise CapabilityInvalidGrant(
            f"{field} must be a list/tuple/set of strings, got {type(value).__name__}",
            field=field,
        )
    out = set()
    for x in value:
        if not isinstance(x, str) or not x:
            raise CapabilityInvalidGrant(
                f"{field} entries must be non-empty strings, got {x!r}",
                field=field,
            )
        out.add(x)
    return frozenset(out)


def _coerce_fs_grants(value, *, field: str = "fs_grants") -> tuple:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise CapabilityInvalidGrant(
            f"{field} must be a list of {{prefix, mode}} objects",
            field=field,
        )
    out = []
    for entry in value:
        if isinstance(entry, FsGrant):
            out.append(entry)
            continue
        if not isinstance(entry, dict):
            raise CapabilityInvalidGrant(
                f"{field} entries must be objects with prefix+mode",
                field=field,
            )
        try:
            out.append(FsGrant(prefix=entry["prefix"], mode=entry["mode"]))
        except KeyError as e:
            raise CapabilityInvalidGrant(
                f"{field} entry missing field {e.args[0]!r}", field=field,
            )
    return tuple(out)


# ── Store ──────────────────────────────────────────────────────────────────


def _row_to_capability(row: sqlite3.Row) -> Capability:
    return Capability(
        cap_id        = row["cap_id"],
        parent_cap_id = row["parent_cap_id"],
        pid           = row["pid"],
        tool_grants   = frozenset(json.loads(row["tool_grants"])),
        fs_grants     = tuple(FsGrant(**g) for g in json.loads(row["fs_grants"])),
        net_grants    = frozenset(json.loads(row["net_grants"])),
        model_grants  = frozenset(json.loads(row["model_grants"])),
        sub_agent     = bool(row["sub_agent"]),
        created_at    = row["created_at"],
    )


def _serialize(grants_tuple: Sequence) -> str:
    return json.dumps(
        [{"prefix": g.prefix, "mode": g.mode} for g in grants_tuple],
        sort_keys=True, separators=(",", ":"),
    )


class CapabilityStore:
    """SQLite-backed capability store sharing kernel.db connection AND
    write lock with the other stores.

    Important: Python's ``sqlite3`` module manages an implicit transaction
    per connection (DEFERRED isolation). Two stores on independent
    Python locks but the same connection would step on each other's
    implicit transactions. We therefore require a shared lock — the
    KernelStore's lock, passed in here — so all writers serialise.
    Reads are lock-free.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        write_lock: Optional[threading.Lock] = None,
    ) -> None:
        self._conn = conn
        # Fall back to a private lock for standalone tests; production
        # always passes the KernelStore lock.
        self._lock = write_lock or threading.Lock()

    # ── Mutations ─────────────────────────────────────────────────────

    def create(
        self,
        *,
        pid: int,
        tool_grants=None,
        fs_grants=None,
        net_grants=None,
        model_grants=None,
        sub_agent: bool = False,
    ) -> Capability:
        if not isinstance(pid, int):
            raise InvalidPayload("pid must be int", field="pid")
        if not isinstance(sub_agent, bool):
            raise CapabilityInvalidGrant(
                f"sub_agent must be bool, got {type(sub_agent).__name__}",
                field="sub_agent",
            )
        tools  = _coerce_str_set(tool_grants,  field="tool_grants")
        nets   = _coerce_str_set(net_grants,   field="net_grants")
        models = _coerce_str_set(model_grants, field="model_grants")
        fsg    = _coerce_fs_grants(fs_grants)

        now = time.time()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT pid FROM agent_processes WHERE pid = ?", (pid,),
                ).fetchone()
                if row is None:
                    raise UnknownPid(pid)
                exists = self._conn.execute(
                    "SELECT 1 FROM agent_capabilities WHERE pid = ?", (pid,),
                ).fetchone()
                if exists:
                    raise CapabilityExists(pid)
                cur = self._conn.execute(
                    """
                    INSERT INTO agent_capabilities
                        (parent_cap_id, pid, tool_grants, fs_grants,
                         net_grants, model_grants, sub_agent, created_at)
                    VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (pid,
                     json.dumps(sorted(tools), separators=(",", ":")),
                     _serialize(fsg),
                     json.dumps(sorted(nets), separators=(",", ":")),
                     json.dumps(sorted(models), separators=(",", ":")),
                     1 if sub_agent else 0,
                     now),
                )
                cap_id = cur.lastrowid
                fetched = self._conn.execute(
                    "SELECT * FROM agent_capabilities WHERE cap_id = ?",
                    (cap_id,),
                ).fetchone()
        return _row_to_capability(fetched)

    def derive(
        self,
        *,
        parent_pid: int,
        child_pid: int,
        tool_grants=None,
        fs_grants=None,
        net_grants=None,
        model_grants=None,
        sub_agent: bool = False,
    ) -> Capability:
        """Derive a child capability whose grants are a subset of the
        parent's. Raises CapabilityDerivationError if any subset rule
        fails."""
        if not isinstance(parent_pid, int):
            raise InvalidPayload("parent_pid must be int", field="parent_pid")
        if not isinstance(child_pid, int):
            raise InvalidPayload("child_pid must be int", field="child_pid")
        if parent_pid == child_pid:
            raise CapabilityDerivationError(
                "parent_pid and child_pid must differ",
                field="child_pid",
            )

        parent = self.get(parent_pid)  # raises CapabilityUnknownPid
        # Coerce + validate child grants (this also runs FsGrant.__post_init__).
        c_tools  = _coerce_str_set(tool_grants,  field="tool_grants")
        c_nets   = _coerce_str_set(net_grants,   field="net_grants")
        c_models = _coerce_str_set(model_grants, field="model_grants")
        c_fsg    = _coerce_fs_grants(fs_grants)

        # ── Subset checks ────────────────────────────────────────────
        if not _tool_or_model_subset(c_tools, parent.tool_grants):
            extras = c_tools - parent.tool_grants if WILDCARD_ALL not in parent.tool_grants else set()
            raise CapabilityDerivationError(
                f"child tool_grants must be a subset of parent's; extras: {sorted(extras)}",
                field="tool_grants",
            )
        if not _tool_or_model_subset(c_models, parent.model_grants):
            extras = c_models - parent.model_grants if WILDCARD_ALL not in parent.model_grants else set()
            raise CapabilityDerivationError(
                f"child model_grants must be a subset of parent's; extras: {sorted(extras)}",
                field="model_grants",
            )
        if not _glob_subset(c_nets, parent.net_grants):
            raise CapabilityDerivationError(
                "child net_grants must be a string subset of parent's "
                "(see RFC 0005 §3 — conservative subset)",
                field="net_grants",
            )
        if not _fs_subset(c_fsg, parent.fs_grants):
            raise CapabilityDerivationError(
                "child fs_grants are not all reachable from parent's "
                "(prefix and mode must each be ⊆ parent's)",
                field="fs_grants",
            )
        if sub_agent and not parent.sub_agent:
            raise CapabilityDerivationError(
                "child sub_agent=True requires parent.sub_agent=True",
                field="sub_agent",
            )

        now = time.time()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT pid FROM agent_processes WHERE pid = ?", (child_pid,),
                ).fetchone()
                if row is None:
                    raise UnknownPid(child_pid)
                exists = self._conn.execute(
                    "SELECT 1 FROM agent_capabilities WHERE pid = ?", (child_pid,),
                ).fetchone()
                if exists:
                    raise CapabilityExists(child_pid)
                cur = self._conn.execute(
                    """
                    INSERT INTO agent_capabilities
                        (parent_cap_id, pid, tool_grants, fs_grants,
                         net_grants, model_grants, sub_agent, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (parent.cap_id, child_pid,
                     json.dumps(sorted(c_tools), separators=(",", ":")),
                     _serialize(c_fsg),
                     json.dumps(sorted(c_nets), separators=(",", ":")),
                     json.dumps(sorted(c_models), separators=(",", ":")),
                     1 if sub_agent else 0,
                     now),
                )
                cap_id = cur.lastrowid
                fetched = self._conn.execute(
                    "SELECT * FROM agent_capabilities WHERE cap_id = ?",
                    (cap_id,),
                ).fetchone()
        return _row_to_capability(fetched)

    # ── Reads ─────────────────────────────────────────────────────────

    def get(self, pid: int) -> Capability:
        if not isinstance(pid, int):
            raise InvalidPayload("pid must be int", field="pid")
        row = self._conn.execute(
            "SELECT * FROM agent_capabilities WHERE pid = ?", (pid,),
        ).fetchone()
        if row is None:
            raise CapabilityUnknownPid(pid)
        return _row_to_capability(row)

    def _get_optional(self, pid: int) -> Optional[Capability]:
        try:
            return self.get(pid)
        except CapabilityUnknownPid:
            return None

    # ── Checks (default-deny on missing rows) ─────────────────────────

    def check_tool(self, pid: int, tool: str) -> bool:
        cap = self._get_optional(pid)
        if cap is None:
            return False
        if not isinstance(tool, str) or not tool:
            return False
        return WILDCARD_ALL in cap.tool_grants or tool in cap.tool_grants

    def check_model(self, pid: int, model: str) -> bool:
        cap = self._get_optional(pid)
        if cap is None:
            return False
        if not isinstance(model, str) or not model:
            return False
        return WILDCARD_ALL in cap.model_grants or model in cap.model_grants

    def check_net(self, pid: int, host: str) -> bool:
        cap = self._get_optional(pid)
        if cap is None:
            return False
        if not isinstance(host, str) or not host:
            return False
        if WILDCARD_ALL in cap.net_grants:
            return True
        for glob in cap.net_grants:
            if host_matches_glob(host, glob):
                return True
        return False

    def check_fs(self, pid: int, path: str, mode: str) -> bool:
        cap = self._get_optional(pid)
        if cap is None:
            return False
        if not isinstance(path, str) or not path.startswith("/"):
            return False
        if mode not in ("r", "rw"):
            return False
        for grant in cap.fs_grants:
            if fs_grant_matches(grant, path, mode):
                return True
        return False


# ── RPC registration ───────────────────────────────────────────────────────


def register(registry: "RpcRegistry", store: CapabilityStore) -> None:
    """Register kernel.cap.* methods. Mirrors methods.py's translation
    pattern: InvalidPayload-shaped errors map to TypeError so the RPC
    dispatcher emits INVALID_PARAMS, other KernelError subclasses come
    back as RuntimeError carrying the class name + message."""
    from .errors import KernelError

    def _translate(fn):
        def wrapper(params, ctx):
            try:
                return fn(params, ctx)
            except (InvalidPayload, CapabilityInvalidGrant) as e:
                raise TypeError(str(e))
            except KernelError as e:
                raise RuntimeError(f"{type(e).__name__}: {e}")
        wrapper.__name__ = fn.__name__
        return wrapper

    @_translate
    def cap_create(params, ctx):
        cap = store.create(
            pid          = _req_int(params, "pid"),
            tool_grants  = params.get("tool_grants"),
            fs_grants    = params.get("fs_grants"),
            net_grants   = params.get("net_grants"),
            model_grants = params.get("model_grants"),
            sub_agent    = bool(params.get("sub_agent", False)),
        )
        return {"cap_id": cap.cap_id, "pid": cap.pid}

    @_translate
    def cap_derive(params, ctx):
        cap = store.derive(
            parent_pid   = _req_int(params, "parent_pid"),
            child_pid    = _req_int(params, "child_pid"),
            tool_grants  = params.get("tool_grants"),
            fs_grants    = params.get("fs_grants"),
            net_grants   = params.get("net_grants"),
            model_grants = params.get("model_grants"),
            sub_agent    = bool(params.get("sub_agent", False)),
        )
        return {"cap_id": cap.cap_id, "pid": cap.pid,
                "parent_cap_id": cap.parent_cap_id}

    @_translate
    def cap_get(params, ctx):
        cap = store.get(_req_int(params, "pid"))
        return cap.to_dict()

    @_translate
    def cap_check_tool(params, ctx):
        return {"allowed": store.check_tool(
            _req_int(params, "pid"), _req_str(params, "tool"))}

    @_translate
    def cap_check_model(params, ctx):
        return {"allowed": store.check_model(
            _req_int(params, "pid"), _req_str(params, "model"))}

    @_translate
    def cap_check_net(params, ctx):
        return {"allowed": store.check_net(
            _req_int(params, "pid"), _req_str(params, "host"))}

    @_translate
    def cap_check_fs(params, ctx):
        return {"allowed": store.check_fs(
            _req_int(params, "pid"),
            _req_str(params, "path"),
            _req_str(params, "mode"))}

    registry.register("kernel.cap.create",      cap_create)
    registry.register("kernel.cap.derive",      cap_derive)
    registry.register("kernel.cap.get",         cap_get)
    registry.register("kernel.cap.check_tool",  cap_check_tool)
    registry.register("kernel.cap.check_model", cap_check_model)
    registry.register("kernel.cap.check_net",   cap_check_net)
    registry.register("kernel.cap.check_fs",    cap_check_fs)


def _req_int(params: dict, key: str) -> int:
    if key not in params:
        raise InvalidPayload(f"missing required field {key!r}", field=key)
    v = params[key]
    if not isinstance(v, int):
        raise InvalidPayload(f"{key!r} must be int", field=key)
    return v


def _req_str(params: dict, key: str) -> str:
    if key not in params:
        raise InvalidPayload(f"missing required field {key!r}", field=key)
    v = params[key]
    if not isinstance(v, str):
        raise InvalidPayload(f"{key!r} must be str", field=key)
    return v
