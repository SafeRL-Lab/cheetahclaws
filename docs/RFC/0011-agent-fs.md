# Design Note: AgentFS — virtual filesystem for kernel-managed state

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Tracks roadmap:** [`0002-daemon-foundation-roadmap.md`](./0002-daemon-foundation-roadmap.md) (Phase 4 — unified state plane)
- **Builds on:** [`0003-agent-process-and-event-log.md`](./0003-agent-process-and-event-log.md), [`0005-capability-model.md`](./0005-capability-model.md), [`0006-resource-ledger.md`](./0006-resource-ledger.md)

This RFC introduces **AgentFS** — a kernel-managed key-value object
store with hierarchical paths. It is the missing unified state plane:
today the codebase scatters durable agent state across `memory/`,
`checkpoint/`, `skill/`, `task/`, and `prompts/` directories under
`~/.cheetahclaws/`, each with its own format, naming, and lifecycle.
The supervisor and tools have to know about each layout. AgentFS
collapses them all into one path-keyed object table that the kernel
owns.

The kernel **stores and audits**; the supervisor and tools call the
AgentFS API to read/write. As with prior phases, AgentFS does not
replace the existing per-feature stores in this RFC. Existing modules
(`memory/`, `checkpoint/`, etc.) keep functioning unchanged. A separate,
optional follow-up patch can re-implement them on top of AgentFS once
the API has bedded in.

This RFC ships **purely additive** code. Schema version bumps v4 → v5.

## 1. Goals & non-goals

**Goals:**

1. **Path-keyed object store.** Hierarchical paths
   (`/memory/<pid>/...`, `/skills/<name>`, `/tasks/<pid>/<id>`)
   address opaque content blobs. The kernel does not interpret content.
2. **Per-agent isolation via capabilities.** A caller passes its `pid`
   on every operation; the kernel writes that into the audit trail.
   Enforcement of "may this pid read /memory/alice/?" is the
   supervisor's job, using existing `kernel.cap.check_fs(pid, path,
   mode)`.
3. **Quota integration.** When an agent has a ledger row for
   `fs_w_bytes`, AgentFS charges write bytes against it. Over-limit
   surfaces as `FsQuotaExceeded`. No charge if the ledger row is
   absent.
4. **Bounded blob size.** Objects are capped at 16 MB by default; the
   kernel rejects larger writes. Configurable per-store but not per
   agent (yet).
5. **Audit.** Every mutation is recorded against the agent's event log
   (kind `kernel.fs.write`, `kernel.fs.delete`).

**Non-goals (v1):**

- **Streaming reads/writes.** All operations are whole-blob. Tools
  needing streaming chunk into multiple objects.
- **Symlinks / mounts.** Paths are flat strings — no
  link-following, no overlay mounts. v2 may add.
- **POSIX permissions.** Mode is just `'ro'` vs `'rw'`. No owner /
  group / mode bits beyond that. Capability layer (RFC 0005) handles
  per-agent reachability.
- **Versioning.** A write replaces. Git-style history is a follow-up
  RFC if needed.
- **Filesystem-backed blobs.** v1 stores content as SQLite BLOB. For
  small objects (memory entries, skill manifests, task state) this is
  fine; for very large checkpoints it bloats kernel.db. A future RFC
  can add an inode → on-disk-file backing without changing the API.

## 2. Mount-point conventions

The kernel **does not enforce** path syntax — paths are opaque keys.
But tooling will rely on these conventions:

| Prefix | Owner | Purpose |
|---|---|---|
| `/memory/<pid>/...` | one agent | Persistent memory entries (RFC 0006 dimension `tokens` charged when populated by LLM) |
| `/checkpoints/<pid>/<ts>` | one agent | Session conversation snapshot |
| `/skills/<name>` | shared | Skill manifests (read by many agents) |
| `/tasks/<pid>/<task_id>` | one agent | Task list entries |
| `/shared/<topic>/...` | shared | Cross-agent shared scratchpad |
| `/scratch/<pid>/...` | one agent | Transient agent scratch — caller is expected to delete on terminate |

These are conventions, not rules. The kernel rejects only:
- empty paths
- paths not starting with `/`
- paths containing `\x00` or other control characters
- paths > 1024 bytes UTF-8
- paths containing literal `/../` (path traversal)

## 3. Data model

### `FsObject`

```python
@dataclass(frozen=True)
class FsObject:
    path:        str
    owner_pid:   int               # who created this row
    size:        int               # bytes
    mode:        str               # 'rw' | 'ro'
    metadata:    dict              # opaque caller hints
    created_at:  float
    updated_at:  float
    accessed_at: float | None
    # content NOT in the dataclass — tools call read() to fetch the BLOB
```

`owner_pid` is **informational** — it records who created the row.
Capability checks decide who can read/write later. Multiple agents may
operate on the same path (think `/skills/research`).

### Schema (DDL)

```sql
CREATE TABLE IF NOT EXISTS agent_fs_objects (
    path        TEXT    PRIMARY KEY,
    owner_pid   INTEGER NOT NULL,
    content     BLOB    NOT NULL,
    size        INTEGER NOT NULL,
    mode        TEXT    NOT NULL DEFAULT 'rw'
                CHECK(mode IN ('rw', 'ro')),
    metadata    TEXT    NOT NULL DEFAULT '{}',
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL,
    accessed_at REAL,
    FOREIGN KEY (owner_pid) REFERENCES agent_processes(pid)
);

CREATE INDEX IF NOT EXISTS idx_agent_fs_owner
    ON agent_fs_objects(owner_pid);

CREATE INDEX IF NOT EXISTS idx_agent_fs_path_prefix
    ON agent_fs_objects(path);
```

Schema v4 → v5. Additive (one new table + indexes).

## 4. Operations

### `write(pid, path, content, *, mode='rw', metadata=None, if_absent=False)`

Atomic create-or-update.

- `pid`: caller (recorded as owner on first create; later writes don't
  change owner).
- `path`: validated.
- `content`: `bytes`. Strings are encoded to UTF-8 first.
- `mode`: 'rw' (default) or 'ro'. After mode='ro' is set, subsequent
  writes raise `FsReadOnly` until `set_mode(path, 'rw')`.
- `metadata`: opaque dict.
- `if_absent=True`: if the path already exists, raise
  `FsAlreadyExists` (create-only).

Charges `len(content)` bytes against the ledger dim `fs_w_bytes` if
the agent has a row for it. Over-limit raises `FsQuotaExceeded` and
**rolls back the write**.

### `read(pid, path) -> (content, FsObject)`

Returns `(content_bytes, metadata)`. Updates `accessed_at`.

### `stat(path) -> FsObject`

Metadata only (no content).

### `list(prefix=None, owner_pid=None, limit=100, offset=0) -> tuple[list[FsObject], int]`

Returns objects whose `path` starts with `prefix`, optionally filtered
to one owner. Order: `path ASC`.

### `delete(pid, path) -> bool`

Removes the row. Returns True if a row was deleted, False otherwise.
Idempotent.

### `exists(path) -> bool`

Cheap existence check.

### `set_mode(path, mode)`

Toggle 'rw' ↔ 'ro'.

### `gc_orphaned(pid) -> int`

Deletes all rows where `owner_pid = pid`. Used by the supervisor on
agent termination cleanup. Caller-driven — the kernel does not auto-
cascade; this preserves audit data when desired.

## 5. Capability + ledger integration

### Capability (advisory)

The kernel does **not** enforce capability checks at the AgentFS
layer in v1. The supervisor calls `kernel.cap.check_fs(pid, path,
mode)` before dispatching the FS op. RFC 0005's `fs_grants` apply
literally to AgentFS paths — a grant `{prefix: "/memory/alice/",
mode: "rw"}` covers AgentFS reads of `/memory/alice/*`.

This means a supervisor can authorise an agent's AgentFS reach using
the same capability model as host fs reach; the supervisor decides
which dispatcher (host or AgentFS) to route through.

### Ledger (enforced)

When the caller has a ledger row for `fs_w_bytes`, every successful
`write` triggers:

```python
result = ledger.charge(pid=pid, dim="fs_w_bytes", amount=len(content))
if result.over_limit and result.first_breach:
    # The write happened (already committed in the same tx); the
    # ChargeResult signals over_limit. The supervisor reads the
    # event log / ChargeResult and decides whether to suspend.
```

Note: in v1 the AgentFS write commits **before** charging. If the
charge would push over, the write is allowed (consistent with RFC
0006: "real-world API calls have already happened"). For "refuse if
charge would exceed", the supervisor calls `ledger.check(pid,
"fs_w_bytes", len(content))` before invoking AgentFS. This keeps the
two layers cleanly separated.

If there's no ledger row for `fs_w_bytes`, no charge occurs. Quota is
strictly opt-in.

## 6. RPC surface

```
kernel.fs.write
  params: { pid, path, content, mode?, metadata?, if_absent? }
  result: { path, size, owner_pid }
  -- content is base64-encoded over JSON-RPC

kernel.fs.read
  params: { pid, path }
  result: { path, content, size, mode, metadata, ... }

kernel.fs.stat
  params: { path }
  result: FsObject (no content)

kernel.fs.list
  params: { prefix?, owner_pid?, limit?, offset? }
  result: { entries: [FsObject], total }

kernel.fs.delete
  params: { pid, path }
  result: { path, removed: bool }

kernel.fs.exists
  params: { path }
  result: { exists: bool }

kernel.fs.set_mode
  params: { path, mode }
  result: { path, mode }

kernel.fs.gc_orphaned
  params: { pid }
  result: { removed: int }
```

`content` in `write` and `read` is **base64-encoded** when it travels
over the JSON-RPC wire (JSON cannot represent arbitrary bytes).
Library bindings can hide this. For local Python callers
(supervisor, tools running in-process), the AgentFSStore API takes/
returns raw `bytes` directly.

### Error codes

| Code | Name | Meaning |
|---|---|---|
| -32161 | `kernel_fs_not_found` | path missing |
| -32162 | `kernel_fs_already_exists` | if_absent=True hit existing |
| -32163 | `kernel_fs_invalid_path` | empty / NUL / control / .. / oversize |
| -32164 | `kernel_fs_read_only` | write to mode='ro' |
| -32165 | `kernel_fs_quota_exceeded` | ledger fs_w_bytes hard_limit hit |

## 7. Backwards compatibility

- Schema bump v4 → v5 is additive. Forward migration transparent.
- No existing module modified.
- Existing `memory/`, `checkpoint/`, etc., keep working untouched.
  They will not be ported in this RFC. A separate optional patch
  later can re-implement them on top of AgentFS — the API surface is
  designed to support that without breaking callers.

## 8. Open questions

1. **Should writes auto-set `accessed_at`?** Currently no — only
   reads bump it. Reasoning: `accessed_at` is for cache eviction
   logic later. Writes are a different signal. **Lean: keep writes
   not bumping accessed_at.**
2. **Default size cap.** 16 MB is comfortable for memory entries and
   most checkpoints, but a 50-page research paper checkpoint with
   embedded plots could exceed. Configurable via `AgentFSStore(...,
   max_object_bytes=...)`. **OK to ship 16 MB default.**
3. **Should `delete` charge a (negative) refund of `fs_w_bytes`?**
   Currently no — `used` only grows. Argument for refund: an agent
   that writes-and-deletes should get its budget back. Argument
   against: tracking historic write volume is itself useful for audit.
   **Lean: no refund. Operator can call `ledger.refund` explicitly
   if desired.**

## 9. Acceptance criteria

A PR claiming this RFC must:

1. Schema migrates v4 → v5 forward.
2. `write` + `read` round-trip raw bytes (incl. binary, non-UTF-8
   payloads).
3. Path validation rejects: empty, no leading slash, NUL,
   control chars, `..` segments, oversize.
4. `if_absent=True` raises `FsAlreadyExists` when the path exists.
5. `set_mode('ro')` then `write` raises `FsReadOnly`.
6. `list(prefix=...)` returns only matching paths; LIKE wildcards in
   prefix are escaped.
7. `delete` is idempotent.
8. With a ledger row for `fs_w_bytes`, `write` charges; over-limit
   surfaces `FsQuotaExceeded` and rolls back the write.
9. Concurrent writes from N threads to N distinct paths leave all N
   rows present and intact.
10. RPC surface works end-to-end through the daemon.
11. No file outside `kernel/`, `tests/`, `docs/RFC/` modified.
