# Design Note: Capability Model — per-agent authorisation

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Tracks roadmap:** [`0002-daemon-foundation-roadmap.md`](./0002-daemon-foundation-roadmap.md) (Phase 2 — authorisation)
- **Builds on:** [`0003-agent-process-and-event-log.md`](./0003-agent-process-and-event-log.md)
- **Sibling RFC:** [`0006-resource-ledger.md`](./0006-resource-ledger.md)

This RFC defines the **second** of three Phase-2 invariants:
**capability is the only authorisation**. Agents cannot use a tool, read
a file, dial a network endpoint, or call a model unless their capability
bag explicitly grants it. Children spawned by a parent agent receive a
**strict subset** of the parent's grants — never a superset, never a
disjoint set.

Phase 1 gave us identity (RFC 0003) and isolation (RFC 0008). This RFC
gives us authorisation. RFC 0006 (ResourceLedger) gives us accounting.
Phase 3 (RFCs 0007, 0009, 0010) builds scheduling and IPC on top.

The kernel **stores and checks** capabilities; **enforcement** is the
supervisor's job (F-4 + future tool dispatcher patches). Until those land,
the kernel exposes `kernel.cap.check_*` as advisory RPCs that the
supervisor will call before tool dispatch. This RFC ships purely
additive code; nothing existing is modified.

## 1. Threat model

The realistic threat: a sub-agent spawned for a narrow task ("rewrite
this file") has, by default, the *same* tool/path/network reach as the
parent that spawned it. A naive prompt-injected sub-agent therefore has
the run of the host. Capability-based security collapses this surface:
a sub-agent only sees what the parent decided to share.

**In scope:**
- Tool whitelist per agent.
- Filesystem read/write whitelist per agent.
- Network domain whitelist per agent.
- Model whitelist per agent.
- Sub-agent spawn permission per agent.
- Strict child-⊆-parent derivation, audited via parent_cap_id chain.

**Out of scope (v1):**
- Quota / budget — that's RFC 0006.
- ACLs across users — single-user host (per RFC 0001 §3).
- Capability transfer between unrelated agents — derivation only.
- Time-bounded grants — capabilities live as long as the agent.
- Capability revocation — implicit on agent termination.

## 2. Data model

### `Capability`

```python
@dataclass(frozen=True)
class Capability:
    cap_id:        int                    # PK
    parent_cap_id: int | None             # the cap this was derived from
    pid:           int                    # owning agent (1:1 with AgentProcess)
    tool_grants:   frozenset[str]         # allowed tool names
    fs_grants:     tuple[FsGrant, ...]    # path prefix + mode list
    net_grants:    frozenset[str]         # allowed domain globs
    model_grants:  frozenset[str]         # allowed model names
    sub_agent:     bool                   # may spawn children?
    created_at:    float
```

### `FsGrant`

```python
@dataclass(frozen=True)
class FsGrant:
    prefix: str    # absolute path prefix (e.g. "/agents/alice/")
    mode:   str    # "r"  (read-only)  |  "rw"  (read+write)
```

A path is allowed for read iff some grant's `prefix` is a prefix of the
canonicalised path AND the grant's `mode` includes "r" (i.e. is "r" or
"rw"). Same for write with mode "rw". Multiple matching grants combine
via union — strongest mode wins.

### Net grants — glob format

Three forms, no other syntax in v1:

| Pattern | Matches |
|---|---|
| `example.com` | exactly `example.com` |
| `*.example.com` | any single-level subdomain (`api.example.com`, but not `a.b.example.com`) |
| `**.example.com` | any depth subdomain including `example.com` itself |

The kernel implements just these three forms. Future RFC may add CIDR or
port restrictions; v1 is hostname-only.

### Reserved tokens

- `tool_grants` containing the literal string `"*"` means "all tools".
  Used by tests and bootstrap; production policies should enumerate.
- `model_grants` containing `"*"` means "all models".
- `net_grants` containing `"*"` means "all hosts" (aliased to
  `**.*` internally).
- `fs_grants` does not have a `"*"` token; an empty tuple is "no fs
  access" and a tuple with `FsGrant("/", "rw")` is "all fs". Explicit by
  design — root access shouldn't be a one-character typo.

## 3. Derivation rules

A child capability `C` derived from parent `P`:

```
C.tool_grants  ⊆ P.tool_grants    (treating "*" as universal)
C.model_grants ⊆ P.model_grants
C.net_grants   ⊆ P.net_grants_in_glob_sense

For each child fs grant fc ∈ C.fs_grants:
    ∃ fp ∈ P.fs_grants such that:
        fc.prefix.startswith(fp.prefix)   AND
        mode_subset(fc.mode, fp.mode)

C.sub_agent = True ⇒ P.sub_agent = True
```

Where:
- `mode_subset("r",  "r"|"rw") = True`
- `mode_subset("rw", "rw")     = True`
- `mode_subset("rw", "r")      = False`

Net glob subset (v1, conservative): the child's pattern set must be a
**string subset** of the parent's. We explicitly do NOT compute glob
subsumption (e.g. accepting child `"api.example.com"` when parent has
`"*.example.com"`) — too easy to get wrong. If the parent grants
`"*.example.com"`, the child must inherit that exact glob string. Future
RFC may relax this with formal subset reasoning.

The strict-string subset is conservative-safe: it never accidentally
grants more than intended. If a parent wants to give a child only
`api.example.com`, it must hold `api.example.com` itself.

Violating any rule raises `CapabilityDerivationError` (RPC code -32111).

## 4. Storage

A new SQLite table in the existing `kernel.db` (RFC 0003 §4). Schema
version bumps from **1 to 2**. The migration is purely additive: the
new table appears via `CREATE TABLE IF NOT EXISTS`; existing tables are
untouched.

### DDL

```sql
CREATE TABLE IF NOT EXISTS agent_capabilities (
    cap_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_cap_id INTEGER,
    pid           INTEGER NOT NULL UNIQUE,        -- one cap per agent
    tool_grants   TEXT NOT NULL,                  -- canonical JSON array
    fs_grants     TEXT NOT NULL,                  -- canonical JSON array
    net_grants    TEXT NOT NULL,                  -- canonical JSON array
    model_grants  TEXT NOT NULL,                  -- canonical JSON array
    sub_agent     INTEGER NOT NULL CHECK(sub_agent IN (0,1)),
    created_at    REAL NOT NULL,
    FOREIGN KEY (pid) REFERENCES agent_processes(pid),
    FOREIGN KEY (parent_cap_id) REFERENCES agent_capabilities(cap_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_capabilities_pid
    ON agent_capabilities(pid);
```

`UNIQUE(pid)` enforces 1:1 between AgentProcess and Capability. A second
`cap.create` for the same pid raises `KernelError` with code
`KERNEL_CAPABILITY_EXISTS` (-32112).

### Schema migration

On daemon start, `init_schema()` reads the recorded version and:

| Recorded | Code expects | Behaviour |
|---|---|---|
| missing | 2 | Create all tables, write `schema_version=2`. |
| 1 | 2 | All v2 tables created via IF NOT EXISTS. UPDATE the stamp to 2. |
| 2 | 2 | No-op. |
| > 2 | 2 | Raise `SchemaMismatch` — refuse to start (operator must downgrade kernel.db or upgrade code). |

Forward migration from 1→2 is safe because v2 only adds tables.

## 5. RPC surface

All under the `kernel.cap.*` namespace.

```
kernel.cap.create
  params:  { pid, tool_grants?, fs_grants?, net_grants?, model_grants?, sub_agent? }
  result:  { cap_id, pid }

kernel.cap.derive
  params:  { parent_pid, child_pid, tool_grants?, fs_grants?, net_grants?, model_grants?, sub_agent? }
  result:  { cap_id, pid: child_pid, parent_cap_id }

kernel.cap.get
  params:  { pid }
  result:  Capability JSON

kernel.cap.check_tool
  params:  { pid, tool }
  result:  { allowed: bool }

kernel.cap.check_fs
  params:  { pid, path, mode }      # mode ∈ {"r", "rw"}
  result:  { allowed: bool }

kernel.cap.check_net
  params:  { pid, host }
  result:  { allowed: bool }

kernel.cap.check_model
  params:  { pid, model }
  result:  { allowed: bool }
```

Defaults for omitted grant arguments:
- `tool_grants`, `net_grants`, `model_grants` default to empty (deny all).
- `fs_grants` defaults to empty.
- `sub_agent` defaults to `False`.

This is **default-deny**: an agent created without explicit grants can
do nothing tool/path/net/model-wise.

### Error codes

| Code | Name | Meaning |
|---|---|---|
| -32111 | `kernel_cap_derivation_invalid` | Child violates ⊆ parent |
| -32112 | `kernel_cap_already_exists` | Second create for same pid |
| -32113 | `kernel_cap_unknown_pid` | Pid has no row in agent_capabilities |
| -32114 | `kernel_cap_invalid_grant` | Bad path / glob / mode value |

## 6. Enforcement (deferred)

The kernel **stores and checks**; it does not enforce. RFC F-4 (subprocess
agent runner) and a future patch to the tool dispatcher are responsible
for calling `kernel.cap.check_*` before dispatch and refusing tool
execution on `allowed=false`. This is a deliberate split:

- The kernel must outlive any one tool layer; if enforcement lived in
  the kernel directly, capability semantics would couple to tool
  registry internals.
- Phasing the enforcement separately means existing tests / users see no
  behaviour change until F-4 + dispatcher patches land. Backwards
  compatibility is preserved.

A `kernel.cap.check_*` RPC always returns a value, even for pids without
a capability row: **default deny** (returns `allowed: false`). A
supervisor that hasn't called `kernel.cap.create` yet for an agent will
correctly see all checks fail closed, which is the safer error mode.

## 7. Backwards compatibility

- Schema bump is additive (1 → 2). Existing kernel.db files will
  forward-migrate transparently on next daemon start with the new code.
- No existing module is modified. The new RPC methods register
  alongside the existing `kernel.agent.*` and `kernel.events.*`
  surfaces.
- `kernel.agent.create` does **not** auto-create a default capability.
  Old test code that creates agents without caps continues to work; the
  kernel remains permissive at the storage layer (it just stores nothing
  about caps for that pid). When the supervisor later asks `check_*`,
  it gets `allowed: false`, which the supervisor handles per its policy
  (deny-and-log, or log-and-permit during a soft rollout).

## 8. Open questions

1. **Should derivation reject empty grants?** A child with all empty
   sets is functionally inert — no tools, no fs, no net, no models, no
   sub-agents. Not useful, but also not unsafe. Current draft: allow.
2. **Glob subset reasoning.** Conservative string-equality is annoying
   when a parent has `*.example.com` and a child wants only
   `api.example.com`. Lean toward adding "child glob is more specific
   than some parent glob" check in v2. Out of scope for this RFC.
3. **Capability events.** Should `kernel.cap.create` / `derive` emit
   `kernel.capability.created` events into the event log? Currently no.
   Pro: complete audit trail. Con: doubles write volume for what is
   already a low-frequency op. Lean yes; will add unless review pushes
   back.

## 9. Acceptance criteria

A PR claiming this RFC must:

1. Schema migrates v1 → v2 forward; existing v1 kernel.db opens cleanly
   and stamps to v2.
2. `kernel.cap.create` / `derive` / `get` round-trip. UNIQUE(pid)
   enforced.
3. Derivation rejects every illegal child case (extra tool, broader fs,
   broader net, broader model, sub_agent without parent's, fs mode upgrade,
   path outside parent).
4. `kernel.cap.check_*` returns false for unknown pids (default deny),
   true for matching grants, false for non-matching.
5. Glob matching covers exact, single-level wildcard, multi-level
   wildcard.
6. Path matching covers prefix + mode subset.
7. No file outside `kernel/` and `docs/RFC/` is modified.
