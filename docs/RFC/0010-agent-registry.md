# Design Note: AgentRegistry — name → pid lookup + service discovery

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Tracks roadmap:** [`0002-daemon-foundation-roadmap.md`](./0002-daemon-foundation-roadmap.md) (Phase 3 — service discovery)
- **Builds on:** [`0003-agent-process-and-event-log.md`](./0003-agent-process-and-event-log.md)
- **Sibling RFC:** [`0009-agent-mailbox.md`](./0009-agent-mailbox.md) (ships in same schema bump v3→v4)

This RFC closes Phase 3 with a small but load-bearing piece: a way to
look agents up **by name** rather than only by pid. Without this, every
piece of code that wants to send a message, schedule work, or read a
ledger has to know the pid out-of-band. With it, the bridge can publish
to `/agents/researcher/alice` regardless of which pid is currently
serving that role.

The registry is deliberately small. It is **not** the place to store
business metadata, capabilities, schedules, or budgets — those live in
their respective tables. The registry maps names → pids and stores a
short tag list for filtering.

This RFC ships **purely additive** code. Schema bump v3 → v4 is shared
with RFC 0009.

## 1. Goals & non-goals

**Goals:**

1. **One name → one pid.** A name is unique across the registry.
   Registering the same name twice raises `RegistryNameExists`.
2. **Hierarchical convention.** Names are opaque strings to the
   kernel; by convention they look like
   `/agents/<role>/<instance>` or `/services/<service>`. The kernel
   does not parse the path.
3. **Tag filtering.** Each entry has a small list of tag strings.
   Operator queries `list(tag="research")` to get every research
   agent.
4. **Cleanup on agent terminate.** When an agent's row in
   `agent_processes` moves to DEAD, the supervisor can call
   `unregister_pid(pid)` to drop all registrations for that pid.
   The kernel does not auto-cascade — explicit cleanup keeps the
   registry useful for audit when desired.

**Non-goals (v1):**

- **Health checks.** The registry stores names and pids; it does not
  ping. Health is `kernel.agent.get(pid).state` plus the live event
  bus.
- **Watch / subscribe.** No "notify me when /agents/x changes". RFC
  0009 mailbox + a custom topic does this fine.
- **TTL / lease.** Entries live until explicit unregister or until
  the daemon's data dir is wiped. No automatic expiry.
- **DNS-style A/AAAA records, ports, etc.** Out of scope. The pid is
  enough; the daemon is the one process anything talks to.

## 2. Data model

```python
@dataclass(frozen=True)
class RegistryEntry:
    name:           str
    pid:            int
    tags:           tuple[str, ...]
    metadata:       dict
    registered_at:  float
```

`name` rules:
- Non-empty, ≤ 256 bytes.
- ASCII printable; no `\x00`. Validated.
- Convention is `^/[A-Za-z0-9_/-]+$`; the kernel checks the
  no-NUL/no-control rule but does not enforce path syntax. Operators
  may store names like `bridge:telegram` if they prefer.

`tags`: list of non-empty strings, deduplicated, ≤ 32 entries. Stored
as canonical-JSON in one column.

`metadata`: opaque JSON. Operator hint storage. The kernel stores and
returns it without interpretation.

## 3. Operations

### `register(name, pid, tags=(), metadata={})`

Inserts a new row. Raises `RegistryNameExists` if the name is already
present. Raises `UnknownPid` if the pid is absent from
`agent_processes`. Validates name + tags.

### `unregister(name)`

Deletes the row keyed by name. Idempotent — unregistering a missing
name returns 0 without raising (this matches the cleanup-on-shutdown
use case where the caller doesn't know if it succeeded last time).
Returns the count deleted (0 or 1).

### `unregister_pid(pid)`

Deletes all rows matching `pid`. Used by the supervisor on agent
termination. Returns count.

### `lookup(name) -> RegistryEntry`

Single-row read. Raises `RegistryNotFound` on absence.

### `resolve_pid(name) -> int`

Sugar around `lookup(name).pid`.

### `list(prefix=None, tag=None, limit=100, offset=0) -> tuple[list, int]`

Returns matching entries + total count. `prefix` filters by name
startswith; `tag` filters by tag set membership.

## 4. Storage

Schema v4 (shared with RFC 0009). One table.

```sql
CREATE TABLE IF NOT EXISTS agent_registry (
    name           TEXT    PRIMARY KEY,
    pid            INTEGER NOT NULL,
    tags           TEXT    NOT NULL DEFAULT '[]',
    metadata       TEXT    NOT NULL DEFAULT '{}',
    registered_at  REAL    NOT NULL,
    FOREIGN KEY (pid) REFERENCES agent_processes(pid)
);

CREATE INDEX IF NOT EXISTS idx_agent_registry_pid
    ON agent_registry(pid);
```

## 5. RPC surface

```
kernel.registry.register
  params: { name, pid, tags?, metadata? }
  result: { name, pid }

kernel.registry.unregister
  params: { name }
  result: { name, removed: int }   # 0 or 1

kernel.registry.unregister_pid
  params: { pid }
  result: { pid, removed: int }

kernel.registry.lookup
  params: { name }
  result: RegistryEntry

kernel.registry.list
  params: { prefix?, tag?, limit?, offset? }
  result: { entries: [RegistryEntry], total }
```

### Error codes

| Code | Name | Meaning |
|---|---|---|
| -32151 | `kernel_registry_not_found` | name absent |
| -32152 | `kernel_registry_name_exists` | duplicate register |
| -32153 | `kernel_registry_invalid_name` | NUL or non-printable in name |

## 6. Backwards compatibility

- Schema v3 → v4 forward migration is additive (one new table).
- No existing module modified.

## 7. Open questions

1. **Should `register` overwrite on conflict?** Current draft is
   strict (raises). An `upsert=True` flag is easy to add. **Lean:
   keep strict, force callers to call `unregister` first; surfaces
   stale-state bugs early.**
2. **Multiple names per pid?** Allowed (the table allows duplicate
   pids); useful for an agent that serves both `/services/research`
   and `/agents/alice`. Adding a `UNIQUE(pid)` constraint is one row
   per pid; deliberately not constrained.

## 8. Acceptance criteria

A PR claiming this RFC must:

1. Schema migrates v3 → v4 forward.
2. `register` rejects duplicate names; `lookup` round-trips.
3. `list(prefix="/agents/")` returns only matching names; `list(tag=...)`
   returns only entries containing that tag.
4. `unregister_pid` clears all rows for the given pid.
5. RPC surface works end-to-end through the daemon.
6. No file outside `kernel/`, `tests/`, `docs/RFC/` modified.
