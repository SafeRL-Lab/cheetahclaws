# Design Note: AgentProcess & EventLog — kernel-level lifecycle and WAL

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Tracks roadmap:** [`0002-daemon-foundation-roadmap.md`](./0002-daemon-foundation-roadmap.md) (Phase 1 — fault domain)
- **Builds on:** [`0001-daemon-design-note.md`](./0001-daemon-design-note.md) (transport, auth, originator)

This RFC defines two of the three Phase-1 invariants needed before
cheetahclaws can be called an agent OS:

1. **AgentProcess** — every running agent is a first-class kernel object
   with a stable PID, an explicit state machine, and durable identity.
2. **EventLog (WAL)** — every state transition and every observable
   action is appended to a single, monotonic, durable event log so the
   system can be replayed, audited, and recovered after a crash.

Phase-1 invariant #3 (sandbox / RLIMIT) is intentionally deferred to a
companion RFC (0008). This note's scope is **identity, lifecycle, and
durability** — nothing about isolation, scheduling, capabilities, or
quotas, all of which compose on top of the contracts defined here.

The RFC ships behind a single opt-in flag (`cheetahclaws serve
--enable-kernel`). With the flag absent, daemon behaviour is byte-for-byte
unchanged; existing users see nothing new.

## 1. AgentProcess

### Data model

```python
@dataclass(frozen=True)
class AgentProcess:
    pid:           AgentPID            # int, monotonic, persistent
    parent_pid:    AgentPID | None
    name:          str                 # e.g. "researcher-alice"
    template:      str                 # e.g. "research/surveyor", "skill/refactor"
    state:         AgentState          # READY|RUNNING|WAITING|SUSPENDED|DEAD
    state_reason:  str | None          # why the state machine moved last
    created_at:    float               # epoch seconds
    updated_at:    float
    started_at:    float | None        # first transition into RUNNING
    ended_at:      float | None        # transition into DEAD
    exit_kind:     str | None          # 'completed'|'cancelled'|'failed'|'crashed'
    exit_detail:   dict | None         # opaque JSON
    metadata:      dict                # opaque, agent_runner / scheduler hints
    last_event_id: int                 # high-water mark in agent_events
```

`pid` is a positive integer minted by SQLite via
`INTEGER PRIMARY KEY AUTOINCREMENT`, which guarantees monotonicity and
non-reuse across the lifetime of the database (rowid high-water mark is
held in `sqlite_sequence`). It is **never reused**, even after a DEAD
agent is purged. The intent is to make every reference (parent_pid,
event.pid, mailbox addresses, capability handles) unambiguous across
the lifetime of the data directory.

`name` is human-readable but not unique. Two agents may share a name
(e.g. retried instances of the same template). Uniqueness for service
discovery is RFC 0011's problem (AgentRegistry), not this RFC's.

`template` is a free-form opaque string. The kernel does not interpret
it; it is used by tooling (CLI, dashboards) to group agents and by
RFC 0007 to apply scheduler defaults.

### State machine

```
                   ┌─────────────┐
            create │             │
       ───────────►│   READY     │
                   │             │
                   └──┬───────▲──┘
                      │       │
                start │       │ resume_from_suspended
                      │       │
                   ┌──▼───────┴──┐                ┌──────────┐
                   │             │  suspend       │          │
                   │  RUNNING    ├───────────────►│ SUSPENDED│
                   │             │                │          │
                   └──┬─▲────────┘                └──┬───────┘
                      │ │                            │
                wait  │ │ resume_from_wait           │
                      │ │                            │
                   ┌──▼─┴───────┐                    │
                   │            │      suspend       │
                   │  WAITING   ├────────────────────┤
                   │            │                    │
                   └──┬─────────┘                    │
                      │                              │
                      └──────────────┬───────────────┘
                                     │
                                     │ terminate / cancel / fail / crash
                                     ▼
                               ┌──────────┐
                               │   DEAD   │  (terminal)
                               └──────────┘
```

Allowed transitions (`prev → next`):

| Trigger | Allowed from | New state |
|---|---|---|
| `create` | (none) | READY |
| `start` | READY | RUNNING |
| `wait` | RUNNING | WAITING |
| `resume_from_wait` | WAITING | RUNNING |
| `suspend` | RUNNING, WAITING | SUSPENDED |
| `resume` | SUSPENDED | READY |
| `terminate` | READY, RUNNING, WAITING, SUSPENDED | DEAD |

DEAD is terminal. Any transition attempt from DEAD is a programming error
and raises `IllegalTransition`.

The state machine is **enforced by the kernel** — clients cannot put an
agent into RUNNING without going through READY first, and cannot drive
DEAD → anything. This is the single most important invariant in the RFC,
because every other Phase-1 guarantee (recovery, accounting, blast
radius) is conditioned on it.

### Identity guarantees

- PIDs are monotonic. `next_pid` is held in `kernel_meta(key='next_pid')`
  and incremented under the kernel write lock.
- PIDs are not reused. Even if the agent_processes row is deleted (which
  this RFC does not do — see "purge" below), the PID is consumed.
- `parent_pid`, when set, must reference a row that existed at the time
  of `create`. The kernel does not enforce that the parent is still
  alive; tools may want to filter by `parent_alive` separately.

### Purge (out of scope for this RFC)

The kernel keeps every AgentProcess row forever in v1. A future RFC
(probably alongside RFC 0006 ResourceLedger) defines a purge policy:
DEAD agents older than N days, with no descendants, may be archived to a
cold table. PIDs are still not reused.

## 2. EventLog (WAL)

### Goals

1. **Durable.** Every event is `fsync`ed before the writing call returns
   success. A crash one byte later loses zero events.
2. **Monotonic.** Event IDs are integers from a single counter. Replay is
   "events with id > N", in id order.
3. **Causal.** Each event records the prior event that caused it
   (`causation_id`), so the agent's history is a directed graph, not a
   bag.
4. **Single-writer.** All writes go through one process-wide lock. Read
   concurrency is limited only by SQLite's WAL journal.
5. **Bounded growth.** The log is append-only but `event_id` is the only
   monotonic index; rotation/archival is RFC 0012's problem, not this
   RFC's.

### Data model

```python
@dataclass(frozen=True)
class Event:
    event_id:       int                # monotonic, kernel-wide
    pid:            AgentPID           # owning agent
    ts:             float              # epoch seconds, monotonic-best-effort
    kind:           str                # see "Event kinds" below
    payload:        dict               # canonical-JSON-serialisable
    causation_id:   int | None         # prior event_id
    correlation_id: str | None         # cross-agent trace id (opaque)
```

### Event kinds

The kernel reserves the `kernel.*` prefix. Higher layers may use any
non-conflicting kind.

| Kind | Emitted by | Meaning |
|---|---|---|
| `kernel.process.created` | `agent.create` | New AgentProcess row |
| `kernel.process.transitioned` | `agent.transition` | State changed |
| `kernel.process.terminated` | `agent.terminate` | Reached DEAD |
| `kernel.process.recovered` | daemon startup | Stale RUNNING/WAITING coerced on restart |
| (free-form) | client RPC `events.append` | Anything the agent runtime wants to record |

`payload` for kernel events is documented inline in the RPC method
specifications (§3) and is part of the API contract — tools that read
the event log SHOULD treat unknown payload keys as forward-compatible
extensions.

### Single-writer + transaction shape

Every state transition is one SQLite transaction containing both:

1. The `UPDATE agent_processes SET state=?, ...` row update.
2. The `INSERT INTO agent_events ...` event row.

Both succeed or both fail. The kernel holds a Python `threading.Lock`
around `BEGIN IMMEDIATE` to serialise writes from multiple request
handler threads; reads (e.g. `agent.get`) do not take the lock and rely
on SQLite WAL for snapshot isolation.

After commit, the kernel publishes the same event to the daemon's
in-memory `EventBus` so SSE subscribers see live activity. Crucially:
the bus publish is **after** the durable commit, so a crash between
commit and publish loses no events (the next read of `events.tail`
backfills them).

### Recovery semantics

On daemon startup with `--enable-kernel`:

1. Open `kernel.db`. If absent, run `init_schema()`. If present, verify
   `schema_version`; refuse to start on mismatch (operator must run
   migrations).
2. Read `next_pid` and `next_event_id` from `kernel_meta`. These bound
   the namespace going forward.
3. Find rows where `state IN ('RUNNING', 'WAITING')`. These belong to
   agents that were live when the daemon last shut down (either crashed
   or stopped without quiescing the kernel).
4. For each such row, transition it to **SUSPENDED** with
   `state_reason='daemon_restart'` and emit a
   `kernel.process.recovered` event. The transition is itself a normal
   write (UPDATE + INSERT in one transaction).
5. Resume normal operation.

Rationale for SUSPENDED rather than DEAD:
- DEAD is irreversible; we want operator/tooling to decide whether the
  agent should resume.
- SUSPENDED is a legal source for `resume → READY`, so a future
  scheduler (RFC 0007) or supervisor (RFC F-4) can re-enqueue these
  agents on a policy decision.
- DEAD would lose information: we don't know whether the daemon crashed
  or was cleanly stopped; we only know we don't trust the last
  pre-crash state.

A `--kernel-recovery=mark-dead` opt-out is provided for operators who
prefer the harsher semantics (e.g. running the kernel as a load test).

## 3. RPC surface

All methods are registered under the `kernel.*` namespace. No existing
method name is touched. Any error is reported through the standard
JSON-RPC error envelope; application-specific codes are listed in §3.5.

### 3.1 Process methods

```
kernel.agent.create
  params:  { name, template, parent_pid?, metadata? }
  result:  { pid, state }                   // state == "READY"

kernel.agent.get
  params:  { pid }
  result:  AgentProcess JSON

kernel.agent.list
  params:  { state?, parent_pid?, limit?=100, offset?=0 }
  result:  { agents: [AgentProcess], total }

kernel.agent.transition
  params:  { pid, target_state, reason? }
  result:  { pid, prev_state, state, event_id }

kernel.agent.terminate
  params:  { pid, exit_kind, exit_detail? }
  result:  { pid, prev_state, state: "DEAD", event_id }
```

`agent.transition` is a deliberately raw primitive — it lets clients
drive any legal transition. Higher layers (RFC 0007 scheduler, RFC F-4
supervisor) build their domain verbs (`schedule`, `dispatch`, `pause`)
on top of it. The kernel's job is to enforce the state machine, not to
shape application policy.

### 3.2 Event methods

```
kernel.events.append
  params:  { pid, kind, payload, causation_id?, correlation_id? }
  result:  { event_id }

kernel.events.tail
  params:  { pid?, kind?, since_event_id?=0, limit?=100 }
  result:  { events: [Event], next_cursor }
```

`events.append` is the runtime's hook to record agent-internal activity
(tool calls, model responses, intermediate state). The kernel does not
interpret `kind` outside the reserved `kernel.*` prefix.

`events.tail` is the read primitive. Combined with the SSE `/events`
stream (which carries the same events live), a client can: backfill
since the last seen `event_id` via `events.tail`, then subscribe to the
SSE stream — no events lost, no events duplicated.

### 3.3 Info

```
kernel.info
  params:  {}
  result:  { schema_version, next_pid, next_event_id, agent_count, event_count }
```

For `cheetahclaws daemon status` and operator inspection.

### 3.4 Concurrency contract

- Reads (`get`, `list`, `tail`, `info`) never block.
- Writes (`create`, `transition`, `terminate`, `events.append`)
  serialise on a single kernel write lock. Throughput is limited by
  fsync latency; this is fine for v1 — the kernel is on the control
  plane, not the data plane.

### 3.5 Error codes

| Code | Name | Meaning |
|---|---|---|
| -32101 | `kernel_not_enabled` | `--enable-kernel` was not passed |
| -32102 | `kernel_unknown_pid` | No row for the given pid |
| -32103 | `kernel_illegal_transition` | State-machine rule violated |
| -32104 | `kernel_invalid_payload` | Required field missing or wrong type |
| -32105 | `kernel_schema_mismatch` | DB schema version doesn't match code |

These slot into the existing JSON-RPC error envelope alongside the
RFC 0001 application codes (-32001 not_originator, -32002 unknown_request).

## 4. Storage layout

```
~/.cheetahclaws/
  kernel.db                  # this RFC; PRAGMA journal_mode=WAL
  kernel.db-wal              # WAL sidecar, managed by SQLite
  kernel.db-shm              # shared-memory file, managed by SQLite
```

Why a separate database from `sessions.db` (RFC 0002 F-2):

1. **Lifecycles diverge.** F-2's tables are session-scoped; the kernel
   namespace is daemon-scoped. Mixing them in one file forces every
   migration to consider both.
2. **Reset blast radius.** `rm ~/.cheetahclaws/kernel.db` wipes the
   kernel without touching session history.
3. **F-2 independence.** This RFC must be reviewable and shippable
   without F-2; a shared file would couple the two PRs.

A future RFC can decide whether to merge the files; the kernel API
surface does not depend on file layout.

### Schema (DDL)

```sql
CREATE TABLE IF NOT EXISTS kernel_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_processes (
    pid           INTEGER PRIMARY KEY,
    parent_pid    INTEGER,
    name          TEXT NOT NULL,
    template      TEXT NOT NULL,
    state         TEXT NOT NULL CHECK(state IN ('READY','RUNNING','WAITING','SUSPENDED','DEAD')),
    state_reason  TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    started_at    REAL,
    ended_at      REAL,
    exit_kind     TEXT,
    exit_detail   TEXT,
    metadata      TEXT NOT NULL DEFAULT '{}',
    last_event_id INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (parent_pid) REFERENCES agent_processes(pid)
);

CREATE INDEX IF NOT EXISTS idx_agent_processes_state
    ON agent_processes(state);
CREATE INDEX IF NOT EXISTS idx_agent_processes_parent
    ON agent_processes(parent_pid);

CREATE TABLE IF NOT EXISTS agent_events (
    event_id       INTEGER PRIMARY KEY,
    pid            INTEGER NOT NULL,
    ts             REAL NOT NULL,
    kind           TEXT NOT NULL,
    payload        TEXT NOT NULL,
    causation_id   INTEGER,
    correlation_id TEXT,
    FOREIGN KEY (pid) REFERENCES agent_processes(pid)
);

CREATE INDEX IF NOT EXISTS idx_agent_events_pid
    ON agent_events(pid, event_id);
CREATE INDEX IF NOT EXISTS idx_agent_events_kind
    ON agent_events(kind);
```

`schema_version` is stored in `kernel_meta` (key `'schema_version'`,
initial value `'1'`). Migrations bump it; mismatch is a hard refuse.

## 5. Backwards compatibility & rollout

### Opt-in flag

A single new flag, additive:

```
cheetahclaws serve --enable-kernel
```

When absent (default), no kernel code is imported, no kernel.db is
opened, and no `kernel.*` RPC methods are registered. The daemon
behaviour is byte-for-byte identical to the pre-RFC build.

When present:
- `kernel/` is imported.
- `kernel.db` is opened (created on first run).
- Recovery runs (§2 "Recovery semantics").
- `kernel.*` methods join the RPC registry.
- Kernel events fan out to the existing event bus alongside their
  durable persistence.

### Feature flag surface area

The flag controls a single binary: kernel-on or kernel-off. There are no
sub-flags in this RFC. `--kernel-recovery=suspend|mark-dead` is the only
modifier; default is `suspend` (the safer choice).

### Existing tests

This RFC does not modify any existing module under `daemon/` — the
sole CLI patch is one argparse argument and a conditional call to
`kernel.register_with_daemon()`. Existing tests
(`test_daemon_cli.py`, `test_daemon_system_methods.py`,
`test_daemon_spike.py`, `e2e_daemon_skeleton.py`) must continue to pass
with no changes to their assertions; the flag is off in their setup.

### Default flip

Out of scope for this RFC. RFC 0002's "Phase D" is the right place to
discuss flipping `--enable-kernel` to default-on. This RFC commits only
to the contract, not to the rollout schedule.

## 6. Open questions

1. **Should `kernel.events.append` be rate-limited?**
   The RFC currently leaves this to the runtime. RFC 0006 (ResourceLedger)
   will add per-agent event-count quotas, after which `events.append`
   that exceeds quota returns -32106 (TBD). For v1, no rate limit.

2. **Should `agent.transition` accept a list of expected `prev_state`s?**
   Common pattern: "move to RUNNING only if currently READY". The RFC
   currently relies on the kernel's state-machine table, which already
   rejects the illegal cases. Making `prev_state` an explicit parameter
   gives clients optimistic-concurrency-control for free; the cost is
   one more required field. **Pushback welcome.**

3. **Causation chains across processes.**
   `causation_id` is currently the prior event's `event_id`. Across
   agents (parent spawning child), the causation is "the spawn event in
   the parent". The schema supports this (causation_id is just an int),
   but the RPC contract should probably document this idiom rather than
   leaving it implicit. **Will firm up before merge.**

## 7. What this RFC does **not** do

To keep the diff reviewable and to avoid sneaking unrelated decisions in:

- **No agent runtime integration.** The `agent.run` path in `agent.py`
  does not learn about the kernel in this RFC. F-4 (subprocess agent
  runner) is the right place; that PR will call `kernel.agent.create` /
  `transition` from the supervisor. Until then, the kernel is exercised
  only by tests and (optionally) by hand via `curl`.
- **No capabilities, no quotas, no scheduler.** RFC 0005, 0006, 0007.
- **No mailbox, no AgentFS.** RFC 0009, 0011.
- **No multi-writer.** All kernel writes serialise on one Python lock
  inside one process. The cluster story (RFC 0015) is years away.
- **No agent.run wiring for `kernel.process.transitioned`.** The kernel
  publishes events to the bus, but no existing client subscribes to
  `kernel.*` events yet — by design. The runtime will subscribe in a
  follow-up RFC.

## 8. Acceptance criteria

For a PR claiming to implement this RFC:

1. With `--enable-kernel` absent, `pytest tests/` is green and the daemon
   shows no behavioural diff vs pre-RFC.
2. With `--enable-kernel` set:
   - `kernel.db` is created on first start with `schema_version=1`.
   - `kernel.agent.create` returns a fresh PID and emits a
     `kernel.process.created` event in `agent_events`.
   - `kernel.agent.transition` enforces the state-machine table:
     - READY → RUNNING (start) succeeds.
     - DEAD → RUNNING returns `kernel_illegal_transition`.
   - Killing the daemon mid-run leaves rows in RUNNING; restarting with
     `--enable-kernel` transitions them to SUSPENDED with
     `state_reason='daemon_restart'` and emits
     `kernel.process.recovered`.
   - `kernel.events.tail` after restart returns the recovery events.
   - `kernel.info` reports schema_version=1 and matching counters.
3. A new test file `tests/test_kernel_*` covers:
   - State-machine enforcement (every transition + every illegal one).
   - Event ordering and durability (kill -9 mid-write doesn't tear).
   - Recovery semantics (suspend default, mark-dead opt-out).
   - RPC surface (each method, including error codes).

---

Once accepted, the implementation lives in `kernel/` (new package).
No code outside `kernel/` and `daemon/cli.py` is modified.
