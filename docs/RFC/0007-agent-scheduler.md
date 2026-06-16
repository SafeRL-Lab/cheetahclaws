# Design Note: AgentScheduler — unified ready queue

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Tracks roadmap:** [`0002-daemon-foundation-roadmap.md`](./0002-daemon-foundation-roadmap.md) (Phase 3 — scheduling)
- **Builds on:** [`0003-agent-process-and-event-log.md`](./0003-agent-process-and-event-log.md), [`0006-resource-ledger.md`](./0006-resource-ledger.md)
- **Sibling RFCs:** [`0009-agent-mailbox.md`](./0009-agent-mailbox.md) (planned), [`0010-agent-registry.md`](./0010-agent-registry.md) (planned)

This RFC introduces the **ready queue** — the single point where every
"this agent should run now" decision is materialised, regardless of how
it was triggered. Today the codebase has at least four different
scheduling surfaces (`monitor/scheduler.py` cron, `research/lab/`
priority queue, the proactive watcher, and ThreadPoolExecutor in
`multi_agent/`). This RFC unifies the *queue substrate* under one
abstraction; the per-flavour triggering logic stays where it is and
calls into the kernel.

The kernel **stores and dispatches**; **execution** happens in the
supervisor (RFC F-4 subprocess agent runner). The scheduler does not
drive agent state-machine transitions itself — the supervisor calls
`kernel.agent.transition(pid, RUNNING)` after claiming a queue entry.

This RFC ships **purely additive** code. Existing scheduling surfaces
keep working until a separate, optional patch wires them through the
kernel. With `--enable-kernel` absent, no scheduler tables are created.

## 1. Goals & non-goals

**Goals:**

1. **One queue.** All "ready to run" entries land in one table, ordered
   by priority + runnable_at. Triggers (cron, event, manual, resume)
   become metadata, not separate code paths.
2. **Atomic claim.** A worker thread claims the next entry in one
   transaction; concurrent claims from N workers cannot return the same
   entry to two of them.
3. **Priority + deadline + delayed run.** Higher integer = sooner.
   Entries with `runnable_at > now` are invisible to claim until time
   catches up. Entries with `deadline < now` are swept to `expired` by
   `gc_expired`.
4. **Decoupled from agent state machine.** The scheduler queue has its
   own state machine (`queued / dispatched / completed / expired /
   cancelled`). The agent's state machine (RFC 0003) is unchanged.
   Supervisor coordinates the two.
5. **Per-agent quotas honoured at admission.** RFC 0006's ledger is
   read at `claim` time: an agent whose tokens / cost / wall budget is
   already breached gets skipped (its entry stays queued). Operator
   resumes by `update_grant` or `cancel`. The scheduler does not write
   to the ledger; only reads.

**Non-goals (v1):**

- **Cron triggers.** A wrapper module or RFC 0007.1 will add a
  cron-spec→`enqueue` runner. Out of scope here.
- **Event triggers.** Same — a thin wrapper that subscribes to the
  EventBus and enqueues on match. Out of scope here.
- **Preemption.** A claimed entry runs to completion; the scheduler
  does not yank dispatched entries. The supervisor can still kill the
  child via the sandbox's wall_seconds enforcer.
- **Fair-share.** Strict priority + FIFO-within-priority, no DRF /
  CFS-style accounting. Future RFC.
- **Dependency graphs.** No `wait_for_sched_id` field. If A must
  follow B, the orchestrator enqueues A only after B completes.
- **Cluster.** Single-daemon. RFC 0015 is years away.

## 2. Data model

### `ScheduleSpec` (input to `enqueue`)

```python
@dataclass(frozen=True)
class ScheduleSpec:
    pid:           int                    # owning agent
    priority:      int = 0                # higher → sooner
    runnable_at:   float = 0.0            # epoch seconds; 0 = "now"
    deadline:      float | None = None    # epoch seconds; None = no deadline
    trigger:       str = "manual"         # 'manual' | 'cron' | 'event' | 'resume' | custom
    payload:       dict = field(default_factory=dict)   # opaque to kernel
```

### `ReadyEntry` (rows in `agent_schedule_queue`)

```python
@dataclass(frozen=True)
class ReadyEntry:
    sched_id:       int
    pid:            int
    priority:       int
    runnable_at:    float
    deadline:       float | None
    trigger:        str
    payload:        dict
    state:          str            # see state machine §3
    worker_id:      str | None     # supervisor / worker that claimed it
    created_at:     float
    dispatched_at:  float | None
    completed_at:   float | None
    exit_kind:      str | None
```

`payload` is opaque JSON. The supervisor uses it to carry "what tool
to run", "which step in a multi-stage agent", etc. The kernel does
not inspect.

## 3. Queue state machine

```
              ┌────────────┐
   enqueue    │            │
   ─────────► │  queued     │
              │            │
              └────┬─────┬─┘
                   │     │
                   │     │ gc_expired (deadline < now)
                   │     ▼
              claim│   ┌──────────┐
                   │   │ expired  │   (terminal)
                   │   └──────────┘
                   │
                   │     ┌────────────┐
                   ▼     │            │
              ┌────────┐ │ cancelled  │   (terminal)
              │dispatch│ │            │
              │  ed    │ └────────────┘
              │        │      ▲
              └───┬────┘      │
                  │           │ cancel  (only from queued)
                  │ complete  │
                  ▼           │
              ┌────────┐      │
              │complete│      │
              │   d    │ ─────┘   (no transition; just shows the cancel-from-queued edge)
              └────────┘
```

Allowed transitions:

| Trigger        | From       | To         | Notes |
|---|---|---|---|
| `claim`        | queued     | dispatched | Atomic; serialised by kernel write lock |
| `gc_expired`   | queued     | expired    | When deadline < now |
| `cancel`       | queued     | cancelled  | Cannot cancel a dispatched entry; supervisor uses `complete(exit_kind=cancelled)` instead |
| `complete`     | dispatched | completed  | Records exit_kind |

`completed`, `expired`, `cancelled` are terminal. Any further mutation
attempt raises `IllegalQueueTransition` (-32131).

Note: cancellation of a *dispatched* entry is NOT a kernel concern —
the supervisor that claimed it must cooperate (kill the subprocess,
then `complete(exit_kind="cancelled")`). The kernel cannot reliably
yank work from a running supervisor.

## 4. `claim` semantics

The supervisor's worker thread calls:

```python
entries = scheduler.claim(worker_id="sup-7", max_n=1, now=time.time())
```

Inside one SQLite transaction:

```sql
BEGIN IMMEDIATE;

SELECT * FROM agent_schedule_queue
WHERE state = 'queued'
  AND runnable_at <= :now
  AND (deadline IS NULL OR deadline > :now)
ORDER BY priority DESC, runnable_at ASC, sched_id ASC
LIMIT :max_n;

-- For each row r in the SELECT:
UPDATE agent_schedule_queue
   SET state = 'dispatched',
       worker_id = :worker_id,
       dispatched_at = :now
 WHERE sched_id = r.sched_id;

COMMIT;
```

Properties:

- **Atomic across workers.** `BEGIN IMMEDIATE` serialises writers; two
  workers cannot return the same `sched_id`.
- **Honours runnable_at.** Cron-style delayed entries don't appear
  before their time.
- **Skips expired-but-not-swept.** If `deadline < now`, the entry is
  invisible to claim. `gc_expired` is the explicit cleanup.
- **Stable order.** Within priority + runnable_at, order is by
  `sched_id` ASC (FIFO within priority).

### Optional: ledger admission filter

`claim` accepts `admission_check=True` (default). When set, the kernel
joins to `agent_ledgers` and skips any queued entry whose owning agent
has any `dim` with `used > hard_limit`. Operator can disable for
debugging via `admission_check=False`.

This is an **advisory** filter; it does not modify the queue. The entry
stays `queued` until either:
- the ledger is updated (`update_grant`) and the next `claim` picks it
  up, OR
- the entry is `cancel`ed by an operator, OR
- `gc_expired` sweeps it (deadline reached without dispatch).

The filter implementation is one extra `WHERE NOT EXISTS (...)` clause;
no extra round trip.

## 5. RPC surface

```
kernel.sched.enqueue
  params: { pid, priority?, runnable_at?, deadline?, trigger?, payload? }
  result: { sched_id }

kernel.sched.claim
  params: { worker_id, max_n?=1, now?=<server>, admission_check?=true }
  result: { entries: [ReadyEntry] }

kernel.sched.complete
  params: { sched_id, exit_kind }      # 'completed' | 'cancelled' | 'failed' | 'crashed'
  result: { sched_id, state: 'completed', prev_state }

kernel.sched.cancel
  params: { sched_id }
  result: { sched_id, state: 'cancelled' }

kernel.sched.get
  params: { sched_id }
  result: ReadyEntry

kernel.sched.list
  params: { state?, pid?, limit?=100, offset?=0 }
  result: { entries: [ReadyEntry], total }

kernel.sched.gc_expired
  params: { now?=<server> }
  result: { swept: int }
```

### Error codes

| Code | Name | Meaning |
|---|---|---|
| -32131 | `kernel_sched_illegal_transition` | e.g. cancel on dispatched, complete on queued |
| -32132 | `kernel_sched_unknown_id` | sched_id not present |
| -32133 | `kernel_sched_invalid_payload` | bad priority / trigger / amount |
| -32134 | `kernel_sched_admission_denied` | only emitted from a future `force_claim` mode |
| -32135 | `kernel_sched_invalid_state_filter` | state filter not in allowed set |

## 6. Storage

Schema version bumps **2 → 3**. Additive: one new table + indexes.

```sql
CREATE TABLE IF NOT EXISTS agent_schedule_queue (
    sched_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    pid           INTEGER NOT NULL,
    priority      INTEGER NOT NULL DEFAULT 0,
    runnable_at   REAL    NOT NULL DEFAULT 0,
    deadline      REAL,
    trigger       TEXT    NOT NULL DEFAULT 'manual',
    payload       TEXT    NOT NULL DEFAULT '{}',
    state         TEXT    NOT NULL DEFAULT 'queued'
                  CHECK(state IN ('queued','dispatched','completed','expired','cancelled')),
    worker_id     TEXT,
    created_at    REAL    NOT NULL,
    dispatched_at REAL,
    completed_at  REAL,
    exit_kind     TEXT,
    FOREIGN KEY (pid) REFERENCES agent_processes(pid)
);

-- Composite index for the `claim` hot-path SELECT.
CREATE INDEX IF NOT EXISTS idx_sched_queue_pickable
    ON agent_schedule_queue (state, priority DESC, runnable_at, sched_id);

CREATE INDEX IF NOT EXISTS idx_sched_queue_pid
    ON agent_schedule_queue (pid);

CREATE INDEX IF NOT EXISTS idx_sched_queue_deadline
    ON agent_schedule_queue (deadline)
    WHERE deadline IS NOT NULL;
```

Forward migration: existing v2 kernel.db transparently gains the new
table on next daemon start. Existing rows untouched.

## 7. Backwards compatibility

- Schema bump v2 → v3 is additive. Existing kernel.db files migrate
  forward.
- No existing module modified.
- No agent runtime is forced to use the queue. The scheduler is a tool
  the supervisor (F-4) and future cron/event wrappers can adopt; until
  they do, the existing scheduling surfaces in `monitor/scheduler.py`,
  `research/lab/`, etc., keep working unchanged.
- `kernel.agent.create` does not auto-enqueue.

## 8. Concurrency contract

- All writes (`enqueue`, `claim`, `complete`, `cancel`, `gc_expired`)
  serialise on the kernel write lock (shared with the other Phase-2
  stores; see CapabilityStore docstring for why a single shared lock is
  mandatory under Python sqlite3's implicit-transaction model).
- Reads (`get`, `list`) are lock-free under SQLite WAL.
- `claim`'s SELECT + UPDATE happens inside one `BEGIN IMMEDIATE`
  transaction. SQLite serialises against any concurrent claim or
  enqueue, so duplicate dispatch is structurally impossible.

## 9. Open questions

1. **`claim` greedy vs paced.** A worker that asks for `max_n=10`
   commits all 10 in one transaction. If the supervisor crashes mid-way
   through executing them, all 10 are stuck in `dispatched` and need
   `gc_dispatched` (currently absent — only `gc_expired` for queued
   entries with deadlines). Lean: add `gc_dispatched(stale_seconds)` in
   v1 too. Will firm up before merge.
2. **Should `enqueue` validate `pid` belongs to a live agent?** Today
   any pid passes if it exists in `agent_processes` — including DEAD
   ones. Lean: reject if state == DEAD; supervisor would never resume
   a dead agent anyway. **Yes, will reject DEAD.**
3. **Idempotent enqueue.** A common pattern in cron triggers is "do
   not enqueue if there's already a queued entry for this pid+trigger".
   The kernel could expose `enqueue_unique(spec, dedupe_by=("pid",
   "trigger"))`. Out of scope for this RFC; orchestrators can do their
   own dedupe in user code.

## 10. Acceptance criteria

A PR claiming this RFC must:

1. Schema migrates v2 → v3 forward; existing v2 kernel.db opens cleanly
   and stamps to v3.
2. `enqueue` / `claim` / `complete` / `cancel` / `get` / `list` round
   trip.
3. Two threads each calling `claim(max_n=1)` against a queue of N
   entries collectively retrieve exactly N distinct sched_ids — no
   duplicates, no losses.
4. `runnable_at` in the future is invisible to `claim` until the time
   has passed.
5. `deadline` past: invisible to `claim`; swept by `gc_expired`.
6. Priority order: higher integer wins; ties broken by `runnable_at`
   then `sched_id`.
7. Admission filter skips entries whose owner has any over-limit
   ledger row, but does not modify the queue state.
8. `kernel.sched.*` RPCs work end-to-end through the daemon.
9. No file outside `kernel/`, `tests/`, and `docs/RFC/` is modified.
