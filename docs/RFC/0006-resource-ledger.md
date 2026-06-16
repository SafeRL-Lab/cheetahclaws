# Design Note: ResourceLedger — per-agent quota and accounting

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Tracks roadmap:** [`0002-daemon-foundation-roadmap.md`](./0002-daemon-foundation-roadmap.md) (Phase 2 — quota)
- **Builds on:** [`0003-agent-process-and-event-log.md`](./0003-agent-process-and-event-log.md)
- **Sibling RFC:** [`0005-capability-model.md`](./0005-capability-model.md)

This RFC defines the **third** Phase-2 invariant: **every action is
bounded**. Each agent's lifetime budget across multiple resource
dimensions is tracked atomically; charges that exceed grants are
recorded but flagged. Per-agent resource accounting is the foundation
for RFC 0007's scheduler (which uses the ledger to make placement and
suspension decisions) and RFC 0014's billing (deferred).

This RFC ships **purely additive** code; nothing existing is modified.
Schema version bumps 1 → 2 (combined with RFC 0005's bump — a single
schema upgrade covers both).

## 1. Goals

1. **Multi-dimensional.** Tokens, USD cost, CPU seconds, wall seconds,
   tool calls, file write bytes — distinct counters, distinct grants.
2. **Atomic.** A `charge()` is one SQLite transaction; concurrent
   charges from multiple supervisor threads cannot lose updates.
3. **Recorded, then signalled.** A charge that exceeds the grant still
   records the actual usage; the response signals `over_limit=true` so
   the caller can suspend the agent. Real-world LLM API calls have
   already happened — refusing to record would lie about consumption.
4. **Composable.** RFC 0007's scheduler reads the ledger via cheap
   `check()` calls before dispatching a tool. RFC F-4 charges into it
   from the supervisor.

## 2. Data model

### Standard dimensions

The kernel treats `dim` as opaque. The supervisor and runtime decide
which dimensions to use; standard names are documented but not enforced
(custom dims allowed for trading bots, research lab, etc.).

| dim | Unit | Source | Typical grant |
|---|---|---|---|
| `tokens` | int | model API response | 100k–10M |
| `cost_micro` | int (USD micro-cents = 10⁻⁶ USD) | model API pricing | 10_000 = $0.01 to 50_000_000 = $50 |
| `cpu_s` | int seconds | RLIMIT_CPU readout | 60–3600 |
| `wall_s` | int seconds | wall-clock killer | 300–86400 |
| `tool_calls` | int | runtime increment | 50–10000 |
| `fs_w_bytes` | int | tool write hook | 10MB–10GB |

`cost_micro` uses integer micro-cents to keep all dimensions integer;
$1.50 = `1_500_000`. Avoids float rounding in ledgers.

### `Ledger`

```python
@dataclass(frozen=True)
class LedgerEntry:
    pid:         int
    dim:         str
    used:        int
    granted:     int      # equal to hard_limit; renamed for clarity
    hard_limit:  int      # alias of granted; explicit so RFC 0007 reads it
    warn_at:     float    # 0.0–1.0; triggers a warn event at used/granted ≥ warn_at
```

```python
@dataclass(frozen=True)
class Ledger:
    pid:     int
    entries: tuple[LedgerEntry, ...]  # one per dim; order = creation order
```

### Schema (DDL)

Lives in the same `kernel.db` as RFC 0003 / 0005. Schema version 2
includes both this table and `agent_capabilities` from RFC 0005.

```sql
CREATE TABLE IF NOT EXISTS agent_ledgers (
    pid          INTEGER NOT NULL,
    dim          TEXT    NOT NULL,
    used         INTEGER NOT NULL DEFAULT 0,
    granted      INTEGER NOT NULL,
    hard_limit   INTEGER NOT NULL,
    warn_at      REAL    NOT NULL DEFAULT 0.8,
    created_at   REAL    NOT NULL,
    updated_at   REAL    NOT NULL,
    PRIMARY KEY (pid, dim),
    FOREIGN KEY (pid) REFERENCES agent_processes(pid)
);

CREATE INDEX IF NOT EXISTS idx_agent_ledgers_pid
    ON agent_ledgers(pid);
```

The composite PK `(pid, dim)` is the natural key. No surrogate id —
the supervisor will never reference a ledger row outside `(pid, dim)`.

## 3. Operations

### `create(pid, grants: dict[str, int], warn_at: float = 0.8)`

Creates one row per dimension with `used=0`. Calling `create` twice for
the same pid with overlapping dims raises `LedgerExists`. A subsequent
call with disjoint dims is allowed (additive).

```python
ledger.create(pid, {"tokens": 200_000, "cost_micro": 2_000_000})
```

### `charge(pid, dim, amount) -> ChargeResult`

Atomically:

```
BEGIN IMMEDIATE
  used' = used + amount
  if used' > hard_limit: over_limit = True
  else:                   over_limit = False
  UPDATE agent_ledgers SET used = used', updated_at = now WHERE pid = ? AND dim = ?
COMMIT
```

Returns:

```python
@dataclass(frozen=True)
class ChargeResult:
    pid:          int
    dim:          str
    amount:       int        # what was charged (= input)
    used:         int        # post-charge total
    granted:      int
    over_limit:   bool       # used > hard_limit
    warned:       bool       # crossed warn_at threshold this call
    first_breach: bool       # transitioned from <hard_limit to >hard_limit
```

`first_breach=true` is the signal the supervisor reads to suspend the
agent on its first over-budget event (rather than every subsequent
charge). Subsequent charges keep `over_limit=true` but
`first_breach=false`.

`charge` on an unknown `(pid, dim)` raises `LedgerUnknownDim`. The
supervisor must `create` first.

`amount` must be a non-negative int. Negative charges are rejected;
refunds are a separate operation (`refund`, see §3.5).

### `check(pid, dim, amount) -> CheckResult`

Read-only; no write lock taken. Returns:

```python
@dataclass(frozen=True)
class CheckResult:
    pid:          int
    dim:          str
    used:         int
    granted:      int
    would_use:    int        # used + amount
    would_exceed: bool       # would_use > hard_limit
```

Used by the scheduler before dispatch: "if I run this tool that's likely
to consume X tokens, would the agent exceed?" Cheap and lock-free.

### `get(pid) -> Ledger`

Returns all dimensions for the agent. Empty tuple if the agent has no
ledger rows yet.

### `list_breached(limit=100) -> list[LedgerEntry]`

Returns currently-over-limit entries across all agents — the scheduler's
hint for "which agents to suspend now". Indexed read.

### `refund(pid, dim, amount)`

Rare. Decrements `used` by `amount`. Only for cases where a charge was
recorded and then rolled back (e.g. the model API call that produced
the tokens later failed and was retried). Used must not go below zero;
attempting to refund more than was used raises `LedgerInvalidRefund`.

### `update_grant(pid, dim, new_grant)`

Operator escape hatch. Adjust the granted/hard_limit for a running
agent. Useful when the supervisor wants to extend a stuck research
agent's budget. Emits a `kernel.ledger.grant_updated` event.

## 4. RPC surface

```
kernel.ledger.create
  params:  { pid, grants, warn_at? }      # grants = {dim: int}
  result:  { pid, dims: [str] }

kernel.ledger.charge
  params:  { pid, dim, amount }
  result:  ChargeResult JSON

kernel.ledger.check
  params:  { pid, dim, amount }
  result:  CheckResult JSON

kernel.ledger.get
  params:  { pid }
  result:  Ledger JSON

kernel.ledger.list_breached
  params:  { limit?=100 }
  result:  { entries: [LedgerEntry] }

kernel.ledger.refund
  params:  { pid, dim, amount }
  result:  { pid, dim, used, granted }

kernel.ledger.update_grant
  params:  { pid, dim, new_grant }
  result:  { pid, dim, granted, used }
```

### Error codes

| Code | Name | Meaning |
|---|---|---|
| -32121 | `kernel_ledger_unknown_dim` | (pid, dim) not present |
| -32122 | `kernel_ledger_already_exists` | dim already created for this pid |
| -32123 | `kernel_ledger_invalid_amount` | amount negative or non-int |
| -32124 | `kernel_ledger_invalid_refund` | would push used below zero |
| -32125 | `kernel_ledger_invalid_warn_at` | warn_at outside [0, 1] |

## 5. Concurrency

`charge`, `refund`, `update_grant` take the kernel write lock and run
inside one SQLite `BEGIN IMMEDIATE` transaction. `check`, `get`,
`list_breached` are read-only and lock-free (rely on SQLite WAL
snapshot isolation).

Throughput: bounded by SQLite WAL fsync (~1ms on SSD). For control-plane
charging this is plenty. If a future bottleneck arises, the
optimisation is batched charges (`charge_many({dim: amount, ...})` in
one transaction) — schema-compatible with this RFC.

## 6. Backwards compatibility

- Schema bump 1 → 2 is shared with RFC 0005. Forward migration is
  additive (CREATE IF NOT EXISTS).
- No existing module modified. New RPC methods register alongside
  existing `kernel.*` namespaces.
- The kernel does not auto-create ledger rows on `kernel.agent.create`.
  Old code path that creates agents without ledgers continues to work;
  `charge` calls before `create` raise `LedgerUnknownDim` rather than
  silently no-op.

## 7. Enforcement (deferred)

This RFC defines the storage and the API. Enforcement decisions —
"when first_breach=true, what does the kernel do?" — live in RFC 0007
(Scheduler). The kernel itself only signals; it does not suspend or
terminate agents. The supervisor (F-4) and scheduler (0007) read the
signal and act.

## 8. Open questions

1. **Should `charge` emit an event into the event log?** Pro: full
   audit. Con: a busy agent could emit thousands of charge events
   per turn. Lean: emit only on `first_breach=true` and on `warned`
   transitions; routine charges stay quiet.
2. **Multi-dim atomicity.** Should `charge` accept a dict of dim→amount
   so a single tool call updates tokens AND cost AND tool_calls
   together? Currently no; each is a separate call. Easy to add later
   without breaking the single-dim API.
3. **Grant decay.** Operator might want "charge 10% of unused budget
   to a parent on agent termination". Out of scope; can be implemented
   as a refund + parent's update_grant from the supervisor.

## 9. Acceptance criteria

A PR claiming this RFC must:

1. Schema migrates v1 → v2; sibling table from RFC 0005 is also present.
2. `create` / `charge` / `check` / `get` / `refund` round-trip.
3. `charge` is atomic under concurrent calls (no lost updates in a
   100-thread fuzz test).
4. `over_limit`, `first_breach`, `warned` are correctly classified
   across the standard transition cases (under, crossing, over,
   over-again).
5. `list_breached` returns only currently-over-limit rows.
6. RPC error codes match table in §4.
7. No file outside `kernel/` and `docs/RFC/` is modified.
