# Design Note: Observability — introspection, metrics, trace, chaos

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Tracks roadmap:** [`0002-daemon-foundation-roadmap.md`](./0002-daemon-foundation-roadmap.md) (Phase 5 — operability)
- **Builds on:** every prior kernel RFC (0003 / 0005 / 0006 / 0007 / 0008 / 0009 / 0010 / 0011)

This RFC closes the kernel's "functional" gap and opens its
"operability" one. Through Phase 4 the kernel can do everything an
agent OS needs to do. Through Phase 5 it becomes **debuggable,
measurable, and resilient under failure** — the qualifications
demanded by the word "production-grade".

Four primitives ship in this RFC:

1. **Introspection** — read-only RPC views that combine state
   across all kernel tables: `kernel.observe.proc(pid)`,
   `kernel.observe.summary()`.
2. **Trace** — walk the `causation_id` chain inside `agent_events`
   to reconstruct the directed graph of agent activity:
   `kernel.observe.trace(event_id, depth)`.
3. **Metrics** — Prometheus exposition format text that the daemon's
   existing `/metrics` endpoint surfaces alongside its current
   payload. No new HTTP endpoint; just additional content.
4. **Chaos primitives** — a small `kernel/chaos.py` module for
   tests, with `ChaosMonkey.kill_random_agent`, `simulate_disk_full`,
   etc. Real chaos suite expansion (network partition, time skew,
   process tree fault) is a v1.1 deliverable; this RFC ships the
   plumbing.

This RFC ships **purely additive** code. **No schema bump** — every
view is computed on demand from existing tables.

## 1. Goals & non-goals

**Goals:**

1. **One agent, full picture.** `kernel.observe.proc(pid)` returns
   one JSON dict combining process row + capability + ledger entries
   + mailbox stats + scheduler queue items + fs object count +
   recent events. The supervisor or a human operator can debug an
   agent without joining six tables.
2. **System rollup.** `kernel.observe.summary()` returns kernel
   uptime, schema version, total agents per state, total events,
   queue depth, mailbox queue total, fs object count.
3. **Causation graph.** `kernel.observe.trace(event_id, depth=10)`
   walks `causation_id` upstream and returns the chain. Useful for
   "why did this agent do this?".
4. **Prometheus alongside existing metrics.** When the kernel is
   active, `/metrics` returns existing daemon metrics + kernel
   metrics. Format conforms to Prometheus text exposition v0.0.4.
5. **Chaos for tests.** A small library with deterministic-seeded
   helpers for fault injection in tests. No production code path
   uses it.

**Non-goals (v1):**

- **Tracing across daemons.** Single-daemon. `correlation_id` is the
  cross-daemon hook for a future cluster RFC.
- **Distributed tracing (OTel).** Out of scope. Future RFC may add
  an OpenTelemetry exporter; the chain we expose here is OpenTelemetry-
  compatible in shape.
- **Persistent metric history.** Prometheus is the right place to
  store historical metrics; the kernel only exposes the current
  snapshot.
- **Live event subscription.** RFC 0001's `/events` SSE channel
  already covers this. RFC 0012 adds read-style observe APIs, not
  push.
- **Audit query language.** `kernel.events.tail` is the read API for
  events; observe.trace just walks causation. A SQL-ish DSL is
  overkill.

## 2. Data model

No new dataclasses; all responses are dicts conforming to:

```
ProcView (kernel.observe.proc result):
  {
    "process":   AgentProcess.to_dict()  | None,
    "capability": Capability.to_dict()    | None,
    "ledger":    [LedgerEntry.to_dict()],
    "mailbox":   {
                    "exists":  bool,
                    "queue_size": int | null,
                    "pending": int | null,
                    "subscriptions": [str],
                  },
    "scheduler": {
                    "queued":     int,
                    "dispatched": int,
                    "completed":  int,
                    "expired":    int,
                    "cancelled":  int,
                  },
    "fs":        { "object_count": int, "total_bytes": int },
    "registry":  { "names": [str] },
    "recent_events": [Event.to_dict()],   # last 20 events for this pid
  }

SummaryView (kernel.observe.summary result):
  {
    "kernel_version":  str,
    "schema_version":  int,
    "started_at":      float,           # ObservabilityStore instantiation time
    "uptime_s":        float,
    "agents": {
        "READY":     int, "RUNNING": int, "WAITING": int,
        "SUSPENDED": int, "DEAD":    int,
        "total":     int,
    },
    "events":     { "total": int, "max_event_id": int },
    "scheduler":  { "queued": int, "dispatched": int, ... },
    "mailbox":    { "mailboxes": int, "pending_messages": int },
    "ledger":     { "agents_with_budgets": int, "breached": int },
    "fs":         { "objects": int, "total_bytes": int },
    "registry":   { "entries": int },
  }

TraceResult (kernel.observe.trace result):
  {
    "events": [Event.to_dict()],   # ordered downstream → upstream
    "depth":  int,                 # actual depth walked (≤ requested)
    "truncated": bool,             # True if depth limit hit
  }
```

## 3. Operations

### `proc(pid) -> ProcView`

One read across all tables filtered by pid. Cheap; uses indexed
queries on each.

### `summary() -> SummaryView`

System rollup. Aggregates COUNT() against each table. The kernel
holds no in-memory cached counters; every call re-queries. SQLite
WAL keeps reads cheap; if this becomes a bottleneck (unlikely at
v1's scale), a future RFC can add cached counters with periodic
sync.

### `trace(event_id, depth=10) -> TraceResult`

Starts at `event_id`, walks `causation_id` upstream up to `depth`
hops, returns the chain. Halts on:
- causation_id is NULL (root cause)
- depth limit hit (sets `truncated=True`)
- referenced event missing (corrupt log; sets `truncated=True`)

`depth` is capped at 100.

### `prometheus_text() -> str`

Returns Prometheus text exposition format string. Includes (at
minimum):

```
# HELP cheetahclaws_kernel_schema_version Kernel schema version
# TYPE cheetahclaws_kernel_schema_version gauge
cheetahclaws_kernel_schema_version 5

# HELP cheetahclaws_kernel_uptime_seconds Seconds since kernel registered
# TYPE cheetahclaws_kernel_uptime_seconds counter
cheetahclaws_kernel_uptime_seconds 12345.6

# HELP cheetahclaws_kernel_agents Number of agents by state
# TYPE cheetahclaws_kernel_agents gauge
cheetahclaws_kernel_agents{state="READY"} 3
cheetahclaws_kernel_agents{state="RUNNING"} 1
... (and DEAD)

# HELP cheetahclaws_kernel_events_total Lifetime event count
# TYPE cheetahclaws_kernel_events_total counter
cheetahclaws_kernel_events_total 9876

# HELP cheetahclaws_kernel_scheduler_queue Items in scheduler queue by state
# TYPE cheetahclaws_kernel_scheduler_queue gauge
cheetahclaws_kernel_scheduler_queue{state="queued"} 5
cheetahclaws_kernel_scheduler_queue{state="dispatched"} 2

# HELP cheetahclaws_kernel_ledger_breached Agents with at least one breached dim
# TYPE cheetahclaws_kernel_ledger_breached gauge
cheetahclaws_kernel_ledger_breached 0

# HELP cheetahclaws_kernel_mailbox_pending Pending messages across all inboxes
# TYPE cheetahclaws_kernel_mailbox_pending gauge
cheetahclaws_kernel_mailbox_pending 12

# HELP cheetahclaws_kernel_fs_objects Total AgentFS objects
# TYPE cheetahclaws_kernel_fs_objects gauge
cheetahclaws_kernel_fs_objects 84

# HELP cheetahclaws_kernel_fs_bytes Total AgentFS bytes stored
# TYPE cheetahclaws_kernel_fs_bytes gauge
cheetahclaws_kernel_fs_bytes 1048576
```

Format compliance:
- `# HELP` and `# TYPE` lines per metric.
- Label values double-quoted, escaped.
- Newline-separated.
- Final newline mandatory.

The daemon's `/metrics` endpoint integration is described in §5.

## 4. RPC surface

```
kernel.observe.proc
  params: { pid }
  result: ProcView

kernel.observe.summary
  params: {}
  result: SummaryView

kernel.observe.trace
  params: { event_id, depth?=10 }
  result: TraceResult

kernel.observe.prometheus
  params: {}
  result: { text: str }       # Prometheus exposition string
```

No new error codes — observe methods just propagate
`InvalidPayload` for bad params (already -32602) and
`UnknownPid` (already -32102) where applicable.

## 5. Daemon `/metrics` integration

The daemon's existing `/metrics` endpoint goes through
`health.payload_for(path, config)`. To preserve byte-for-byte
backwards compatibility, the kernel does NOT modify
`/metrics` directly. Instead:

- `kernel.observe.prometheus` is the canonical kernel-metrics RPC.
- A future, optional patch can wire kernel metrics into the daemon's
  `/metrics` payload by reading `daemon_state.observability_store`
  if present. This RFC does not commit to that wiring; it ships the
  RPC and leaves the merge as a follow-up.

This separation means `/metrics` remains a stable JSON contract for
clients that already scrape it, while Prometheus users opt into
kernel metrics through the RPC channel (or via a future text passthrough).

## 6. Chaos primitives

A new module `kernel/chaos.py`, intended for use **only by
tests**. Production code paths must not import from it.

```python
class ChaosMonkey:
    def __init__(self, *, seed: int | None = None): ...

    def kill_random_agent(self, kernel_store) -> int | None:
        """Pick a non-DEAD agent at random; transition it to DEAD with
        exit_kind='crashed'. Returns the pid killed, or None if no
        live agents exist. Deterministic given the same seed."""

    def fill_mailbox(self, mailbox_store, pid: int) -> int:
        """Send messages to a pid's mailbox until it's full. Returns
        count of successful sends."""

    def simulate_disk_full(self) -> contextmanager:
        """Context manager that monkey-patches sqlite3 to raise
        sqlite3.OperationalError('disk full') on next write."""

    def lose_event(self, kernel_store, event_id: int) -> bool:
        """Manually delete an event row. Tests robustness against
        log corruption (which should never happen, but the kernel
        should degrade gracefully)."""
```

The module is a few hundred lines at most and ships with one smoke
test:

`tests/test_kernel_chaos_smoke.py` — Spawn a daemon, register kernel,
create 5 agents, run 3 chaos operations, verify the daemon doesn't
crash and `kernel.observe.summary()` still returns sensible numbers.

## 7. Backwards compatibility

- No schema change.
- No existing module modified.
- The daemon's `/metrics` JSON payload is unchanged.

## 8. Open questions

1. **Should `summary()` include event-rate (events/sec)?** Requires
   tracking a windowed counter in memory. Useful but adds state. Lean:
   ship without rate; add in v1.1 if Prometheus users ask.
2. **Should `proc(pid)` include sandbox info?** The kernel doesn't
   own sandbox state — that lives in the supervisor's process tree.
   Lean: leave out; supervisor's own observability covers it.
3. **Should `prometheus_text` be split per-agent labels?** A 100-agent
   system would emit 700+ time series. Cardinality matters. Lean:
   no per-pid labels at v1; aggregate only. Per-pid is opt-in via a
   separate `kernel.observe.proc_prometheus(pid)` if anyone asks.

## 9. Acceptance criteria

A PR claiming this RFC must:

1. `kernel.observe.proc(pid)` returns a single dict with all
   sub-fields, even when the pid has no rows in some tables (those
   sub-fields are empty/None, not missing).
2. `kernel.observe.summary()` reports correct counts that match a
   manual SUM over each table.
3. `kernel.observe.trace(event_id)` walks causation_id correctly
   (verified by enqueueing a known causation chain and asserting the
   walk).
4. `kernel.observe.prometheus()` produces text that parses with the
   `prometheus-client` parser if available, OR passes a basic regex
   contract test.
5. `ChaosMonkey.kill_random_agent` is deterministic given a seed.
6. The smoke test demonstrates a daemon survives 3 chaos operations
   without crashing.
7. No file outside `kernel/`, `tests/`, `docs/RFC/` modified.
