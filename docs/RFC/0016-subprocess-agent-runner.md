# Design Note: Subprocess Agent Runner — kernel ↔ OS process bridge

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Tracks roadmap:** [`0002-daemon-foundation-roadmap.md`](./0002-daemon-foundation-roadmap.md) §F-4 (subprocess agent runner)
- **Builds on:** every prior kernel RFC, especially:
  - [`0003`](./0003-agent-process-and-event-log.md) — agent state machine
  - [`0006`](./0006-resource-ledger.md) — ledger
  - [`0008`](./0008-agent-sandbox.md) — sandbox

This RFC defines the bridge that turns **kernel primitives** (data
models, state machines, ledgers, sandboxes) into **running OS
processes**. Up through Phase 5 the kernel is rich but inert — nothing
actually runs an agent. RFC 0016 makes the kernel *active* by adding a
supervisor that spawns subprocesses, applies sandbox policies, wires
state transitions, and reports back through the kernel's audit and
ledger surfaces.

The runner is **purely additive** to existing code. The current
`agent_runner.py` (autonomous Markdown agent loop) is **not modified**
and continues to work unchanged. RFC 0016 introduces a parallel,
kernel-managed runner that lives in a new `kernel/runner/` package.
A future patch can choose to migrate `agent_runner.py` onto this
substrate; this RFC does not commit to that migration.

## 1. Goals & non-goals

**Goals:**

1. **Subprocess-per-agent.** Each spawned agent is its own OS process,
   inheriting RFC 0008's sandbox guarantees: RLIMIT, optional
   bubblewrap, wall-clock kill on the parent side.
2. **Kernel state coordination.** Spawning transitions the AgentProcess
   READY → RUNNING; clean exit transitions RUNNING → DEAD with the
   right `exit_kind`; crashes go to DEAD with `exit_kind="crashed"`.
   All transitions go through `KernelStore.transition` so the event log
   captures them.
3. **JSON-line IPC.** Supervisor and runner communicate via line-
   delimited JSON over the runner's stdin (S → R) and stdout (R → S).
   Stderr is captured for diagnostics.
4. **Ledger integration.** Wall-time is charged against
   `wall_s` if the agent has a row. Token / cost / tool-call charges
   are reported by the runner via IPC; the supervisor applies them.
5. **Crash isolation.** A `kill -9` on the runner is invisible to the
   daemon. The supervisor detects the wait status and transitions the
   agent to DEAD with `exit_kind="crashed"`. Other runners are
   unaffected.

**Non-goals (v1):**

- **Real LLM provider integration.** This RFC ships the substrate;
  the runner_main entry point shipped here is a minimal echo loop for
  testing. The "real" agent runner (with model calls, tool dispatch,
  permission requests routed through `daemon/permissions.py`) is a
  follow-up PR — easy to layer on the substrate.
- **Restart on crash.** A crashed runner stays DEAD. A future
  scheduler-driven retry policy lives in RFC 0007's territory: enqueue
  a new schedule entry pointing at the same agent, supervisor spawns
  a fresh subprocess.
- **Cross-host spawn.** Single daemon. RFC 0015 cluster.
- **Streaming model output via the IPC channel.** Output is
  per-iteration / per-event JSON messages; large model streams are
  the runner's problem (it can buffer and emit periodic chunks).
- **Auto-recovery of running subprocesses on daemon restart.** The
  supervisor is in-memory; daemon restart loses the subprocess
  registry. Kernel-side, RFC 0003's startup recovery moves stale
  RUNNING → SUSPENDED, which is correct: those subprocesses no longer
  exist, only the supervisor's view of them was lost.

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ daemon process                                                    │
│                                                                    │
│  ┌──────────────┐   ┌─────────────────────────┐                  │
│  │  daemon   │   │  kernel.runner       │                  │
│  │  RPC server  │   │  ┌───────────────────┐   │                  │
│  └──────┬───────┘   │  │ RunnerSupervisor  │   │                  │
│         │           │  └─┬─────┬─────┬─────┘   │                  │
│         │           │    │     │     │         │                  │
│         │           │  ┌─▼─┐ ┌─▼─┐ ┌─▼─┐       │                  │
│         │           │  │ K │ │ K │ │ K │       │                  │
│         │           │  │ I │ │ I │ │ I │       │                  │
│         │           │  │ P │ │ P │ │ P │       │                  │
│         │           │  │ C │ │ C │ │ C │       │                  │
│         │           │  └─┬─┘ └─┬─┘ └─┬─┘       │                  │
│         │           └────┼─────┼─────┼─────────┘                  │
│         │                │     │     │                            │
│  ┌──────▼────────────────▼─────▼─────▼───────────────┐            │
│  │ kernel.db (process state, events, ledger, ...)      │            │
│  └─────────────────────────────────────────────────────┘            │
└──────────┬──────────────┬─────────────┬──────────────────────────┘
           │              │             │
           │ stdin/stdout │             │
           ▼              ▼             ▼
      ┌─────────┐    ┌─────────┐   ┌─────────┐
      │ agent A │    │ agent B │   │ agent C │   ← subprocesses
      │ (PID 1) │    │ (PID 2) │   │ (PID 3) │     under sandbox
      └─────────┘    └─────────┘   └─────────┘
```

## 3. Data model

### `RunnerHandle`

```python
@dataclass(frozen=True)
class RunnerHandle:
    pid:           int                  # AgentProcess pid
    os_pid:        int                  # subprocess.Popen.pid
    started_at:    float
    sandbox:       SandboxPolicy
```

In-memory only; held in the supervisor's registry.

### `RunnerExitInfo`

```python
@dataclass(frozen=True)
class RunnerExitInfo:
    pid:          int
    exit_kind:    str                   # 'completed'|'cancelled'|'failed'|'crashed'
    exit_code:    int                   # subprocess return code
    stdout_tail:  bytes                 # last 4 KB
    stderr_tail:  bytes                 # last 4 KB
    duration_s:   float
    ledger_charged: dict                # {dim: amount} actually charged
```

## 4. IPC protocol

Newline-delimited JSON. One JSON object per line. UTF-8.

### Supervisor → Runner (stdin)

```jsonc
// First message after spawn:
{ "op": "init", "pid": 42, "payload": {…} }

// Stop signal (graceful):
{ "op": "stop" }
```

### Runner → Supervisor (stdout)

```jsonc
// Required first response:
{ "op": "ready", "pid": 42 }

// Optional progress / log:
{ "op": "log", "level": "info", "msg": "…" }
{ "op": "iteration_start", "iter": 1 }
{ "op": "iteration_done",  "iter": 1, "tokens": 150, "cost_micro": 250 }

// Optional: ask supervisor to charge a custom ledger dim
{ "op": "charge", "dim": "tool_calls", "amount": 1 }

// Required final message:
{ "op": "exit", "exit_kind": "completed", "summary": "…" }
```

The runner exits its process **after** writing the exit message and
flushing stdout. The supervisor reads exit, then waits for the OS
process; mismatched (subprocess dies without `exit` message) =
`exit_kind="crashed"`.

### Stderr

Stderr is captured and tailed (last 4 KB) into `RunnerExitInfo`. Not
part of the protocol — runners may use it for unstructured logs.

## 5. RunnerSupervisor API (Python)

This RFC ships **no new RPC methods**. The supervisor is a Python API
called by daemon-internal code (and by future RPC patches that want
to expose `kernel.runner.*` to clients). Avoiding RPC at v1 keeps
spawn-arbitrary-code out of the wire surface; supervisor RPC is a
separate RFC to write later.

```python
class RunnerSupervisor:
    def __init__(self, kernel_store, *,
                 ledger_store=None,
                 default_policy=SANDBOX_DEFAULT,
                 ipc_timeout_s=5.0):
        ...

    def spawn(self, *, pid: int, argv: Sequence[str],
              policy: SandboxPolicy | None = None,
              init_payload: dict | None = None,
              env: Mapping[str, str] | None = None,
              cwd: str | None = None) -> RunnerHandle:
        """Spawn a subprocess, complete the init/ready handshake,
        transition agent READY→RUNNING. Raises if agent isn't READY,
        or if init handshake fails (returns IllegalRunnerState)."""

    def wait(self, pid: int, timeout: float | None = None) -> RunnerExitInfo:
        """Block until the runner emits 'exit' and the OS process is
        reaped. Charges the ledger as instructed by 'charge' messages
        (and a final wall_s charge). Transitions agent to DEAD."""

    def stop(self, pid: int, *, exit_kind: str = "cancelled") -> RunnerExitInfo:
        """Send 'stop' over stdin; if the runner doesn't exit within
        ipc_timeout_s, terminate the process group (SIGTERM →
        SIGKILL after 1s grace via the existing wall-clock killer
        pattern)."""

    def list(self) -> list[RunnerHandle]:
        """Snapshot of currently-tracked handles."""

    def cleanup(self) -> int:
        """Reap any zombies, prune dead handles. Returns count cleaned."""
```

## 6. Sandbox application

The supervisor builds the subprocess command line by:

1. If `policy.use_bubblewrap=True`: prepend `bwrap` arguments via
   `kernel.sandbox.wrap_with_bubblewrap`.
2. Always: pass `apply_rlimits_in_child(policy)` as `preexec_fn`.
3. Always: redirect stdin/stdout to pipes; stderr to a pipe the
   supervisor reads with a tail.
4. Optional: spawn a wall-clock killer thread per the existing
   `_wall_clock_killer` from RFC 0008.

Sandbox failures (bwrap missing while `use_bubblewrap=True`) raise
`SandboxNotAvailable` from `wrap_with_bubblewrap` — propagate
unchanged.

## 7. Ledger integration

Two charge paths:

**Wall-time** (always, if ledger row exists for `wall_s`):

The supervisor records `started_at` at spawn. On `wait()` completion,
it computes `int(duration_s)` and calls `ledger.charge(pid, "wall_s",
seconds)`. Failure to charge (no row, etc.) is silent.

**Custom dims** (per-runner):

When the runner emits `{"op":"charge", "dim":"tokens", "amount":150}`,
the supervisor calls `ledger.charge(pid, "tokens", 150)` and stores
the `over_limit` result locally. Over-limit doesn't kill the runner
in v1 — it's the supervisor's policy. We log first_breach via the
event log:

```python
if charge_result.first_breach:
    # Supervisor is a kernel CLIENT, not the kernel itself, so it
    # cannot use the reserved ``kernel.*`` event prefix. The audit
    # event lives under ``runner.*`` instead.
    kernel.events_append(
        pid=pid, kind="runner.first_breach",
        payload={"dim": dim, "used": ..., "granted": ...},
    )
```

The supervisor or scheduler (RFC 0007) can react in a follow-up
patch.

## 8. Backwards compatibility

- No schema change.
- No file outside `kernel/runner/` (new package), `tests/`, and
  `docs/RFC/` is modified.
- Existing `agent_runner.py` is not touched. The new runner is a
  parallel surface for kernel-managed agents.
- No new RPC methods — supervisor is a Python API for now.

## 9. Open questions

1. **Should `wait` charge `wall_s` even if the agent has no row?** No
   — silent skip is the principle for ledger integration. Operators
   create the row if they want tracking.
2. **Should `stop` use `kernel.agent.transition` to SUSPENDED first,
   then to DEAD?** Currently goes straight to DEAD. SUSPENDED is for
   "paused, may resume"; once we've killed the OS process, resume
   isn't possible. Direct DEAD is correct.
3. **Concurrency: is one supervisor instance per daemon enough?**
   Yes. The supervisor's lock is per-instance; multiple instances on
   the same kernel.db would race for state transitions. v1 ships
   one-supervisor-per-daemon as the contract.

## 10. Acceptance criteria

A PR claiming this RFC must:

1. `RunnerSupervisor.spawn` correctly transitions READY → RUNNING
   and creates a `kernel.process.transitioned` event in the event
   log.
2. A clean runner (writes `exit` and exits 0) → `wait` returns
   `RunnerExitInfo(exit_kind="completed", exit_code=0)` and the agent
   is DEAD.
3. A `kill -9` on the OS pid → `wait` returns
   `RunnerExitInfo(exit_kind="crashed", exit_code != 0)` and the
   agent is DEAD; the daemon process keeps running.
4. `stop` sends graceful, then escalates after `ipc_timeout_s`; the
   runner gets either SIGTERM or SIGKILL.
5. With `use_bubblewrap=True`, the spawned process can't read paths
   outside `bind_ro` (smoke test against `~/.ssh/`).
6. With `wall_seconds` set in the policy and a runaway runner, the
   wall-clock killer fires; agent goes to DEAD.
7. With a `wall_s` ledger row, `wait` charges (duration_s, integer
   seconds); the row's `used` reflects.
8. Custom `charge` messages from the runner translate to ledger
   charges; first_breach generates a `kernel.runner.first_breach`
   event.
9. No file outside `kernel/`, `tests/`, `docs/RFC/` modified.
