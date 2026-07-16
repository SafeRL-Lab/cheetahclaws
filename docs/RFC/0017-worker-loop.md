# Design Note: WorkerLoop — scheduler ↔ supervisor glue

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0007-agent-scheduler.md`](./0007-agent-scheduler.md), [`0016-subprocess-agent-runner.md`](./0016-subprocess-agent-runner.md)

This RFC closes the last gap between the kernel's two execution
primitives: the **scheduler** (RFC 0007) holds work; the **supervisor**
(RFC 0016) runs subprocesses. Until now, nothing in the kernel
decides which queued entry becomes a subprocess and when. The
**WorkerLoop** is that decider.

The loop is intentionally thin. All hard decisions — what to claim
(scheduler), how to sandbox (RFC 0008), what to charge (RFC 0006),
how to transition (RFC 0003) — are owned by the underlying RFCs.
WorkerLoop adds only:

1. A repeating tick that calls `scheduler.claim` then
   `supervisor.spawn`.
2. A concurrency cap (`max_concurrent`) so the supervisor isn't
   asked to spawn 1000 subprocesses at once.
3. Mapping from `RunnerExitInfo.exit_kind` to
   `scheduler.complete(sched_id, exit_kind)`.
4. A graceful shutdown that drains in-flight runners or kills them
   after a deadline.

This RFC ships **purely additive** code. **No schema bump**, no new
RPC methods, no kernel.* surface change. The contract test
(RFC 0013) doesn't move.

## 1. Goals & non-goals

**Goals:**

1. **Single-binary tick.** `WorkerLoop.tick()` advances the world by
   one step: claim ≤1 entry, spawn it, return. Synchronous; useful
   for tests and CLI.
2. **Background mode.** `WorkerLoop.start()` spawns one driver
   thread that calls tick on a poll interval; `stop()` triggers
   graceful shutdown.
3. **Bounded parallelism.** A semaphore caps concurrent in-flight
   runners. The driver doesn't claim past the cap.
4. **Correct accounting.** Every claimed entry that gets spawned
   ends up with a matching `scheduler.complete`. Failed spawns
   re-enqueue (or get cancelled, depending on policy).

**Non-goals (v1):**

- **Smart prefetching.** The loop claims one at a time. A future
  optimisation can claim N when the cap allows, but ordering
  becomes harder to reason about.
- **Adaptive backoff.** Idle ticks sleep for a fixed
  `poll_interval`. A future RFC may tune this based on queue depth.
- **Cross-supervisor coordination.** One worker loop per supervisor
  per daemon. Multiple loops on the same scheduler would race for
  claims (the kernel's atomic claim handles this, but cap policy
  becomes confused).
- **Restart on runner crash.** If a runner crashes, the loop calls
  `scheduler.complete(exit_kind="crashed")` and moves on. Retry is
  the orchestrator's job: enqueue a new entry.

## 2. API

```python
class WorkerLoop:
    def __init__(
        self,
        *,
        kernel_store:       KernelStore,
        scheduler_store:    SchedulerStore,
        supervisor:         RunnerSupervisor,
        argv_factory:       Callable[[ReadyEntry], Sequence[str]],
        policy_factory:     Callable[[ReadyEntry], SandboxPolicy] | None = None,
        env_factory:        Callable[[ReadyEntry], dict | None] | None = None,
        worker_id:          str = "worker-0",
        max_concurrent:     int = 4,
        poll_interval_s:    float = 1.0,
        wait_timeout_s:     float = 300.0,
    ): ...

    def tick(self) -> bool:
        """One iteration. Claim ≤1 entry (subject to capacity);
        spawn it on a fresh thread that runs `wait` then `complete`.
        Returns True if work was claimed, False if idle."""

    def start(self) -> None:
        """Spawn the driver thread that calls tick() in a loop until
        stop()."""

    def stop(self, *, drain: bool = True, drain_timeout_s: float = 30.0) -> int:
        """Halt the driver; if drain=True, wait up to drain_timeout_s
        for in-flight runners. Returns count of forcibly-killed
        runners (0 if drain succeeded)."""

    def in_flight(self) -> int:
        """Current count of running runners under this loop."""
```

### `argv_factory` / `policy_factory` / `env_factory`

These are caller-supplied because:

- The kernel doesn't know what command to run for an arbitrary
  agent. The orchestrator owns that mapping.
- The default factory (`lambda entry: [sys.executable, "-m",
  "kernel.runner.runner_main"]`) suffices for tests.
- Policy / env can vary per agent — different sandboxes, different
  budgets — and the orchestrator decides.

## 3. Tick semantics

```
tick():
  if in_flight >= max_concurrent:
      return False                       # backpressure
  entries = scheduler.claim(worker_id, max_n=1)
  if not entries:
      return False                       # nothing ready
  entry = entries[0]
  argv   = argv_factory(entry)
  policy = (policy_factory(entry) if policy_factory
            else supervisor._default_policy)
  env    = env_factory(entry) if env_factory else None

  # Spawn on a worker thread so the driver can keep ticking.
  thread = Thread(target=_run_one, args=(entry, argv, policy, env))
  thread.start()                         # in_flight += 1
  return True
```

`_run_one` is:

```
_run_one(entry, argv, policy, env):
  try:
      supervisor.spawn(pid=entry.pid, argv=argv, policy=policy, env=env)
      info = supervisor.wait(entry.pid, timeout=wait_timeout_s)
      scheduler.complete(entry.sched_id, exit_kind=info.exit_kind)
  except RunnerIllegalState:
      # Agent state changed under us (e.g. concurrent transition).
      # Mark the queue entry cancelled.
      scheduler.complete(entry.sched_id, exit_kind="cancelled")
  except Exception as e:
      # Spawn failed (handshake timeout, sandbox unavailable, etc.).
      # Mark the queue entry failed and let observability surface it.
      scheduler.complete(entry.sched_id, exit_kind="failed")
  finally:
      in_flight -= 1
```

## 4. Shutdown

`stop(drain=True, drain_timeout_s=30)`:

1. Driver thread exits its tick loop on the next poll.
2. Wait up to `drain_timeout_s` for in-flight runners to complete.
3. After deadline: call `supervisor.stop(pid)` on every still-live
   handle. `stop` escalates SIGTERM → SIGKILL via the existing
   supervisor logic.
4. Return the count of runners killed.

`stop(drain=False)`:

1. Driver exits.
2. `supervisor.stop` immediately on every live handle.
3. Returns the kill count.

## 5. RFC mapping table

| Layer | Owns | WorkerLoop's role |
|---|---|---|
| Scheduler (0007) | queue state, atomic claim | calls `claim` and `complete` |
| Supervisor (0016) | subprocess lifecycle, sandbox, ledger | calls `spawn` and `wait` |
| Kernel state (0003) | agent state machine | observed only — supervisor mutates it |
| Sandbox (0008) | RLIMIT, bubblewrap, wall-killer | applied by supervisor; loop passes the policy through |
| Ledger (0006) | per-agent budgets | charged by supervisor on wait |
| Capability (0005) | tool/fs/net/model whitelist | enforcement is the runner's responsibility (out of scope here) |

## 6. Backwards compatibility

- No schema change.
- No new RPC method.
- No file outside `kernel/`, `tests/`, `docs/RFC/` modified.

## 7. Open questions

1. **Claim batch size.** v1 claims `max_n=1` per tick. Bumping to
   `max_n=N` reduces tick overhead but complicates capacity
   accounting. Lean: defer.
2. **What if `scheduler.complete` fails?** A second `complete` call
   on a `completed` entry raises `SchedIllegalTransition`. We log
   and ignore — the desired terminal state has been reached.
3. **Multiple workers per scheduler?** Two `WorkerLoop` instances
   sharing one scheduler would each call `claim`. The kernel's
   `BEGIN IMMEDIATE` already serialises so duplication can't
   happen. Capacity caps remain per-loop. Use case: scaling out
   into multiple supervisors. Lean: support but document.

## 8. Acceptance criteria

A PR claiming this RFC must:

1. `tick()` claims a queued entry and the scheduler reflects
   `dispatched → completed` after the runner exits.
2. With `max_concurrent=2` and 5 entries enqueued, never more than
   2 runners in-flight; total time ≈ ⌈5/2⌉ × runner duration.
3. `stop(drain=True)` lets in-flight runners finish; returns 0.
4. `stop(drain=False)` (or drain timeout) kills in-flight runners;
   the affected scheduler entries get
   `complete(exit_kind="cancelled")` (or "crashed").
5. Background `start()` keeps draining the queue without explicit
   tick calls until `stop()`.
6. No file outside `kernel/`, `tests/`, `docs/RFC/` modified.
