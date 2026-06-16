"""worker.py — WorkerLoop ties scheduler.claim → supervisor.spawn (RFC 0017).

The bridge between the **scheduler** (passive ready queue) and the
**supervisor** (active subprocess lifecycle). Until WorkerLoop, the
two were independent primitives that callers had to wire by hand.
This module does the wiring.

Strictly additive:
  * No schema changes, no new RPC methods.
  * Existing `agent_runner.py` path is untouched.
  * Ships behind no flag — the loop is a Python class; only callers
    that explicitly construct one see new behaviour.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from .errors import (
    RunnerHandshakeFailed,
    RunnerIllegalState,
    RunnerIpcTimeout,
    SchedIllegalTransition,
)
from .sandbox import SandboxPolicy

if TYPE_CHECKING:
    from .runner import RunnerSupervisor
    from .scheduler import ReadyEntry, SchedulerStore
    from .store import KernelStore


log = logging.getLogger(__name__)


ArgvFactory   = Callable[["ReadyEntry"], Sequence[str]]
PolicyFactory = Callable[["ReadyEntry"], SandboxPolicy]
EnvFactory    = Callable[["ReadyEntry"], Optional[dict]]


# Mapping from RunnerExitInfo.exit_kind to scheduler complete arg.
# Both vocabularies happen to align in v1 — kept explicit for clarity
# and to localise the mapping if either set diverges.
_EXIT_KIND_MAP = {
    "completed": "completed",
    "cancelled": "cancelled",
    "failed":    "failed",
    "crashed":   "crashed",
}


class WorkerLoop:
    """One driver thread + N transient runner threads.

    Lifecycle:
      ``WorkerLoop(...)`` → idle.
      ``start()``         → driver thread begins ticking.
      ``stop()``          → driver halts; in-flight may drain or
                            be killed depending on flags.

    Capacity:
      ``max_concurrent`` caps the number of in-flight runner threads.
      Each in-flight runner owns one OS subprocess (via the
      supervisor) plus the Python thread waiting on it.

    Reentrancy:
      ``tick()`` is safe to call from multiple threads; the
      supervisor and scheduler serialise their internal writes. Only
      one ``start()`` driver per WorkerLoop instance, though.
    """

    def __init__(
        self,
        *,
        kernel_store:      "KernelStore",
        scheduler_store:   "SchedulerStore",
        supervisor:        "RunnerSupervisor",
        argv_factory:      ArgvFactory,
        policy_factory:    Optional[PolicyFactory] = None,
        env_factory:       Optional[EnvFactory]    = None,
        worker_id:         str = "worker-0",
        max_concurrent:    int = 4,
        poll_interval_s:   float = 1.0,
        wait_timeout_s:    float = 300.0,
    ) -> None:
        self._kernel = kernel_store
        self._sched  = scheduler_store
        self._sup    = supervisor
        self._argv_factory   = argv_factory
        self._policy_factory = policy_factory
        self._env_factory    = env_factory
        self._worker_id      = worker_id
        self._max_concurrent = max(1, int(max_concurrent))
        self._poll_interval  = max(0.05, float(poll_interval_s))
        self._wait_timeout   = max(1.0, float(wait_timeout_s))

        self._capacity = threading.Semaphore(self._max_concurrent)
        self._in_flight: dict[int, threading.Thread] = {}
        self._lock = threading.Lock()

        self._driver_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── public surface ────────────────────────────────────────────────

    def in_flight(self) -> int:
        with self._lock:
            return len(self._in_flight)

    def tick(self) -> bool:
        """One iteration. Claim ≤1 entry (subject to capacity), spawn
        it on a worker thread, return whether work was claimed."""
        # Try to acquire a capacity slot non-blocking. If full, return
        # False so callers (the driver) know to back off.
        if not self._capacity.acquire(blocking=False):
            return False
        try:
            entries = self._sched.claim(
                worker_id=self._worker_id, max_n=1,
            )
        except Exception:
            self._capacity.release()
            raise
        if not entries:
            self._capacity.release()
            return False
        entry = entries[0]

        # Spawn on a worker thread so the driver can keep ticking.
        thread = threading.Thread(
            target=self._run_one, args=(entry,),
            daemon=True, name=f"{self._worker_id}-pid{entry.pid}",
        )
        with self._lock:
            self._in_flight[entry.pid] = thread
        thread.start()
        return True

    def start(self) -> None:
        """Start the background driver. Idempotent — calling twice
        on a running instance is a no-op."""
        with self._lock:
            if self._driver_thread is not None and self._driver_thread.is_alive():
                return
            self._stop_event.clear()
            self._driver_thread = threading.Thread(
                target=self._driver, daemon=True,
                name=f"{self._worker_id}-driver",
            )
            self._driver_thread.start()

    def stop(
        self, *,
        drain: bool = True,
        drain_timeout_s: float = 30.0,
    ) -> int:
        """Stop the driver. If drain=True, wait up to drain_timeout_s
        for in-flight runners. After timeout (or with drain=False),
        kill remaining runners' OS processes directly via
        ``supervisor.kill``. Returns count force-killed.

        We deliberately use ``supervisor.kill`` (signal-only) rather
        than ``supervisor.stop`` (signal + wait + transition).
        Calling stop from here would race with the worker thread's
        already-running ``wait()`` on the same pid — both try to grab
        the handle, then one raises RunnerUnknownPid (RFC 0016 §6
        documents the contention). Killing the OS process is enough:
        the worker thread's wait() observes EOF, transitions the
        agent to DEAD, and completes the scheduler entry.
        """
        self._stop_event.set()
        if self._driver_thread is not None:
            self._driver_thread.join(timeout=2.0)

        if drain:
            deadline = time.monotonic() + max(0.0, drain_timeout_s)
            while time.monotonic() < deadline:
                if self.in_flight() == 0:
                    return 0
                time.sleep(0.1)

        # Either drain=False, or drain timed out — kill remaining.
        with self._lock:
            pids = list(self._in_flight.keys())
        killed = 0
        for pid in pids:
            try:
                if self._sup.kill(pid):
                    killed += 1
            except Exception:
                pass
        # Wait for worker threads to observe exit and clean up.
        cleanup_deadline = time.monotonic() + 10.0
        while time.monotonic() < cleanup_deadline and self.in_flight() > 0:
            time.sleep(0.05)
        return killed

    # ── internals ────────────────────────────────────────────────────

    def _driver(self) -> None:
        while not self._stop_event.is_set():
            try:
                worked = self.tick()
            except Exception as e:  # noqa: BLE001 — driver must not crash
                log.exception("worker driver tick raised: %s", e)
                worked = False
            # If we did claim work, loop again immediately to drain
            # the queue while capacity allows.
            if worked:
                continue
            # Idle: wait poll_interval (or wake on stop).
            self._stop_event.wait(timeout=self._poll_interval)

    def _run_one(self, entry: "ReadyEntry") -> None:
        """Spawn → wait → complete. Releases capacity on exit."""
        try:
            argv = self._argv_factory(entry)
            policy = (self._policy_factory(entry)
                      if self._policy_factory else None)
            env = (self._env_factory(entry)
                   if self._env_factory else None)
            sched_exit_kind = "completed"
            try:
                self._sup.spawn(
                    pid=entry.pid, argv=argv,
                    policy=policy, env=env,
                    init_payload={
                        "sched_id": entry.sched_id,
                        "trigger":  entry.trigger,
                        "payload":  entry.payload,
                    },
                )
            except RunnerIllegalState:
                # Agent state mismatched expectations; treat as
                # cancelled at the queue level.
                sched_exit_kind = "cancelled"
            except (RunnerHandshakeFailed, RunnerIpcTimeout) as e:
                log.warning("worker: spawn failed for sched_id=%s pid=%s: %s",
                            entry.sched_id, entry.pid, e)
                sched_exit_kind = "failed"
            except Exception as e:  # noqa: BLE001
                log.exception("worker: spawn raised for sched_id=%s pid=%s",
                              entry.sched_id, entry.pid)
                sched_exit_kind = "failed"
            else:
                # Runner started; wait + map exit_kind.
                try:
                    info = self._sup.wait(
                        entry.pid, timeout=self._wait_timeout,
                    )
                    sched_exit_kind = _EXIT_KIND_MAP.get(
                        info.exit_kind, "failed",
                    )
                except Exception as e:  # noqa: BLE001
                    log.exception("worker: wait raised for sched_id=%s pid=%s",
                                  entry.sched_id, entry.pid)
                    sched_exit_kind = "failed"

            # Complete the queue entry. Idempotent against
            # double-complete; we tolerate the failure.
            try:
                self._sched.complete(
                    entry.sched_id, exit_kind=sched_exit_kind,
                )
            except SchedIllegalTransition:
                # Already completed (e.g. supervisor.stop took both
                # paths, or a concurrent operator cancelled). OK.
                pass
            except Exception as e:  # noqa: BLE001
                log.exception("worker: complete raised for sched_id=%s",
                              entry.sched_id)
        finally:
            with self._lock:
                self._in_flight.pop(entry.pid, None)
            self._capacity.release()
