#!/usr/bin/env python3
"""kernel_e2e_smoke.py — end-to-end smoke for the cheetahclaws kernel.

Demonstrates the full chain wired together:

    Kernel.open
      → create agents (RFC 0003)
      → grant capabilities (RFC 0005)
      → set ledger budgets (RFC 0006)
      → register names (RFC 0010)
      → enqueue work (RFC 0007)
      → WorkerLoop spawns sandboxed subprocesses (RFC 0008/0016/0017)
      → ledger charges + first_breach signals
      → observability summary
      → clean shutdown

Run with::

    python -m examples.kernel_e2e_smoke

Exits 0 on success. Self-contained — uses a tmp dir for kernel.db,
no daemon, no LLM.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

from cheetahclaws import kernel
# Module-level classes referenced inside _run_demo, where the `kernel`
# param holds a Kernel *instance* and would otherwise shadow the module.
from cheetahclaws.kernel import ScheduleSpec, SandboxPolicy


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "kernel.db"
        # NB: bind to `kern`, not `kernel` — `as kernel` would shadow the
        # `kernel` module on this same line (UnboundLocalError).
        with kernel.Kernel.open(db_path) as kern:
            return _run_demo(kern)


def _run_demo(kernel: kernel.Kernel) -> int:
    # ── Step 1: create agents ────────────────────────────────────────
    parent = kernel.create_agent(
        name="orchestrator", template="demo/parent",
    )
    child_a = kernel.create_agent(
        name="worker-a", template="demo/leaf",
        parent_pid=parent.pid,
    )
    child_b = kernel.create_agent(
        name="worker-b", template="demo/leaf",
        parent_pid=parent.pid,
    )
    print(f"[1] Created agents: parent={parent.pid}, "
          f"workers={child_a.pid},{child_b.pid}")

    # ── Step 2: capabilities (parent permissive, children narrow) ──
    kernel.cap.create(
        pid=parent.pid,
        tool_grants=["*"],                 # all tools (parent)
        net_grants=["*"],
        model_grants=["*"],
        sub_agent=True,
    )
    for c in (child_a, child_b):
        kernel.cap.derive(
            parent_pid=parent.pid, child_pid=c.pid,
            tool_grants=["Read"],          # tightened
            net_grants=[],
            model_grants=[],
            sub_agent=False,
        )
    print(f"[2] Capabilities: parent='*', children='Read'-only, "
          f"sub_agent=parent only")

    # ── Step 3: ledger budgets ──────────────────────────────────────
    for c in (child_a, child_b):
        kernel.ledger.create(
            pid=c.pid,
            grants={"tokens": 10_000, "wall_s": 60, "tool_calls": 50},
        )
    print(f"[3] Ledger budgets: tokens=10000, wall_s=60, tool_calls=50 "
          f"per child")

    # ── Step 4: register names ──────────────────────────────────────
    kernel.registry.register(
        name="/agents/demo/parent", pid=parent.pid,
        tags=["demo", "parent"],
    )
    kernel.registry.register(
        name="/agents/demo/worker-a", pid=child_a.pid, tags=["demo"],
    )
    kernel.registry.register(
        name="/agents/demo/worker-b", pid=child_b.pid, tags=["demo"],
    )
    looked = kernel.registry.lookup("/agents/demo/worker-a")
    print(f"[4] Registry: lookup '/agents/demo/worker-a' → pid={looked.pid}")

    # ── Step 5: enqueue work for the workers ────────────────────────
    sids = []
    for c in (child_a, child_b):
        sid = kernel.scheduler.enqueue(ScheduleSpec(
            pid=c.pid,
            priority=1,
            trigger="manual",
            payload={"task": "demo"},
        ))
        sids.append(sid)
    print(f"[5] Enqueued {len(sids)} work items: sched_ids={sids}")

    # ── Step 6: build worker loop (spawns echo runner) ──────────────
    worker = kernel.make_worker(
        argv_factory=lambda entry: [
            sys.executable, "-m", "cheetahclaws.kernel.runner.runner_main",
        ],
        policy_factory=lambda entry: SandboxPolicy(
            wall_seconds=10,
            cpu_seconds=5,
        ),
        env_factory=lambda entry: {
            **os.environ,
            "CC_RUNNER_BEHAVIOR": "echo",
        },
        max_concurrent=2,
        poll_interval_s=0.1,
        wait_timeout_s=30,
    )
    print(f"[6] WorkerLoop ready (max_concurrent=2)")

    # ── Step 7: drive to completion ─────────────────────────────────
    worker.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        queued, _    = kernel.scheduler.list(state="queued")
        dispatched, _= kernel.scheduler.list(state="dispatched")
        if not queued and not dispatched:
            break
        time.sleep(0.1)
    else:
        print("[!] timeout waiting for work to drain", file=sys.stderr)
        return 1
    worker.stop(drain=True, drain_timeout_s=5)
    print(f"[7] Worker drained; {len(sids)} entries completed")

    # ── Step 8: observability check ─────────────────────────────────
    info = kernel.info()
    summary_lines = [
        f"  schema_version: {info['schema_version']}",
        f"  agents.total:   {info['agents']['total']}",
        f"  agents.DEAD:    {info['agents']['DEAD']}",
        f"  events.total:   {info['events']['total']}",
        f"  scheduler.completed: {info['scheduler']['completed']}",
        f"  registry.entries: {info['registry']['entries']}",
    ]
    print("[8] Observability summary:")
    print("\n".join(summary_lines))

    # ── Step 9: assertions (a runnable smoke is also a test) ────────
    expected = {
        "agents.DEAD":        2,         # both children completed
        "scheduler.completed": len(sids),
    }
    actual = {
        "agents.DEAD":        info["agents"]["DEAD"],
        "scheduler.completed": info["scheduler"]["completed"],
    }
    if actual != expected:
        print(f"[!] MISMATCH expected={expected}, got={actual}",
              file=sys.stderr)
        return 1
    # Each child has at least 3 events (created, transitioned to RUNNING,
    # transitioned to DEAD).
    for c in (child_a, child_b):
        events = kernel.process.events_tail(pid=c.pid, limit=100)
        kinds = [e.kind for e in events]
        assert "kernel.process.created" in kinds, kinds
        assert "kernel.process.transitioned" in kinds, kinds
        assert "kernel.process.terminated" in kinds, kinds
    print("[9] All assertions passed.")

    # ── Step 10: trace one of the chains ────────────────────────────
    last_event_id = info["events"]["max_event_id"]
    if last_event_id > 0:
        trace = kernel.observability.trace(last_event_id, depth=5)
        print(f"[10] Trace from event_id={last_event_id}: "
              f"{trace['depth']} hops, truncated={trace['truncated']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
