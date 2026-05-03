"""commands/lab_cmd.py — `/lab` slash command for the research lab.

Subcommands:

  /lab start <topic>            Spawn a research run in a background thread.
  /lab status                   List all runs and their stage / budget.
  /lab status <run_id>          Detailed status for one run, with last messages.
  /lab abort <run_id>           Request cancellation; current stage finishes.
  /lab resume <run_id>          (placeholder — Phase 2; v0 doesn't resume)
  /lab logs <run_id>            Print the last N agent messages.

The actual orchestrator runs on a daemon thread per run. Cancellation
is cooperative: the orchestrator polls a per-run cancel flag between
stages and rounds.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

from ui.render import clr, info, ok, warn, err


# Per-run cancel flags.  Run-id → threading.Event.
_cancel_flags: dict[str, threading.Event] = {}
_run_threads: dict[str, threading.Thread] = {}


def cmd_lab(args: str, _state, config) -> bool:
    parts = args.strip().split(None, 1)
    if not parts:
        _print_usage()
        return True
    sub, rest = parts[0], (parts[1] if len(parts) > 1 else "")
    if sub == "start":
        return _cmd_start(rest, config)
    if sub == "status":
        return _cmd_status(rest)
    if sub == "abort":
        return _cmd_abort(rest)
    if sub == "logs":
        return _cmd_logs(rest)
    if sub == "resume":
        return _cmd_resume(rest)
    if sub in ("help", "?", "-h", "--help"):
        _print_usage()
        return True
    err(f"Unknown /lab subcommand: {sub!r}")
    _print_usage()
    return True


def _print_usage() -> None:
    print(clr("/lab — autonomous research lab", "cyan", "bold"))
    print(
        "  /lab start <topic>      Start a new research run\n"
        "  /lab status [<run_id>]  Show run(s) status\n"
        "  /lab abort <run_id>     Request cancellation\n"
        "  /lab logs <run_id>      Print recent agent messages\n"
        "  /lab resume <run_id>    (Phase 2 — placeholder for now)\n"
    )


# ── start ─────────────────────────────────────────────────────────────────


def _cmd_start(topic: str, config: dict) -> bool:
    topic = topic.strip()
    if not topic:
        err("Usage: /lab start <topic>")
        return True
    from research.lab.orchestrator import run_one_lab_session
    from research.lab.storage import LabStorage
    storage = LabStorage()

    # Read budget overrides from config (with sensible defaults).
    budget_tokens = int(config.get("lab_budget_tokens", 5_000_000))
    budget_cost_cents = int(config.get("lab_budget_cost_cents", 5000))
    max_rounds = int(config.get("lab_max_rounds", 5))
    role_override = config.get("lab_role_override") or {}

    # Pre-allocate the run record so the user gets a run_id immediately,
    # then run the orchestrator in a background thread.
    rec = storage.create_run(
        topic=topic,
        budget_tokens=budget_tokens,
        budget_cost_cents=budget_cost_cents,
        max_rounds=max_rounds,
    )
    cancel = threading.Event()
    _cancel_flags[rec.run_id] = cancel

    def _runner():
        # Re-create the run inside the worker so we can pass cancel_check.
        from research.lab.orchestrator import _drive, LabRun, LabState, Stage
        from research.lab.roles import build_default_assignment
        from research.lab.convergence import ConvergenceConfig
        roles = build_default_assignment(config, override=role_override)
        state = LabState(run_id=rec.run_id, topic=topic, stage=Stage.QUESTIONING)
        run = LabRun(
            state=state, storage=storage, roles=roles, config=config,
            convergence=ConvergenceConfig(max_rounds=max_rounds),
        )
        storage.update_run_status(rec.run_id, "running",
                                   current_stage=state.stage.value)
        try:
            _drive(run, cancel_check=cancel.is_set)
            if state.cancel_requested:
                storage.update_run_status(rec.run_id, "aborted",
                                          current_stage=state.stage.value)
                print(clr(f"\n  ✗ /lab {rec.run_id}: aborted at "
                          f"{state.stage.value}", "yellow"))
            else:
                storage.update_run_status(rec.run_id, "done",
                                          current_stage=state.stage.value)
                out = (Path.home() / ".cheetahclaws" / "research_papers"
                        / rec.run_id / "report.md")
                print(clr(f"\n  ✓ /lab {rec.run_id}: done. "
                          f"Report → {out}", "green"))
        except Exception as exc:
            storage.update_run_status(rec.run_id, "failed",
                                      current_stage=state.stage.value,
                                      error=str(exc))
            print(clr(f"\n  ✗ /lab {rec.run_id}: failed: {exc}", "red"))

    t = threading.Thread(target=_runner, name=f"lab-{rec.run_id}", daemon=True)
    _run_threads[rec.run_id] = t
    t.start()
    ok(f"Started lab run {rec.run_id}")
    info(f"  topic       : {topic}")
    info(f"  budget      : {budget_tokens:,} tokens / ${budget_cost_cents/100:.2f}")
    info(f"  max_rounds  : {max_rounds} per stage")
    info(f"  status      : /lab status {rec.run_id}")
    info(f"  abort       : /lab abort {rec.run_id}")
    return True


# ── status ────────────────────────────────────────────────────────────────


def _cmd_status(arg: str) -> bool:
    from research.lab.storage import LabStorage
    storage = LabStorage()
    arg = arg.strip()
    if arg:
        rec = storage.get_run(arg)
        if rec is None:
            err(f"No such run: {arg}")
            return True
        _print_run_detail(rec, storage)
        return True
    runs = storage.list_runs(limit=20)
    if not runs:
        info("No lab runs yet. Try: /lab start <topic>")
        return True
    print(clr("recent /lab runs:", "cyan", "bold"))
    print(f"  {'run_id':<18} {'status':<10} {'stage':<14} {'topic':<40}")
    for r in runs:
        topic = (r.topic[:37] + "…") if len(r.topic) > 38 else r.topic
        stage = r.current_stage or "—"
        print(f"  {r.run_id:<18} {r.status:<10} {stage:<14} {topic}")
    return True


def _print_run_detail(rec, storage) -> None:
    print(clr(f"run {rec.run_id}", "cyan", "bold"))
    print(f"  topic        : {rec.topic}")
    print(f"  status       : {rec.status}")
    print(f"  stage        : {rec.current_stage or '—'}")
    tok, cents = storage.get_budget(rec.run_id)
    print(f"  tokens       : {tok:,} / {rec.budget_tokens:,}"
          if rec.budget_tokens else f"  tokens       : {tok:,} / unlimited")
    print(f"  cost         : ${cents/100:.2f} / ${(rec.budget_cost_cents or 0)/100:.2f}"
          if rec.budget_cost_cents else f"  cost         : ${cents/100:.2f} / unlimited")
    if rec.error:
        print(clr(f"  error        : {rec.error}", "red"))
    stages = storage.list_stages(rec.run_id)
    if stages:
        print(clr("  stages:", "dim"))
        for s in stages:
            dur = ""
            if s.ended_at and s.started_at:
                dur = f" ({s.ended_at - s.started_at:.1f}s)"
            outcome = s.outcome or "pending"
            print(f"    {s.stage:<14} round={s.round} {outcome}{dur}")


# ── abort ─────────────────────────────────────────────────────────────────


def _cmd_abort(arg: str) -> bool:
    arg = arg.strip()
    if not arg:
        err("Usage: /lab abort <run_id>")
        return True
    flag = _cancel_flags.get(arg)
    if flag is None:
        warn(f"No active in-process run matching {arg}; "
             f"if it's still in storage, edit status manually.")
        return True
    flag.set()
    ok(f"Cancellation requested for {arg}; current stage will finish then stop.")
    return True


# ── logs ──────────────────────────────────────────────────────────────────


def _cmd_logs(arg: str) -> bool:
    from research.lab.storage import LabStorage
    args = arg.strip().split()
    if not args:
        err("Usage: /lab logs <run_id> [n]")
        return True
    run_id = args[0]
    n = int(args[1]) if len(args) > 1 else 30
    storage = LabStorage()
    msgs = storage.list_messages(run_id, limit=n * 4)
    msgs = msgs[-n:]
    if not msgs:
        info(f"No messages for {run_id}.")
        return True
    print(clr(f"last {len(msgs)} messages for {run_id}:", "cyan", "bold"))
    for m in msgs:
        prefix = clr(f"[{m.stage}/r{m.round} {m.role} {m.kind}]", "dim")
        print(prefix)
        body = m.content
        if len(body) > 800:
            body = body[:800] + clr(f"\n  …+{len(m.content) - 800} more chars", "dim")
        print(body)
        print()
    return True


# ── resume placeholder ───────────────────────────────────────────────────


def _cmd_resume(arg: str) -> bool:
    warn("/lab resume is a Phase 2 feature — not implemented in v0.")
    info("v0 runs that crashed mid-stage need to be re-started with /lab start.")
    return True
