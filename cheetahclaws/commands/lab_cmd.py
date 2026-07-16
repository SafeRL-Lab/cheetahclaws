"""commands/lab_cmd.py — `/lab` slash command for the research lab.

Subcommands:

  /lab start <topic>             Spawn a research run in a background thread.
  /lab status                    List all runs and their stage / budget.
  /lab status <run_id>           Detailed status for one run, with last messages.
  /lab abort <run_id>            Request cancellation; current stage finishes.
  /lab resume <run_id> [<stage>] Resume a paused/aborted/done run, optionally
                                 rewinding to a specific stage.
  /lab iterate <run_id>          Score the final report and re-run the weakest
                                 stage; loops until target / max / plateau.
  /lab logs <run_id>             Print the last N agent messages.
  /lab backlog add <topic> [--iterate] [--target=N] [--max=N] [--prio=N]
  /lab backlog list / remove <id> / clear
  /lab daemon start / stop / status
                                 Run pending backlog items 24/7 in a worker.

The orchestrator runs on a daemon thread per run. Cancellation is
cooperative: the orchestrator polls a per-run cancel flag between
stages and rounds.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

from cheetahclaws.ui.render import clr, info, ok, warn, err


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
        return _cmd_resume(rest, config)
    if sub == "iterate":
        return _cmd_iterate(rest, config)
    if sub == "backlog":
        return _cmd_backlog(rest, config)
    if sub == "daemon":
        return _cmd_daemon(rest, config)
    if sub == "models":
        return _cmd_models(rest, config)
    if sub == "migrate-paths":
        return _cmd_migrate_paths(rest, config)
    if sub in ("help", "?", "-h", "--help"):
        _print_usage()
        return True
    err(f"Unknown /lab subcommand: {sub!r}")
    _print_usage()
    return True


def _print_usage() -> None:
    print(clr("/lab — autonomous research lab", "cyan", "bold"))
    print(
        "  /lab start <topic>             Start a new research run\n"
        "  /lab status [<run_id>]         Show run(s) status\n"
        "  /lab abort <run_id>            Request cancellation\n"
        "  /lab logs <run_id> [n]         Print last N agent messages\n"
        "  /lab resume <run_id> [<stage>] Continue a run; optionally rewind\n"
        "  /lab iterate <run_id>          Score + revise loop until target / max\n"
        "  /lab backlog add <topic> ...   Queue a topic\n"
        "  /lab backlog list|remove|clear Manage queue\n"
        "  /lab daemon start|stop|status  24/7 worker pulling from backlog\n"
        "  /lab models                    Show effective per-role model assignment\n"
        "  /lab migrate-paths [--apply]   Rename legacy lab_xxx/ dirs to human-readable form\n"
    )


# ── start ─────────────────────────────────────────────────────────────────


def _cmd_start(topic: str, config: dict) -> bool:
    topic = topic.strip()
    if not topic:
        err("Usage: /lab start <topic>")
        return True
    from cheetahclaws.research.lab.orchestrator import run_one_lab_session
    from cheetahclaws.research.lab.storage import LabStorage
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

    from cheetahclaws.research.lab.storage import output_dir_for
    out_dir = output_dir_for(rec.run_id, rec.topic, rec.created_at)
    report_path = out_dir / "report.md"

    def _on_stage_change(stage):
        # Live-print stage transitions so the user can see progress in
        # the REPL without polling /lab status.
        print(clr(f"  ↳ /lab {rec.run_id}  ► {stage.value}", "dim"))

    def _runner():
        # Re-create the run inside the worker so we can pass cancel_check.
        from cheetahclaws.research.lab.orchestrator import _drive, LabRun, LabState, Stage
        from cheetahclaws.research.lab.roles import build_default_assignment
        from cheetahclaws.research.lab.convergence import ConvergenceConfig
        roles = build_default_assignment(config, override=role_override)
        state = LabState(run_id=rec.run_id, topic=topic, stage=Stage.QUESTIONING)
        run = LabRun(
            state=state, storage=storage, roles=roles, config=config,
            convergence=ConvergenceConfig(max_rounds=max_rounds),
            on_stage_change=_on_stage_change,
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
                print(clr(f"\n  ✓ /lab {rec.run_id}: done. "
                          f"Report → {report_path}", "green"))
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
    info(f"  report path : {report_path}")
    info(f"  watch live  : stage transitions print here as they happen")
    info(f"  poll        : /lab status {rec.run_id}")
    info(f"  details     : /lab logs {rec.run_id}")
    info(f"  abort       : /lab abort {rec.run_id}")
    return True


# ── status ────────────────────────────────────────────────────────────────


def _cmd_status(arg: str) -> bool:
    from cheetahclaws.research.lab.storage import LabStorage
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
    from cheetahclaws.research.lab.storage import output_dir_for, DEFAULT_OUTPUT_DIR
    new_dir = output_dir_for(rec.run_id, rec.topic, rec.created_at)
    legacy_dir = DEFAULT_OUTPUT_DIR / rec.run_id
    # Prefer the human-readable path; if a legacy run already wrote
    # files there, surface that path instead so the user can find them.
    if new_dir.exists():
        out_dir = new_dir
    elif legacy_dir.exists():
        out_dir = legacy_dir
    else:
        out_dir = new_dir   # path the run *will* use when it finalises
    report = out_dir / "report.md"
    print(clr(f"run {rec.run_id}", "cyan", "bold"))
    print(f"  topic        : {rec.topic}")
    print(f"  status       : {rec.status}")
    print(f"  stage        : {rec.current_stage or '—'}")
    print(f"  output dir   : {out_dir}")
    print(f"  report.md    : {report} {'(exists)' if report.exists() else '(pending)'}")
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
    from cheetahclaws.research.lab.storage import LabStorage
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


# ── resume ────────────────────────────────────────────────────────────────


def _cmd_resume(arg: str, config: dict) -> bool:
    """`/lab resume <run_id> [<stage>]` — continue or rewind a run."""
    parts = arg.strip().split()
    if not parts:
        err("Usage: /lab resume <run_id> [<stage>]")
        info("       stage ∈ {questioning, survey, outline, implementation, "
             "experiment, analysis, drafting, verification, finalization}")
        return True
    run_id = parts[0]
    stage_arg = parts[1].lower() if len(parts) > 1 else ""

    from cheetahclaws.research.lab import resume as _resume
    from cheetahclaws.research.lab.orchestrator import Stage
    from cheetahclaws.research.lab.storage import LabStorage

    storage = LabStorage()
    rec = storage.get_run(run_id)
    if rec is None:
        err(f"No such run: {run_id}")
        return True

    target_stage = None
    if stage_arg:
        try:
            target_stage = Stage(stage_arg)
        except ValueError:
            valid = ", ".join(s.value for s in Stage)
            err(f"Invalid stage {stage_arg!r}. Valid: {valid}")
            return True

    cancel = threading.Event()
    _cancel_flags[run_id] = cancel

    def _on_finish(success: bool, msg: str) -> None:
        if success:
            ok(f"\n  ✓ /lab resume {run_id}: {msg}")
        else:
            err(f"\n  ✗ /lab resume {run_id} failed: {msg}")

    t, _ = _resume.resume_run_in_thread(
        run_id=run_id, config=config, start_stage=target_stage,
        on_finish=_on_finish,
    )
    # Wire the same cancel event the user-facing /lab abort uses.  The
    # resume thread we just spawned has its own cancel; replace it with
    # ours so abort works against this resume too.
    _run_threads[run_id] = t

    label = f"from {target_stage.value}" if target_stage else "from saved stage"
    ok(f"Resuming lab run {run_id} {label}")
    info(f"  topic   : {rec.topic}")
    info(f"  status  : /lab status {run_id}")
    info(f"  abort   : /lab abort {run_id}")
    return True


# ── iterate ───────────────────────────────────────────────────────────────


def _cmd_iterate(arg: str, config: dict) -> bool:
    """`/lab iterate <run_id> [--target=N] [--max=N]` — meta-loop."""
    parts = arg.strip().split()
    if not parts:
        err("Usage: /lab iterate <run_id> [--target=7.0] [--max=5]")
        return True
    run_id = parts[0]
    target = None
    max_iter = None
    for tok in parts[1:]:
        if tok.startswith("--target="):
            try: target = float(tok.split("=", 1)[1])
            except ValueError: pass
        elif tok.startswith("--max="):
            try: max_iter = int(tok.split("=", 1)[1])
            except ValueError: pass

    from cheetahclaws.research.lab import iterate as _it
    from cheetahclaws.research.lab.storage import LabStorage

    storage = LabStorage()
    rec = storage.get_run(run_id)
    if rec is None:
        err(f"No such run: {run_id}")
        return True
    if storage.get_latest_artifact(run_id, "report") is None:
        err(f"Run {run_id} has not produced a 'report' artifact yet — "
            "iterate is only valid after the run finalised.")
        info("If the run is still mid-flight: /lab status " + run_id)
        return True

    iter_cfg = _it.IterationConfig(
        target_score=(target if target is not None
                      else float(config.get("lab_iterate_target", 7.0))),
        max_iterations=(max_iter if max_iter is not None
                        else int(config.get("lab_iterate_max", 5))),
        plateau_eps=float(config.get("lab_iterate_plateau_eps", 0.3)),
        plateau_consec=int(config.get("lab_iterate_plateau_consec", 2)),
        n_reviewers=int(config.get("lab_iterate_reviewers", 3)),
    )

    cancel = threading.Event()
    _cancel_flags[run_id] = cancel

    def _on_iter(result) -> None:
        msg = (f"  ↳ iter {result.iter_n}: avg={result.score_avg:.2f} "
               f"Δ={result.delta:+.2f}")
        if result.revise_stage:
            msg += f"  weakest→{result.revise_stage.value}"
        else:
            msg += "  ✓ converged"
        info(msg)

    def _on_finish(history) -> None:
        if not history:
            warn(f"\n  /lab iterate {run_id}: no iterations completed")
            return
        last = history[-1]
        verdict = "converged" if last.revise_stage is None else "stopped"
        ok(f"\n  ✓ /lab iterate {run_id}: {verdict} after {len(history)} iter(s); "
           f"final avg={last.score_avg:.2f}")

    t, _ = _it.iterate_in_thread(
        run_id=run_id, config=config, iter_cfg=iter_cfg,
        on_finish=_on_finish,
    )
    _run_threads[run_id] = t

    ok(f"Iterating lab run {run_id}")
    info(f"  target  : ≥ {iter_cfg.target_score:.1f}")
    info(f"  max     : {iter_cfg.max_iterations} iteration(s)")
    info(f"  abort   : /lab abort {run_id}")
    return True


# ── backlog ───────────────────────────────────────────────────────────────


def _cmd_backlog(arg: str, config: dict) -> bool:
    parts = arg.strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    from cheetahclaws.research.lab.backlog import BacklogManager
    from cheetahclaws.research.lab.storage import LabStorage
    mgr = BacklogManager(LabStorage())

    if sub == "add":
        return _backlog_add(mgr, rest, config)
    if sub == "list":
        return _backlog_list(mgr)
    if sub == "remove":
        return _backlog_remove(mgr, rest)
    if sub == "clear":
        return _backlog_clear(mgr)
    err(f"Unknown /lab backlog subcommand: {sub!r}")
    info("Use: add, list, remove <id>, clear")
    return True


def _backlog_add(mgr, rest: str, config: dict) -> bool:
    if not rest:
        err("Usage: /lab backlog add <topic> [--iterate] [--target=N] "
            "[--max=N] [--prio=N]")
        return True
    # Strip flags from the topic. Flags are only honoured *after* the
    # topic (or quoted) — once we see the first --flag, everything from
    # there is treated as flags, and any non-flag token after that is a
    # typo (not silently appended to the topic).
    iterate_flag = False
    target_score = None
    max_iter = 5
    priority = 0
    tokens = rest.split()
    topic_words: list[str] = []
    in_flags = False
    unknown: list[str] = []
    for tok in tokens:
        if tok.startswith("--"):
            in_flags = True
            if tok == "--iterate":
                iterate_flag = True
            elif tok.startswith("--target="):
                try: target_score = float(tok.split("=", 1)[1])
                except ValueError: unknown.append(tok)
            elif tok.startswith("--max="):
                try: max_iter = max(1, int(tok.split("=", 1)[1]))
                except ValueError: unknown.append(tok)
            elif tok.startswith("--prio="):
                try: priority = int(tok.split("=", 1)[1])
                except ValueError: unknown.append(tok)
            else:
                unknown.append(tok)
        elif in_flags:
            # A bare word after the flag block is almost always a typo
            # (e.g. user accidentally typed `... --max=5 start`). Reject
            # explicitly rather than silently merging into the topic.
            unknown.append(tok)
        else:
            topic_words.append(tok)
    if unknown:
        err(f"Unknown tokens after flags: {' '.join(unknown)}")
        info("Flags allowed: --iterate, --target=N, --max=N, --prio=N")
        info("Place the topic FIRST (quoted if it contains spaces), "
             "then flags last.")
        return True
    topic = " ".join(topic_words).strip().strip('"').strip("'")
    if not topic:
        err("Empty topic after stripping flags.")
        return True
    if iterate_flag and target_score is None:
        target_score = float(config.get("lab_iterate_target", 7.0))
    item_id = mgr.add(
        topic=topic, iterate=iterate_flag,
        target_score=target_score, max_iterations=max_iter,
        priority=priority,
    )
    ok(f"Queued #{item_id}: {topic}"
       + (f"  (iterate→{target_score:.1f}, max={max_iter})"
          if iterate_flag else ""))
    info("Use /lab daemon start to begin processing the queue.")
    return True


def _backlog_list(mgr) -> bool:
    items = mgr.list()
    if not items:
        info("Backlog is empty.")
        return True
    print(clr(f"backlog ({len(items)} item(s)):", "cyan", "bold"))
    print(f"  {'id':>4} {'status':<8} {'prio':>4} {'iter':<5} "
          f"{'topic':<40} {'run_id':<18}")
    for x in items:
        topic = (x['topic'][:37] + "…") if len(x['topic']) > 38 else x['topic']
        rid = x['run_id'] or "—"
        it = "yes" if x['iterate'] else "no"
        print(f"  #{x['id']:>3} {x['status']:<8} {x['priority']:>4} "
              f"{it:<5} {topic:<40} {rid}")
    return True


def _backlog_remove(mgr, arg: str) -> bool:
    arg = arg.strip()
    if not arg.isdigit():
        err("Usage: /lab backlog remove <id>")
        return True
    if mgr.remove(int(arg)):
        ok(f"Removed backlog #{arg}")
    else:
        err(f"No backlog item with id={arg}")
    return True


def _backlog_clear(mgr) -> bool:
    items = mgr.list()
    if not items:
        info("Backlog already empty.")
        return True
    n = 0
    for it in items:
        if it["status"] in ("pending", "skipped", "failed"):
            mgr.remove(it["id"]); n += 1
    ok(f"Cleared {n} pending/skipped/failed item(s); "
       f"running/done items kept for audit.")
    return True


# ── daemon ────────────────────────────────────────────────────────────────


def _cmd_daemon(arg: str, config: dict) -> bool:
    sub = (arg.strip().split() or ["status"])[0].lower()
    from cheetahclaws.research.lab import backlog as _bl

    if sub == "start":
        h = _bl.start_daemon(config=config)
        if h.thread.is_alive():
            from cheetahclaws.research.lab.storage import LabStorage
            pending = LabStorage().list_backlog(status="pending")
            ok("Lab daemon running. Pulls from /lab backlog continuously.")
            info(f"  started_at : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(h.started_at))}")
            info(f"  pending    : {len(pending)} item(s) in queue")
            info(f"  reports → ~/.cheetahclaws/research_papers/<run_id>/report.md")
            info(f"  watch live : stage transitions print here as the daemon runs")
            info(f"  poll       : /lab status   |   /lab status <run_id>")
            info(f"  abort run  : /lab abort <run_id>")
            info(f"  stop daemon: /lab daemon stop")
        else:
            err("Daemon failed to start.")
        return True
    if sub == "stop":
        was_running = _bl.stop_daemon()
        (ok if was_running else info)(
            "Lab daemon stopped." if was_running else "No daemon was running."
        )
        return True
    if sub in ("status", ""):
        h = _bl.get_daemon()
        if h is None or not h.running:
            info("Lab daemon: not running.  Start with /lab daemon start")
            return True
        ago = time.time() - h.started_at
        ok(f"Lab daemon: running (uptime {int(ago)//60}m{int(ago)%60}s)")
        return True
    err(f"Unknown /lab daemon subcommand: {sub!r}.  Use: start, stop, status")
    return True


# ── migrate-paths (rename legacy lab_xxx/ dirs to human-readable form) ──


def _cmd_migrate_paths(arg: str, config: dict) -> bool:
    """`/lab migrate-paths [--dry-run]` — rename legacy ``lab_xxx/``
    output directories to ``<date>_<time>_<topic-slug>_<short>``.

    Idempotent: a directory already in the new format is ignored.
    Default is **dry-run** so the user can review before committing —
    pass ``--apply`` to actually rename.
    """
    from cheetahclaws.research.lab.storage import (
        DEFAULT_OUTPUT_DIR, LabStorage, output_dir_for,
    )

    apply = "--apply" in arg.split()
    if not apply:
        info("Dry run — pass `--apply` to actually rename.")
    storage = LabStorage()
    if not DEFAULT_OUTPUT_DIR.exists():
        info("No research_papers/ directory yet.")
        return True

    plan: list[tuple[Path, Path]] = []
    skipped_unknown: list[Path] = []
    for entry in sorted(DEFAULT_OUTPUT_DIR.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("lab_"):
            # Already in new format (or unrelated dir we shouldn't touch).
            continue
        rec = storage.get_run(name)
        if rec is None:
            skipped_unknown.append(entry)
            continue
        new_dir = output_dir_for(rec.run_id, rec.topic, rec.created_at)
        if new_dir.exists() and new_dir.resolve() != entry.resolve():
            # Conflict — both old and new exist. Skip rather than risk loss.
            warn(f"  ✗ {name}: target {new_dir.name} already exists, skip")
            continue
        plan.append((entry, new_dir))

    if not plan and not skipped_unknown:
        ok("Nothing to migrate.")
        return True

    print(clr(f"\n  Plan ({len(plan)} dir(s)):", "cyan", "bold"))
    for src, dst in plan:
        print(f"    {src.name}")
        print(clr(f"      → {dst.name}", "dim"))

    if skipped_unknown:
        print(clr(f"\n  Skipped {len(skipped_unknown)} unknown dir(s)"
                  " (no matching run in DB):", "yellow"))
        for s in skipped_unknown[:5]:
            print(f"    {s.name}")

    if not apply:
        info("\nRun `/lab migrate-paths --apply` to rename.")
        return True

    n = 0
    for src, dst in plan:
        try:
            src.rename(dst)
            n += 1
        except OSError as e:
            err(f"  ✗ {src.name}: {e}")
    ok(f"Migrated {n}/{len(plan)} dir(s).")
    return True


# ── models (per-role model inspection) ────────────────────────────────────


def _cmd_models(_arg: str, config: dict) -> bool:
    """`/lab models` — show which model each of the 9 roles will use.

    Resolution priority (per role): config['lab_role_override'][<role>]
    if set, else family auto-pick driven by which API keys are in env,
    else config['model'] fallback.  This view exists so the user can
    confirm reviewers really span 3 different families before kicking
    off a multi-day daemon (homogeneous review = same-source bias).
    """
    import os
    from cheetahclaws.research.lab.roles import build_default_assignment

    override = config.get("lab_role_override") or {}
    assignment = build_default_assignment(config, override=override)

    # Map model → likely env-var that selected it (best-effort label).
    model_family: dict[str, str] = {}
    for prefix, env in [
        ("claude-",      "ANTHROPIC_API_KEY"),
        ("gpt-",         "OPENAI_API_KEY"),
        ("o1-",          "OPENAI_API_KEY"),
        ("o3-",          "OPENAI_API_KEY"),
        ("o4-",          "OPENAI_API_KEY"),
        ("gemini",       "GEMINI_API_KEY"),
        ("deepseek",     "DEEPSEEK_API_KEY"),
        ("qwen",         "DASHSCOPE_API_KEY"),
        ("zhipu",        "ZHIPU_API_KEY"),
        ("glm-",         "ZHIPU_API_KEY"),
        ("kimi",         "MOONSHOT_API_KEY"),
        ("moonshot",     "MOONSHOT_API_KEY"),
        ("ollama/",      ""),
        ("lmstudio/",    ""),
    ]:
        model_family[prefix] = env

    def _label(model: str) -> str:
        for prefix, env in model_family.items():
            if model.startswith(prefix):
                if not env:
                    return "local"
                ok = "✓" if os.environ.get(env) else "✗"
                return f"{env} {ok}"
        return "?"

    rows: list[tuple[str, str, str, str]] = []
    rows.append(("pi",           assignment.pi.model,        _label(assignment.pi.model),         "ties + direction"))
    rows.append(("questioner",   assignment.questioner.model, _label(assignment.questioner.model), "RQ generation"))
    rows.append(("surveyor",     assignment.surveyor.model,   _label(assignment.surveyor.model),   "lit search + gap"))
    rows.append(("designer",     assignment.designer.model,   _label(assignment.designer.model),   "methodology"))
    rows.append(("engineer",     assignment.engineer.model,   _label(assignment.engineer.model),   "experiment code"))
    rows.append(("analyst",      assignment.analyst.model,    _label(assignment.analyst.model),    "results section"))
    rows.append(("writer",       assignment.writer.model,     _label(assignment.writer.model),     "paper body"))
    for r in assignment.reviewers:
        rows.append((r.name,     r.model,                      _label(r.model),                     "independent review"))
    rows.append(("lay_reader",   assignment.lay_reader.model, _label(assignment.lay_reader.model), "clarity check"))

    print(clr("/lab models — effective per-role assignment", "cyan", "bold"))
    print(f"  {'role':<14} {'model':<30} {'env-key':<22} note")
    for role, model, env_label, note in rows:
        is_override = role in override
        marker = clr("●", "yellow") if is_override else " "
        print(f"  {marker} {role:<12} {model:<30} {env_label:<22} {clr(note, 'dim')}")
    print()
    if override:
        info(f"  ● = manually overridden via lab_role_override "
             f"({len(override)} role{'s' if len(override) != 1 else ''})")
    else:
        info("  All roles using auto-selected defaults. Override via:")
        info("    /config lab_role_override={\"writer\": \"claude-opus-4-6\", "
             "\"reviewer_1\": \"gpt-4o\"}")

    # Reviewer diversity warning.
    rev_models = [r.model for r in assignment.reviewers]
    rev_families = set()
    for m in rev_models:
        for prefix in model_family:
            if m.startswith(prefix):
                rev_families.add(prefix); break
    if len(rev_families) < len(rev_models):
        warn(f"  Reviewers span only {len(rev_families)} model "
             f"famil{'ies' if len(rev_families) != 1 else 'y'}; "
             f"homogeneous review reduces meta-loop signal. Set more API "
             f"keys (Anthropic / OpenAI / Gemini / DeepSeek / Qwen) for diversity.")
    return True
