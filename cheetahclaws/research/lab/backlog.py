"""research/lab/backlog.py — multi-topic queue + single worker.

Two pieces:

1. ``BacklogManager`` — thin wrapper over the SQLite ``lab_backlog`` table
   (defined in :mod:`research.lab.storage`). Pure CRUD, no threads.

2. ``run_backlog_worker`` — a single-worker loop that claims the next
   pending item, runs ``/lab start``, optionally runs ``/lab iterate``,
   and updates the backlog row. Designed to be called from the daemon
   thread; safe to start/stop with a ``threading.Event``.

Concurrency model: **single worker per backlog**. Two daemons running
against the same DB will compete for the same row via
``claim_next_backlog`` (atomic pending → running update), so duplicate
work won't happen — but the loser's claim returns None and it sleeps.
v0 does not implement multi-worker fan-out; that's Phase B.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from . import storage as _storage
from .iterate import (
    IterationConfig,
    iterate_until_converged,
)
from .orchestrator import (
    CallLLM,
    Stage,
    _default_call_llm,
    run_one_lab_session,
)


# ── BacklogManager ────────────────────────────────────────────────────────


@dataclass
class BacklogManager:
    """Thin facade so the REPL doesn't have to import storage directly."""
    storage: _storage.LabStorage

    def add(self, topic: str, *,
            iterate: bool = False,
            target_score: Optional[float] = None,
            max_iterations: int = 5,
            priority: int = 0,
            notes: Optional[str] = None) -> int:
        return self.storage.add_backlog(
            topic=topic, iterate=iterate, target_score=target_score,
            max_iterations=max_iterations, priority=priority, notes=notes,
        )

    def list(self, status: Optional[str] = None, limit: int = 100):
        return self.storage.list_backlog(status=status, limit=limit)

    def remove(self, item_id: int) -> bool:
        return self.storage.remove_backlog(item_id)

    def reset_running(self) -> int:
        return self.storage.reset_running_backlog()


# ── Worker loop ────────────────────────────────────────────────────────────


def run_backlog_worker(*, config: dict,
                      stop_event: threading.Event,
                      poll_interval_s: float = 5.0,
                      storage_obj: Optional[_storage.LabStorage] = None,
                      call_llm: Optional[CallLLM] = None,
                      on_item_start: Optional[Callable[[dict], None]] = None,
                      on_item_finish: Optional[Callable[[dict, str], None]] = None,
                      ) -> None:
    """Block until ``stop_event`` is set, claiming + running backlog items.

    For each claimed item:

    * Start a fresh lab run on the topic (uses budget defaults from config).
    * If ``item.iterate`` is set, run ``iterate_until_converged`` after
      finalization with the item's ``target_score`` / ``max_iterations``.
    * Update the backlog row with status=done|failed and run_id linkage.

    Cancellation: the worker checks ``stop_event`` between items only,
    and again before kicking off iterate. We do **not** interrupt an
    in-flight ``run_one_lab_session`` mid-stage — the user can /lab abort
    that specific run_id if they need to.
    """
    storage = storage_obj or _storage.LabStorage()
    call = call_llm or _default_call_llm

    # On daemon startup, recover any items the previous daemon left in
    # 'running' state. This is safe because lab_runs themselves track
    # their own status; we just unstick the queue.
    storage.reset_running_backlog()

    while not stop_event.is_set():
        item = storage.claim_next_backlog()
        if item is None:
            # Queue empty — sleep with early wake-up on stop.
            stop_event.wait(timeout=poll_interval_s)
            continue

        if on_item_start:
            try:
                on_item_start(item)
            except Exception:
                pass

        run_id: Optional[str] = None
        item_status = "failed"
        notes = ""
        # Default progress sink — print every stage transition so the
        # REPL shows live activity without the user polling /lab status.
        # Caller can override via config["lab_daemon_silent"]=True.
        from pathlib import Path as _Path
        from cheetahclaws.ui.render import clr as _clr
        silent = bool(config.get("lab_daemon_silent", False))

        def _stage_pr(stage):
            # run_id closure is None until run_one_lab_session returns,
            # so we look up the active run via storage. Cheap (1 SELECT).
            if silent:
                return
            try:
                rid = run_id
                if not rid:
                    runs = storage.list_runs(status="running", limit=1)
                    rid = runs[0].run_id if runs else "?"
            except Exception:
                rid = "?"
            print(_clr(f"  ↳ /lab daemon  ► [{rid}] {stage.value}", "dim"))

        try:
            # Pre-print so the user knows where the eventual report lands
            # *before* the orchestrator starts producing.
            if not silent:
                topic_short = item["topic"][:60] + (
                    "…" if len(item["topic"]) > 60 else "")
                print(_clr(
                    f"  ↳ /lab daemon  ▶ starting backlog #{item['id']}: "
                    f"{topic_short}",
                    "cyan",
                ))

            # ── 1. Run the topic ──────────────────────────────────────
            run = run_one_lab_session(
                topic=item["topic"],
                config=config,
                storage_obj=storage,
                budget_tokens=int(config.get("lab_budget_tokens", 5_000_000)),
                budget_cost_cents=int(config.get("lab_budget_cost_cents", 5000)),
                max_rounds=int(config.get("lab_max_rounds", 5)),
                role_override=config.get("lab_role_override"),
                call_llm=call,
                cancel_check=stop_event.is_set,
                on_stage_change=_stage_pr,
            )
            run_id = run.state.run_id

            if not silent:
                from cheetahclaws.research.lab.storage import output_dir_for
                rec = storage.get_run(run_id)
                if rec is not None:
                    report_path = output_dir_for(
                        rec.run_id, rec.topic, rec.created_at
                    ) / "report.md"
                else:
                    report_path = (_Path.home() / ".cheetahclaws"
                                   / "research_papers" / run_id / "report.md")
                print(_clr(
                    f"  ↳ /lab daemon  📄 [{run_id}] will save to: "
                    f"{report_path}",
                    "dim",
                ))

            # Link the run early so /lab backlog list shows progress
            storage.update_backlog(item_id=item["id"], run_id=run_id)

            # ── 2. Iterate if requested ───────────────────────────────
            if item["iterate"] and not stop_event.is_set():
                iter_cfg = IterationConfig(
                    target_score=(item["target_score"]
                                  if item["target_score"] is not None
                                  else float(config.get(
                                      "lab_iterate_target", 7.0))),
                    max_iterations=item["max_iterations"],
                    plateau_eps=float(config.get(
                        "lab_iterate_plateau_eps", 0.3)),
                    plateau_consec=int(config.get(
                        "lab_iterate_plateau_consec", 2)),
                    n_reviewers=int(config.get("lab_iterate_reviewers", 3)),
                )
                history = iterate_until_converged(
                    run_id=run_id, config=config, iter_cfg=iter_cfg,
                    storage_obj=storage, call_llm=call,
                    cancel_check=stop_event.is_set,
                )
                if history:
                    last = history[-1]
                    notes = (f"final_score={last.score_avg:.2f} "
                             f"iters={len(history)} "
                             f"converged={'yes' if last.revise_stage is None else 'no'}")
                else:
                    notes = "iterate skipped (cancelled or score-fetch failed)"

            item_status = "done"
        except Exception as exc:
            notes = f"{type(exc).__name__}: {exc}"
            item_status = "failed"

        storage.update_backlog(
            item_id=item["id"], status=item_status,
            run_id=run_id, notes=notes, mark_ended=True,
        )
        if on_item_finish:
            try:
                on_item_finish(item, item_status)
            except Exception:
                pass


# ── Threaded daemon ───────────────────────────────────────────────────────


@dataclass
class DaemonHandle:
    """Returned by ``start_daemon`` so the REPL can stop / inspect it."""
    thread: threading.Thread
    stop_event: threading.Event
    started_at: float

    def stop(self, *, join_timeout_s: float = 30.0) -> None:
        self.stop_event.set()
        self.thread.join(timeout=join_timeout_s)

    @property
    def running(self) -> bool:
        return self.thread.is_alive()


_daemon_lock = threading.Lock()
_daemon_handle: Optional[DaemonHandle] = None


def start_daemon(*, config: dict,
                 storage_obj: Optional[_storage.LabStorage] = None,
                 ) -> DaemonHandle:
    """Start the singleton backlog worker. Idempotent."""
    global _daemon_handle
    with _daemon_lock:
        if _daemon_handle is not None and _daemon_handle.running:
            return _daemon_handle
        stop = threading.Event()
        t = threading.Thread(
            target=run_backlog_worker,
            kwargs={
                "config": config,
                "stop_event": stop,
                "storage_obj": storage_obj,
            },
            name="lab-daemon", daemon=True,
        )
        t.start()
        _daemon_handle = DaemonHandle(
            thread=t, stop_event=stop, started_at=time.time(),
        )
        return _daemon_handle


def stop_daemon(*, join_timeout_s: float = 30.0) -> bool:
    """Stop the singleton daemon. Returns True if one was actually running."""
    global _daemon_handle
    with _daemon_lock:
        if _daemon_handle is None:
            return False
        h = _daemon_handle
        _daemon_handle = None
    h.stop(join_timeout_s=join_timeout_s)
    return True


def get_daemon() -> Optional[DaemonHandle]:
    return _daemon_handle
