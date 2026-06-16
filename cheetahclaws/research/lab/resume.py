"""research/lab/resume.py — reconstruct LabState from SQLite and continue.

The orchestrator builds its in-memory ``LabState`` field-by-field as each
stage runs.  Every value it cares about is also persisted (artifacts table
for outputs, ``lab_experiments`` for sandbox runs, ``lab_runs.current_stage``
for progress).  This module reads those rows back into a ``LabState``
instance so a crashed / aborted / iterated run can pick up where it left
off (or be rolled back to an earlier stage on demand).

Public API:

    resume_run(run_id, config, *, start_stage=None, ...)  → LabRun

Resume entry points are idempotent at stage granularity:

* If ``start_stage`` is None, resume from ``runs.current_stage``.
* Otherwise we **rewind** to ``start_stage`` — this is what the meta-loop
  in :mod:`research.lab.iterate` calls to redo (e.g.) drafting after a
  reviewer panel scored the current report below target.

Intra-stage resume (mid review-loop, mid experiment debug) is **not**
implemented in v0; resume restarts the in-flight stage from the top, which
is acceptable because every stage is idempotent at the artifact level
(``put_artifact`` bumps the version rather than overwriting).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Optional

from . import sandbox as _sandbox
from . import storage as _storage
from .convergence import ConvergenceConfig
from .orchestrator import (
    CallLLM,
    LabRun,
    LabState,
    Stage,
    _LINEAR_ORDER,
    _default_call_llm,
    _drive,
)
from .roles import build_default_assignment


# ── State reconstruction ──────────────────────────────────────────────────


def reconstruct_state(storage: _storage.LabStorage,
                      run_id: str,
                      *,
                      start_stage: Optional[Stage] = None) -> LabState:
    """Build a fresh ``LabState`` from persisted artifacts + experiments.

    ``start_stage`` overrides the resume point.  When given, every artifact
    that belongs to a stage *after* ``start_stage`` is intentionally **not
    loaded** — the rewind semantics is that we want the orchestrator to
    re-derive those artifacts from scratch in the new iteration.

    When ``start_stage`` is None, we use ``runs.current_stage`` and load
    every artifact we have, so the resume continues from exactly where
    the previous run died.
    """
    rec = storage.get_run(run_id)
    if rec is None:
        raise ValueError(f"No lab run with id={run_id!r}")

    persisted_stage = (Stage(rec.current_stage)
                       if rec.current_stage else Stage.QUESTIONING)
    target_stage = start_stage or persisted_stage
    target_idx = _LINEAR_ORDER.index(target_stage)

    def _stage_idx(stage: Stage) -> int:
        return _LINEAR_ORDER.index(stage)

    # Helper: only load an artifact if its producing stage is *before*
    # target_stage. (Rewinding to drafting should not preload an existing
    # draft_full, otherwise the producer-with-review-loop would pass it in
    # as the "previous draft" and never produce something fresh.)
    def _load_if_before(art_kind: str, producing_stage: Stage) -> str:
        if _stage_idx(producing_stage) >= target_idx:
            return ""
        a = storage.get_latest_artifact(rec.run_id, art_kind)
        return a.content if a else ""

    # Each artifact's producing stage — we only preload it if the resume
    # target is strictly *after* that stage. Rewinding *to* a stage drops
    # the artifact that stage produces so the orchestrator regenerates it.
    rq_text = _load_if_before("rq", Stage.QUESTIONING)
    survey  = _load_if_before("survey", Stage.SURVEY)
    outline = _load_if_before("outline", Stage.OUTLINE)
    results_section = _load_if_before("results_section", Stage.ANALYSIS)
    draft_full = _load_if_before("draft_full", Stage.DRAFTING)

    # Experiment code + result: keep when target is *strictly past* the
    # producing stages. Code is produced by IMPLEMENTATION, result is
    # produced by EXPERIMENT.
    exp_code = ""
    exp_attempt = 0
    exp_result: Optional[_sandbox.SandboxResult] = None
    if _stage_idx(Stage.IMPLEMENTATION) < target_idx:
        # Walk experiment_code_v* artifacts to find the latest version.
        for art in storage.list_artifacts(rec.run_id):
            if not art.kind.startswith("experiment_code_v"):
                continue
            try:
                v = int(art.kind[len("experiment_code_v"):])
            except ValueError:
                continue
            if v > exp_attempt:
                exp_attempt = v
                exp_code = art.content
        last_exp = storage.get_latest_experiment(rec.run_id)
        if last_exp:
            artifact_paths = []
            for p in last_exp.artifacts or []:
                try:
                    artifact_paths.append(Path(p))
                except Exception:
                    pass
            # workspace is required by SandboxResult; we don't have the
            # original tempdir so synthesise a stable per-run path. Code
            # downstream only formats it for prompts, never re-runs from it.
            workspace = Path(f"/tmp/lab_resume_workspace_{rec.run_id}")
            exp_result = _sandbox.SandboxResult(
                exit_code=last_exp.exit_code if last_exp.exit_code is not None else -1,
                duration_s=last_exp.duration_s or 0.0,
                timed_out=bool(last_exp.timed_out),
                stdout=last_exp.stdout or "",
                stderr=last_exp.stderr or "",
                artifacts=artifact_paths,
                workspace=workspace,
            )

    # skip_experiment is decided in IMPLEMENTATION based on PI's reading of
    # the topic. We can recover it from the persisted PI decision messages
    # when we're past IMPLEMENTATION; otherwise leave default False.
    skip_experiment = False
    if _stage_idx(Stage.IMPLEMENTATION) < target_idx:
        skip_experiment = _detect_skip_experiment(storage, rec.run_id)

    return LabState(
        run_id=rec.run_id,
        topic=rec.topic,
        stage=target_stage,
        round=0,
        research_questions=[q for q in (rq_text or "").split("\n") if q.strip()],
        survey_summary=survey,
        outline=outline,
        experiment_code=exp_code,
        experiment_result=exp_result,
        experiment_attempt=exp_attempt,
        results_section=results_section,
        section_drafts=({"full_body": draft_full} if draft_full else {}),
        citations_raw="",
        cancel_requested=False,
        skip_experiment=skip_experiment,
    )


def _detect_skip_experiment(storage: _storage.LabStorage, run_id: str) -> bool:
    """Inspect PI decisions for a 'skip experiment' signal.

    The implementation stage emits a PI decision message whose content
    starts with "skip experiment:" when the topic is non-experimental
    (survey-only paper).  This is the only authoritative recovery
    signal — if no such message exists, we assume experiments are
    expected.
    """
    msgs = storage.list_messages(run_id, stage=Stage.IMPLEMENTATION.value,
                                  limit=50)
    for m in msgs:
        if m.role == "pi" and m.kind == "decision":
            txt = (m.content or "").lower()
            if "skip experiment" in txt or "skip_experiment" in txt:
                return True
    return False


# ── Public entrypoint ─────────────────────────────────────────────────────


def resume_run(*, run_id: str,
               config: dict,
               start_stage: Optional[Stage] = None,
               storage_obj: Optional[_storage.LabStorage] = None,
               role_override: Optional[dict] = None,
               convergence: Optional[ConvergenceConfig] = None,
               call_llm: Optional[CallLLM] = None,
               cancel_check: Optional[Callable[[], bool]] = None,
               on_stage_change: Optional[Callable[[Stage], None]] = None,
               output_root: Optional[Path] = None,
               ) -> LabRun:
    """Continue a previously-started lab run.

    The run's status is moved back to ``running``; on completion it
    becomes ``done`` (or ``aborted`` / ``failed``) like a fresh run.
    """
    storage = storage_obj or _storage.LabStorage()
    rec = storage.get_run(run_id)
    if rec is None:
        raise ValueError(f"No lab run with id={run_id!r}")

    state = reconstruct_state(storage, run_id, start_stage=start_stage)

    role_override = role_override or config.get("lab_role_override") or {}
    roles = build_default_assignment(config, override=role_override)
    convergence = convergence or ConvergenceConfig(max_rounds=rec.max_rounds)

    run = LabRun(
        state=state, storage=storage, roles=roles, config=config,
        convergence=convergence,
        call_llm=call_llm or _default_call_llm,
        on_stage_change=on_stage_change,
        output_root=output_root,
    )
    storage.update_run_status(rec.run_id, "running",
                              current_stage=state.stage.value)
    try:
        _drive(run, cancel_check=cancel_check)
        if state.cancel_requested:
            storage.update_run_status(rec.run_id, "aborted",
                                      current_stage=state.stage.value)
        else:
            storage.update_run_status(rec.run_id, "done",
                                      current_stage=state.stage.value)
    except Exception as exc:
        storage.update_run_status(rec.run_id, "failed",
                                  current_stage=state.stage.value,
                                  error=str(exc))
        raise
    return run


# ── Threaded helper used by the REPL ─────────────────────────────────────


def resume_run_in_thread(*, run_id: str, config: dict,
                         start_stage: Optional[Stage] = None,
                         on_finish: Optional[Callable[[bool, str], None]] = None,
                         **kwargs) -> tuple[threading.Thread, threading.Event]:
    """Spawn a daemon thread that runs ``resume_run`` and signal cancel.

    Returns ``(thread, cancel_event)``.  ``on_finish(success, message)``
    fires from the worker after the run terminates.
    """
    cancel = threading.Event()

    def _runner() -> None:
        ok = True
        msg = ""
        try:
            resume_run(
                run_id=run_id, config=config, start_stage=start_stage,
                cancel_check=cancel.is_set, **kwargs,
            )
            msg = "done"
        except Exception as exc:
            ok = False
            msg = f"{type(exc).__name__}: {exc}"
        if on_finish:
            try:
                on_finish(ok, msg)
            except Exception:
                pass

    t = threading.Thread(target=_runner, name=f"lab-resume-{run_id}",
                         daemon=True)
    t.start()
    return t, cancel
