"""research/lab/iterate.py — meta-loop: score the final report, rewind
to the weakest stage, re-run, repeat until convergence or budget.

Workflow per iteration:

    1. SCORE:   the reviewer pool reads the latest ``report`` artifact
                and rates 4 dimensions (novelty, rigor, clarity, evidence)
                on 1-10. We average across reviewers per dimension, then
                across dimensions for the overall score.

    2. ROUTE:   the lowest-scoring dimension picks which stage to rewind to
                (see ``DIMENSION_TO_STAGE`` below). E.g. low novelty →
                QUESTIONING (rethink the RQ); low evidence → EXPERIMENT
                (more / stronger experiments).

    3. REWIND:  call :func:`research.lab.resume.resume_run` with
                ``start_stage`` set to the routed stage. The resume code
                already drops every artifact produced *at or after* that
                stage from the in-memory state, so the orchestrator
                regenerates them. Old versions remain in the artifact
                table for audit.

    4. STOP:    when score ≥ ``target_score`` (default 7.0), or
                ``max_iterations`` reached, or the score plateaus
                (|delta| < ``plateau_eps`` for ``plateau_consec``
                consecutive iterations), or budget exhausted.

The whole iterate cycle records into ``lab_iterations`` (one row per
iteration), so the daemon and ``/lab status`` can show iteration history.
"""
from __future__ import annotations

import json
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import storage as _storage
from .orchestrator import (
    CallLLM,
    LabRun,
    Stage,
    _default_call_llm,
    _safe_load_template,
)
from .resume import resume_run
from .roles import Role, build_default_assignment


# ── Score model ───────────────────────────────────────────────────────────

# Dimensions a reviewer is asked to score the final paper on.
DIMENSIONS = ("novelty", "rigor", "clarity", "evidence")


# Map "weakest dimension" → which stage to rewind to. Conservative
# routing — we don't try to be clever, we go to the latest stage that
# meaningfully shapes that dimension.
DIMENSION_TO_STAGE: dict[str, Stage] = {
    "novelty":  Stage.QUESTIONING,   # rethink RQs
    "rigor":    Stage.IMPLEMENTATION,  # better methodology / code
    "clarity":  Stage.DRAFTING,      # rewrite the paper body
    "evidence": Stage.EXPERIMENT,    # run more / different experiments
}


@dataclass
class IterationResult:
    iter_n: int
    score_avg: float
    score_breakdown: dict[str, float]    # dim → 0..10
    revise_stage: Optional[Stage]        # None when we converged
    delta: float                         # score_avg − previous score_avg
    notes: str = ""


@dataclass
class IterationConfig:
    """Caller-tunable knobs for the meta-loop."""
    target_score: float = 7.0       # paper "good enough" cutoff
    max_iterations: int = 5
    plateau_eps: float = 0.3        # |delta| below this is "no progress"
    plateau_consec: int = 2         # … this many in a row → stop
    n_reviewers: int = 3            # reviewers used for scoring


# ── Score parser ──────────────────────────────────────────────────────────


# Permissive: accept any non-negative number; the clamp in
# parse_review_scores caps to [0, 10] so reviewers writing 11/10 or 15
# don't break us. Negatives don't match (no leading sign in the group).
_SCORE_LINE_RE = re.compile(
    r"^\s*([A-Za-z]+)\s*[:=]\s*(\d+(?:\.\d+)?)",
    re.MULTILINE,
)


def parse_review_scores(text: str) -> dict[str, float]:
    """Extract ``dim: score`` lines from a reviewer response.

    Reviewers are prompted to emit exactly four lines like::

        novelty: 6
        rigor: 7.5
        clarity: 7
        evidence: 5

    Anything they say outside these lines is ignored. Missing dimensions
    are returned as ``5.0`` (neutral) so a single malformed reply doesn't
    poison the whole iteration.
    """
    out: dict[str, float] = {}
    for m in _SCORE_LINE_RE.finditer(text or ""):
        key = m.group(1).strip().lower()
        if key in DIMENSIONS:
            try:
                out[key] = max(0.0, min(10.0, float(m.group(2))))
            except ValueError:
                pass
    for d in DIMENSIONS:
        out.setdefault(d, 5.0)
    return out


def aggregate_scores(per_reviewer: list[dict[str, float]]) -> dict[str, float]:
    """Per-dimension mean across reviewers (returns the 4 dims even if
    ``per_reviewer`` is empty — defaults to 5.0 each)."""
    out: dict[str, float] = {}
    for d in DIMENSIONS:
        vals = [r.get(d, 5.0) for r in per_reviewer]
        out[d] = statistics.fmean(vals) if vals else 5.0
    return out


def overall_score(by_dim: dict[str, float]) -> float:
    return statistics.fmean(by_dim.get(d, 5.0) for d in DIMENSIONS)


def weakest_dimension(by_dim: dict[str, float]) -> str:
    return min(DIMENSIONS, key=lambda d: by_dim.get(d, 5.0))


# ── Self-review (one iteration's scoring step) ────────────────────────────


_REVIEWER_USER_TEMPLATE = """You are reviewer #{n} of {total} reading the
final draft of an autonomous research paper. Score the paper on each of
these four dimensions on a 1-10 scale (10 = top venue ready, 5 = weak but
publishable on arXiv, 1 = unpublishable):

* novelty   — how novel is the research question and the contribution?
* rigor     — is the methodology sound? are assumptions made explicit?
* clarity   — is the writing clear and well-structured?
* evidence  — do the experiments / citations support the claims?

Respond in **exactly four lines**, no preamble, no commentary, in this
format:

novelty: <score>
rigor: <score>
clarity: <score>
evidence: <score>

Topic: {topic}

--- BEGIN REPORT ---
{report}
--- END REPORT ---
"""


def score_report(*, run: LabRun,
                 report_text: str,
                 reviewers: list[Role],
                 call_llm: CallLLM) -> dict[str, float]:
    """Run each reviewer once on the report; aggregate per-dimension."""
    per_reviewer: list[dict[str, float]] = []
    total = len(reviewers)
    for i, reviewer in enumerate(reviewers, 1):
        user = _REVIEWER_USER_TEMPLATE.format(
            n=i, total=total,
            topic=run.state.topic,
            report=report_text[:12000],
        )
        try:
            system = _safe_load_template(reviewer)
            resp = call_llm(
                role_name=reviewer.name,
                model=reviewer.model,
                system=system,
                user=user,
                config=run.config,
            )
            scores = parse_review_scores(resp.text)
        except Exception:
            scores = {d: 5.0 for d in DIMENSIONS}
        per_reviewer.append(scores)
        run.storage.append_message(
            run.state.run_id, stage="iterate", round_=0,
            role=f"reviewer_{i}", kind="critique",
            content=json.dumps(scores),
        )
    return aggregate_scores(per_reviewer)


# ── Single iteration step ─────────────────────────────────────────────────


def run_one_iteration(*, run_id: str,
                      iter_n: int,
                      config: dict,
                      iter_cfg: IterationConfig,
                      previous_score: Optional[float] = None,
                      storage_obj: Optional[_storage.LabStorage] = None,
                      call_llm: Optional[CallLLM] = None,
                      cancel_check: Optional[Callable[[], bool]] = None,
                      ) -> IterationResult:
    """Score the current report, decide whether to revise, do the revise.

    The caller is expected to wrap this in a loop. We separate score+revise
    into one step so the daemon can interleave iterations across runs and
    so a single iterate can be aborted mid-step cleanly.
    """
    storage = storage_obj or _storage.LabStorage()
    call = call_llm or _default_call_llm

    storage.add_iteration(
        run_id=run_id, iter_n=iter_n,
        target_score=iter_cfg.target_score,
        notes=f"iter {iter_n} starting; target≥{iter_cfg.target_score}",
    )
    storage.update_iteration(run_id=run_id, iter_n=iter_n, status="scoring")

    # ── 1. SCORE ───────────────────────────────────────────────────────
    rec = storage.get_run(run_id)
    if rec is None:
        raise ValueError(f"No lab run with id={run_id!r}")
    report_art = storage.get_latest_artifact(run_id, "report")
    if report_art is None:
        raise RuntimeError(f"Run {run_id} has no 'report' artifact yet — "
                            "iterate is for runs that already finalized")
    role_override = config.get("lab_role_override") or {}
    roles = build_default_assignment(config, override=role_override)
    reviewers = roles.reviewers[: iter_cfg.n_reviewers] or [roles.pi]

    # Build the same LabRun shape the orchestrator uses, so message
    # logging hits the right run_id and config.
    from .orchestrator import LabState, LabRun as _LabRun
    fake_state = LabState(run_id=run_id, topic=rec.topic, stage=Stage.FINALIZATION)
    run = _LabRun(state=fake_state, storage=storage, roles=roles,
                  config=config, call_llm=call)

    by_dim = score_report(
        run=run, report_text=report_art.content,
        reviewers=reviewers, call_llm=call,
    )
    avg = overall_score(by_dim)
    delta = (avg - previous_score) if previous_score is not None else 0.0

    storage.update_iteration(
        run_id=run_id, iter_n=iter_n,
        score_avg=avg, score_breakdown=by_dim, delta=delta,
        notes=f"avg={avg:.2f} dims={by_dim}",
    )

    # ── 2. CONVERGED? ─────────────────────────────────────────────────
    if avg >= iter_cfg.target_score:
        storage.update_iteration(
            run_id=run_id, iter_n=iter_n,
            status="done", revise_stage=None, mark_done=True,
            notes=f"converged at avg={avg:.2f} ≥ {iter_cfg.target_score}",
        )
        return IterationResult(
            iter_n=iter_n, score_avg=avg, score_breakdown=by_dim,
            revise_stage=None, delta=delta,
            notes="converged",
        )

    # ── 3. REWIND + RESUME ────────────────────────────────────────────
    weakest = weakest_dimension(by_dim)
    revise_stage = DIMENSION_TO_STAGE[weakest]
    storage.update_iteration(
        run_id=run_id, iter_n=iter_n,
        status="reverting", revise_stage=revise_stage.value,
        notes=f"weakest={weakest} ({by_dim[weakest]:.2f}) → "
              f"rewind to {revise_stage.value}",
    )

    if cancel_check and cancel_check():
        storage.update_iteration(run_id=run_id, iter_n=iter_n,
                                  status="skipped", mark_done=True,
                                  notes="cancelled before re-run")
        return IterationResult(
            iter_n=iter_n, score_avg=avg, score_breakdown=by_dim,
            revise_stage=revise_stage, delta=delta, notes="cancelled",
        )

    storage.update_iteration(run_id=run_id, iter_n=iter_n, status="running")
    try:
        resume_run(run_id=run_id, config=config,
                    start_stage=revise_stage,
                    storage_obj=storage,
                    call_llm=call,
                    cancel_check=cancel_check)
        storage.update_iteration(run_id=run_id, iter_n=iter_n,
                                  status="done", mark_done=True,
                                  notes=f"re-ran from {revise_stage.value}")
    except Exception as exc:
        storage.update_iteration(run_id=run_id, iter_n=iter_n,
                                  status="failed", mark_done=True,
                                  notes=f"resume failed: {type(exc).__name__}: {exc}")
        raise

    return IterationResult(
        iter_n=iter_n, score_avg=avg, score_breakdown=by_dim,
        revise_stage=revise_stage, delta=delta,
        notes=f"revised at {revise_stage.value}",
    )


# ── Full iterate-until-converged driver ───────────────────────────────────


def iterate_until_converged(*, run_id: str,
                            config: dict,
                            iter_cfg: Optional[IterationConfig] = None,
                            storage_obj: Optional[_storage.LabStorage] = None,
                            call_llm: Optional[CallLLM] = None,
                            cancel_check: Optional[Callable[[], bool]] = None,
                            on_iteration: Optional[Callable[[IterationResult], None]] = None,
                            ) -> list[IterationResult]:
    """Loop ``run_one_iteration`` until convergence / max / plateau.

    Returns the list of iteration results in order. The caller reads the
    last entry to see the final score and whether the run converged.
    """
    storage = storage_obj or _storage.LabStorage()
    cfg = iter_cfg or IterationConfig(
        target_score=float(config.get("lab_iterate_target", 7.0)),
        max_iterations=int(config.get("lab_iterate_max", 5)),
        plateau_eps=float(config.get("lab_iterate_plateau_eps", 0.3)),
        plateau_consec=int(config.get("lab_iterate_plateau_consec", 2)),
        n_reviewers=int(config.get("lab_iterate_reviewers", 3)),
    )
    history: list[IterationResult] = []
    prev_score: Optional[float] = None
    plateau_streak = 0
    base_n = storage.latest_iteration_n(run_id)

    for k in range(1, cfg.max_iterations + 1):
        if cancel_check and cancel_check():
            break
        try:
            result = run_one_iteration(
                run_id=run_id, iter_n=base_n + k,
                config=config, iter_cfg=cfg,
                previous_score=prev_score,
                storage_obj=storage, call_llm=call_llm,
                cancel_check=cancel_check,
            )
        except Exception:
            break
        history.append(result)
        if on_iteration:
            try:
                on_iteration(result)
            except Exception:
                pass
        if result.revise_stage is None:  # converged
            break
        if prev_score is not None and abs(result.delta) < cfg.plateau_eps:
            plateau_streak += 1
        else:
            plateau_streak = 0
        if plateau_streak >= cfg.plateau_consec:
            storage.update_iteration(
                run_id=run_id, iter_n=result.iter_n,
                notes=f"{result.notes}; stopped (plateau)",
            )
            break
        prev_score = result.score_avg

    return history


# ── Threaded entry for the REPL ───────────────────────────────────────────


def iterate_in_thread(*, run_id: str, config: dict,
                      iter_cfg: Optional[IterationConfig] = None,
                      on_finish: Optional[Callable[[list[IterationResult]], None]] = None,
                      **kwargs) -> tuple[threading.Thread, threading.Event]:
    cancel = threading.Event()

    def _runner() -> None:
        try:
            history = iterate_until_converged(
                run_id=run_id, config=config,
                iter_cfg=iter_cfg,
                cancel_check=cancel.is_set, **kwargs,
            )
        except Exception:
            history = []
        if on_finish:
            try:
                on_finish(history)
            except Exception:
                pass

    t = threading.Thread(target=_runner, name=f"lab-iter-{run_id}",
                         daemon=True)
    t.start()
    return t, cancel
