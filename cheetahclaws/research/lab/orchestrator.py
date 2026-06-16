"""research/lab/orchestrator.py — multi-agent stage graph driver.

Stage graph (engine v0, no experiments):

    [topic]
       │
       ▼
    QUESTIONING        — Questioner + Lay Reader debate; PI signs off RQs
       │
       ▼
    SURVEY             — Surveyor runs literature pipeline; Reviewer × N
                         judge gap analysis
       │
       ▼
    OUTLINE            — Designer drafts paper outline; Reviewer × N judge
       │
       ▼
    DRAFTING           — Writer drafts each section; Reviewer × N + Lay
       │ ↑ revise       Reader critique; loop until convergence
       ▼ │
    VERIFICATION       — Citation verifier checks every reference
       │
       ▼
    FINALIZATION       — Markdown report written, run marked done

Each stage runs through ``run_stage_with_convergence`` which:
  1. invokes the producer agent (e.g. designer for OUTLINE)
  2. invokes reviewers (parallel-ish; serial for v0 simplicity)
  3. asks ``decide_advance``
  4. either advances, iterates (calls producer with critiques), or
     bails to a redesign back-edge

Concurrency: orchestrator runs on a single thread; reviewer LLM calls
are serial.  v0 prioritizes simplicity; v1 can parallelize reviewers.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from . import storage as _storage
from . import sandbox as _sandbox
from .convergence import (
    BudgetStatus, ConvergenceConfig, ConvergenceDecision,
    ReviewerVerdict, decide_advance,
)
from .roles import (
    ROLE_ANALYST, ROLE_DESIGNER, ROLE_ENGINEER, ROLE_LAY_READER,
    ROLE_PI, ROLE_QUESTIONER, ROLE_REVIEWER, ROLE_SURVEYOR, ROLE_WRITER,
    Role, RoleAssignment, build_default_assignment, load_role_template,
)


# ── Stage enum ────────────────────────────────────────────────────────────


class Stage(str, Enum):
    QUESTIONING    = "questioning"
    SURVEY         = "survey"
    OUTLINE        = "outline"
    IMPLEMENTATION = "implementation"   # Engineer drafts code
    EXPERIMENT     = "experiment"        # Sandbox runs code (engineer-runner debug loop)
    ANALYSIS       = "analysis"          # Analyst interprets results
    DRAFTING       = "drafting"
    VERIFICATION   = "verification"
    FINALIZATION   = "finalization"


_LINEAR_ORDER: list[Stage] = [
    Stage.QUESTIONING, Stage.SURVEY, Stage.OUTLINE,
    Stage.IMPLEMENTATION, Stage.EXPERIMENT, Stage.ANALYSIS,
    Stage.DRAFTING, Stage.VERIFICATION, Stage.FINALIZATION,
]


# ── LLM call surface ──────────────────────────────────────────────────────
# We deliberately accept a callable rather than wiring providers.stream
# directly, so tests can substitute a stub. The real binding is created
# in run_one_lab_session below.

class CallLLM(Protocol):
    def __call__(self, *, role_name: str, model: str,
                 system: str, user: str,
                 config: dict) -> "LLMResponse": ...


@dataclass
class LLMResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_cents: int = 0


def _default_call_llm(*, role_name: str, model: str,
                      system: str, user: str, config: dict) -> LLMResponse:
    """Wraps providers.stream with token / cost accounting.

    Falls back to a synthetic stub when providers/imports fail (so unit
    tests on a network-less CI box don't hang).
    """
    try:
        from cheetahclaws import providers
    except Exception:
        return LLMResponse(text=f"[{role_name} stub: providers unavailable]")

    chunks: list[str] = []
    last_turn = None
    try:
        for ev in providers.stream(
            model=model, system=system,
            messages=[{"role": "user", "content": user}],
            tool_schemas=[],
            config={**config, "max_tokens": 4000, "thinking": False},
        ):
            t = getattr(ev, "text", None)
            if t:
                chunks.append(t)
            if hasattr(ev, "tokens_in") or hasattr(ev, "tokens_out"):
                last_turn = ev
    except Exception as exc:
        return LLMResponse(text=f"[{role_name} error: {exc}]")
    text = "".join(chunks)
    t_in = int(getattr(last_turn, "tokens_in", 0) or 0)
    t_out = int(getattr(last_turn, "tokens_out", 0) or 0)
    cost = 0
    try:
        from cheetahclaws.config import calc_cost
        cost = int(round(calc_cost(model, t_in, t_out) * 100))
    except Exception:
        pass
    return LLMResponse(text=text, tokens_in=t_in, tokens_out=t_out,
                       cost_cents=cost)


# ── State + lifecycle ─────────────────────────────────────────────────────


@dataclass
class LabState:
    """Mutable per-run state held by the orchestrator."""
    run_id: str
    topic: str
    stage: Stage
    round: int = 0
    research_questions: list[str] = field(default_factory=list)
    survey_summary: str = ""
    outline: str = ""
    experiment_code: str = ""           # latest engineer draft
    experiment_result: Optional[_sandbox.SandboxResult] = None
    experiment_attempt: int = 0
    results_section: str = ""           # analyst's draft
    section_drafts: dict[str, str] = field(default_factory=dict)
    citations_raw: str = ""             # writer's bibliography output
    cancel_requested: bool = False
    skip_experiment: bool = False        # true when topic isn't experiment-amenable


@dataclass
class LabRun:
    """Owns storage, role assignments, callbacks. One per `/lab start` call."""
    state: LabState
    storage: _storage.LabStorage
    roles: RoleAssignment
    config: dict
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    call_llm: CallLLM = _default_call_llm
    on_stage_change: Optional[Callable[[Stage], None]] = None
    # Optional override for the filesystem root where this run's
    # report.md, references.bib, sandbox workspace, etc. all go.
    # When None, falls back to ~/.cheetahclaws/research_papers/<human-name>/.
    # Tests inject a tmp_path here so they don't pollute the user's home.
    output_root: Optional["Path"] = None

    @property
    def output_dir(self) -> "Path":
        """Single canonical filesystem dir for this run — report.md,
        references.bib, sandbox workspaces, citation_verified.json all
        live under here, so one ``/lab`` invocation = one folder.
        """
        from .storage import DEFAULT_OUTPUT_DIR, output_dir_for
        rec = self.storage.get_run(self.state.run_id)
        if rec is not None and rec.created_at:
            return output_dir_for(
                rec.run_id, rec.topic, rec.created_at,
                root=self.output_root,
            )
        # Legacy / test path: no DB record yet, fall back to run_id.
        root = self.output_root or DEFAULT_OUTPUT_DIR
        return root / self.state.run_id


# ── Public entrypoint ─────────────────────────────────────────────────────


def run_one_lab_session(
    *, topic: str,
    config: dict,
    storage_obj: Optional[_storage.LabStorage] = None,
    role_override: Optional[dict] = None,
    convergence: Optional[ConvergenceConfig] = None,
    call_llm: Optional[CallLLM] = None,
    budget_tokens: Optional[int] = 5_000_000,
    budget_cost_cents: Optional[int] = 5000,
    max_rounds: int = 5,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_stage_change: Optional[Callable[[Stage], None]] = None,
    output_root: Optional["Path"] = None,
) -> LabRun:
    """Run a full lab session start→finalization in the calling thread.

    The caller should typically run this in a background thread (the
    REPL slash command does this) since each stage may take many
    minutes.  Cancellation is checked between stages and between
    rounds via ``cancel_check``.
    """
    storage = storage_obj or _storage.LabStorage()
    rec = storage.create_run(
        topic=topic,
        budget_tokens=budget_tokens,
        budget_cost_cents=budget_cost_cents,
        max_rounds=max_rounds,
    )
    roles = build_default_assignment(config, override=role_override)
    state = LabState(run_id=rec.run_id, topic=topic, stage=Stage.QUESTIONING)
    run = LabRun(
        state=state, storage=storage, roles=roles, config=config,
        convergence=convergence or ConvergenceConfig(max_rounds=max_rounds),
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


# ── Main driver ───────────────────────────────────────────────────────────


def _drive(run: LabRun, *, cancel_check: Optional[Callable[[], bool]] = None):
    while True:
        if cancel_check and cancel_check():
            run.state.cancel_requested = True
            return
        budget_status = _budget_status(run)
        if budget_status.exceeded:
            run.storage.append_message(
                run.state.run_id, stage=run.state.stage.value, round_=0,
                role=ROLE_PI, kind="decision",
                content=f"Budget exhausted ({budget_status.fraction_used():.0%});"
                        f" finalizing early.",
            )
            run.state.stage = Stage.FINALIZATION

        if run.on_stage_change:
            try:
                run.on_stage_change(run.state.stage)
            except Exception:
                pass
        run.storage.update_run_status(run.state.run_id, "running",
                                       current_stage=run.state.stage.value)

        if run.state.stage == Stage.QUESTIONING:
            _stage_questioning(run)
        elif run.state.stage == Stage.SURVEY:
            _stage_survey(run)
        elif run.state.stage == Stage.OUTLINE:
            _stage_outline(run)
        elif run.state.stage == Stage.IMPLEMENTATION:
            _stage_implementation(run)
        elif run.state.stage == Stage.EXPERIMENT:
            _stage_experiment(run)
        elif run.state.stage == Stage.ANALYSIS:
            _stage_analysis(run)
        elif run.state.stage == Stage.DRAFTING:
            _stage_drafting(run)
        elif run.state.stage == Stage.VERIFICATION:
            _stage_verification(run)
        elif run.state.stage == Stage.FINALIZATION:
            _stage_finalization(run)
            return  # terminal

        # Advance linearly (back-edges are handled inside individual stages).
        next_idx = _LINEAR_ORDER.index(run.state.stage) + 1
        if next_idx >= len(_LINEAR_ORDER):
            return
        run.state.stage = _LINEAR_ORDER[next_idx]
        run.state.round = 0


# ── Stage implementations ────────────────────────────────────────────────


def _stage_questioning(run: LabRun) -> None:
    """Topic → narrowable research questions, signed off by PI."""
    s = run.state
    run.storage.start_stage(s.run_id, Stage.QUESTIONING.value, 0)
    user_prompt = (
        f"Research topic from the user:\n\n{run.state.topic}\n\n"
        "Produce 3-5 narrow, falsifiable research questions, ranked by"
        " novelty + tractability. Output a numbered list."
    )
    resp = _invoke(run, run.roles.questioner,
                    user=user_prompt, kind="draft")
    s.research_questions = _extract_numbered(resp.text)
    run.storage.put_artifact(s.run_id, "rq", "\n".join(s.research_questions))

    # PI signs off.
    pi_prompt = (
        f"Topic: {run.state.topic}\n\n"
        f"Proposed research questions from the questioner:\n"
        + "\n".join(f"{i+1}. {q}" for i, q in enumerate(s.research_questions))
        + "\n\nAs the PI, pick the single most promising RQ and explain"
          " your choice in ≤ 3 sentences."
    )
    pi_resp = _invoke(run, run.roles.pi, user=pi_prompt, kind="decision")
    run.storage.put_artifact(s.run_id, "rq_decision", pi_resp.text)
    if s.research_questions:
        # PI's pick: use the first RQ named in their response, or fallback to #1.
        pick = _find_first_rq_match(pi_resp.text, s.research_questions)
        if pick is not None:
            s.research_questions = [pick] + [q for q in s.research_questions
                                              if q != pick]
    run.storage.end_stage(s.run_id, Stage.QUESTIONING.value, 0,
                          outcome="advance",
                          notes=f"selected RQ: {s.research_questions[0] if s.research_questions else ''}")


def _stage_survey(run: LabRun) -> None:
    """Surveyor maps related work + identifies gaps.

    Now grounded in real search hits from `research.aggregator.research`
    rather than pure model-memory hallucination. The surveyor LLM is
    given top-N titles + abstracts as context, and must base its
    "## Related work" + "## Citations" sections on those — fabricated
    citations get caught later by the verifier, but priming with real
    data dramatically lowers their rate.

    On any failure (no API keys, all sources rate-limited, etc.) we
    fall back to the old prompt-only path so a fresh laptop without
    Tavily/Brave/etc. can still get some output.
    """
    s = run.state
    run.storage.start_stage(s.run_id, Stage.SURVEY.value, 0)
    rq = s.research_questions[0] if s.research_questions else s.topic

    search_block = _gather_search_context(run, rq)

    if search_block:
        # Persist search results so /lab logs has them and reviewer
        # iterations can replay against the same evidence.
        run.storage.put_artifact(s.run_id, "survey_search_hits", search_block)

    if search_block:
        user_prompt = (
            f"Topic: {run.state.topic}\n"
            f"Selected research question: {rq}\n\n"
            "Real search results (use these as the primary source — do "
            "NOT cite anything not listed below unless you can name a "
            "specific paper from training data with high confidence):\n\n"
            f"{search_block}\n\n"
            "Produce:\n"
            "  ## Related work\n"
            "    A 2-3 paragraph synthesis of major prior threads, with"
            " inline citations like [Author, Year]. Cite the real hits"
            " above; group similar ones together.\n"
            "  ## Identified gap\n"
            "    The specific gap our work intends to fill, in 2-3 sentences.\n"
            "    Be concrete — a gap is a question the literature above"
            " has not answered, not just a sentence about needing more"
            " research.\n"
            "  ## Citations\n"
            "    Bullet list of every reference you used, format:\n"
            "    `- Title (Authors, Year). Optional arXiv:NNNN.`\n"
            "Output Markdown."
        )
    else:
        # Fallback to the original prompt — no search hits available.
        user_prompt = (
            f"Topic: {run.state.topic}\n"
            f"Selected research question: {rq}\n\n"
            "Survey the relevant literature you know about. Produce:\n"
            "  ## Related work\n"
            "    A 2-3 paragraph synthesis of major prior threads, with"
            " inline citations like [Author, Year].\n"
            "  ## Identified gap\n"
            "    The specific gap our work intends to fill, in 2-3 sentences.\n"
            "  ## Citations\n"
            "    A bullet list: `- Title (Authors, Year). Optional arXiv:NNNN.`\n"
            "Output Markdown."
        )

    resp = _invoke(run, run.roles.surveyor, user=user_prompt, kind="draft")
    s.survey_summary = resp.text
    run.storage.put_artifact(s.run_id, "survey", s.survey_summary)
    s.citations_raw = _extract_citations_block(s.survey_summary)
    run.storage.end_stage(s.run_id, Stage.SURVEY.value, 0, outcome="advance")


def _gather_search_context(run: LabRun, rq: str, *,
                           per_source_limit: int = 8,
                           max_total: int = 30,
                           max_chars: int = 8000) -> str:
    """Run /research's aggregator on the topic + RQ, format as context.

    Returns "" if the aggregator fails wholesale (network down, no
    sources available, etc.) so the surveyor falls back gracefully.
    Best-effort: a partial result is still better than no grounding.

    Failure modes are logged to the message bus so /lab logs surfaces
    *why* grounding skipped this run — silent fallback was unacceptable
    when we couldn't tell missing keys from a real bug.
    """
    try:
        from cheetahclaws.research.aggregator import research as _research
        # Bias toward academic + tech, the buckets that matter for survey.
        # We deliberately don't pass --sources so the classifier can pick
        # the strongest available ones (arxiv / openalex / semantic_scholar
        # / huggingface_papers / google_scholar).
        brief = _research(
            topic=f"{run.state.topic} :: {rq}",
            domains=["academic", "tech"],
            limit=per_source_limit,
            use_cache=True,
            synthesize=False,        # we don't want the model brief here
            max_total_results=max_total,
            config=run.config,
        )
    except Exception as exc:
        run.storage.append_message(
            run.state.run_id, stage=Stage.SURVEY.value, round_=0,
            role="surveyor", kind="note",
            content=(f"[grounding skipped] research.aggregator raised "
                     f"{type(exc).__name__}: {exc}"),
        )
        return ""

    if not brief or not getattr(brief, "results", None):
        # Surface per-source failures so the user can fix them
        # (e.g. set TAVILY_API_KEY or wait out a 429 burst).
        statuses = getattr(brief, "statuses", None) if brief else None
        diag_lines = []
        if statuses:
            for st in statuses:
                if not getattr(st, "ok", False):
                    diag_lines.append(
                        f"  - {getattr(st, 'name', '?')}: "
                        f"{getattr(st, 'error', 'no results')}"
                    )
        diag = "\n".join(diag_lines) if diag_lines else "  (no source diagnostics)"
        run.storage.append_message(
            run.state.run_id, stage=Stage.SURVEY.value, round_=0,
            role="surveyor", kind="note",
            content=(f"[grounding skipped] aggregator returned 0 results.\n"
                     f"per-source status:\n{diag}"),
        )
        return ""

    # Format hits compactly. Each hit ≈ 250 chars; cap on total chars.
    parts: list[str] = []
    used = 0
    for i, r in enumerate(brief.results, 1):
        title = (getattr(r, "title", "") or "").strip().replace("\n", " ")
        url = (getattr(r, "url", "") or "").strip()
        snippet = (getattr(r, "snippet", "") or "").strip().replace("\n", " ")
        src = (getattr(r, "source", "") or "").strip()
        snippet = snippet[:300]
        line = f"[{i}] ({src}) {title}\n    {url}\n    {snippet}".rstrip()
        if used + len(line) > max_chars:
            parts.append(f"… ({len(brief.results) - i + 1} more hits omitted for length)")
            break
        parts.append(line)
        used += len(line) + 1
    return "\n".join(parts)


def _stage_outline(run: LabRun) -> None:
    """Designer drafts paper outline; reviewers judge."""
    s = run.state
    user_prompt = (
        f"Topic: {run.state.topic}\n"
        f"Research question: {s.research_questions[0] if s.research_questions else 'TBD'}\n\n"
        f"Survey + gap:\n{s.survey_summary[:2000]}\n\n"
        "Produce a paper outline in Markdown. Use H2 (`##`) for sections."
        " Sections expected: Introduction, Background, Approach, Discussion,"
        " Conclusion, References. Under each, 1-3 bullet points naming what"
        " content goes there. Be specific to this topic; do not output a"
        " generic template."
    )
    decision = _producer_with_review_loop(
        run, stage=Stage.OUTLINE,
        producer=run.roles.designer,
        user_prompt=user_prompt,
        update_artifact=lambda r: setattr(s, "outline", r),
        artifact_kind="outline",
    )
    if decision is None:
        # User cancelled.
        return


def _stage_implementation(run: LabRun) -> None:
    """Engineer drafts a self-contained Python script for the experiment.

    If the user's config disables experiments (``lab_experiments=false``)
    OR the topic is fundamentally non-experimental (the Engineer signals
    by outputting ``# SKIP_EXPERIMENT: <reason>`` instead of code), this
    stage is a no-op and the run continues at DRAFTING.
    """
    s = run.state
    if not run.config.get("lab_experiments", True):
        s.skip_experiment = True
        run.storage.append_message(
            s.run_id, stage=Stage.IMPLEMENTATION.value, round_=0,
            role=ROLE_PI, kind="decision",
            content="Config has lab_experiments=false; skipping experiment stages.",
        )
        return
    run.storage.start_stage(s.run_id, Stage.IMPLEMENTATION.value, 0)
    rq = s.research_questions[0] if s.research_questions else s.topic
    user_prompt = (
        f"Topic: {run.state.topic}\n"
        f"Research question: {rq}\n\n"
        f"Survey + gap:\n{s.survey_summary[:1500]}\n\n"
        f"Outline:\n{s.outline[:1500]}\n\n"
        "Draft a single Python script that runs the smallest meaningful"
        " experiment supporting the paper's claims.\n\n"
        "If this topic does not admit an experiment (e.g. it is purely"
        " survey/position), respond with exactly one line:\n"
        "  `# SKIP_EXPERIMENT: <one-sentence reason>`\n"
        "and no code. Otherwise output the Python script per the format"
        " in your role prompt."
    )
    resp = _invoke(run, run.roles.engineer, user=user_prompt, kind="draft")
    if "SKIP_EXPERIMENT" in resp.text and "```" not in resp.text:
        s.skip_experiment = True
        run.storage.append_message(
            s.run_id, stage=Stage.IMPLEMENTATION.value, round_=0,
            role=ROLE_PI, kind="decision",
            content=f"Engineer skipped: {resp.text.strip()[:200]}",
        )
        run.storage.end_stage(s.run_id, Stage.IMPLEMENTATION.value, 0,
                              outcome="advance", notes="skipped (non-experimental topic)")
        return
    code = _sandbox.extract_python_block(resp.text) or ""
    s.experiment_code = code
    run.storage.put_artifact(s.run_id, "experiment_code_v1", code)
    run.storage.end_stage(s.run_id, Stage.IMPLEMENTATION.value, 0,
                          outcome="advance",
                          notes=f"engineer drafted {len(code)} chars of code")


def _stage_experiment(run: LabRun) -> None:
    """Run the engineer's code in the sandbox; on failure, debug-loop with engineer."""
    s = run.state
    if s.skip_experiment or not s.experiment_code:
        return
    # Sandbox workspace lives INSIDE the run's canonical output dir, so
    # one /lab invocation = one folder containing report + workspace.
    workspace = run.output_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    timeout_s = float(run.config.get("lab_experiment_timeout_s", 180))
    max_attempts = int(run.config.get("lab_experiment_max_attempts", 3))

    code = s.experiment_code
    last_result: Optional[_sandbox.SandboxResult] = None
    for attempt in range(1, max_attempts + 1):
        run.storage.start_stage(s.run_id, Stage.EXPERIMENT.value, attempt)
        s.experiment_attempt = attempt
        result = _sandbox.run_python_in_sandbox(
            code, workspace_dir=workspace, timeout_s=timeout_s,
        )
        last_result = result
        run.storage.record_experiment(
            run_id=s.run_id, attempt=attempt, code=code,
            exit_code=result.exit_code, stdout=result.stdout,
            stderr=result.stderr, duration_s=result.duration_s,
            timed_out=result.timed_out,
            artifacts=[p.name for p in result.artifacts],
        )
        run.storage.append_message(
            s.run_id, stage=Stage.EXPERIMENT.value, round_=attempt,
            role="runner", kind="note",
            content=_sandbox.format_result_for_prompt(result),
            meta={"exit_code": result.exit_code,
                  "duration_s": result.duration_s,
                  "timed_out": result.timed_out},
        )
        if result.exit_code == 0 and not result.timed_out:
            run.storage.end_stage(s.run_id, Stage.EXPERIMENT.value, attempt,
                                  outcome="advance",
                                  notes=f"exit 0 in {result.duration_s:.2f}s")
            s.experiment_result = result
            return
        if attempt >= max_attempts:
            run.storage.end_stage(s.run_id, Stage.EXPERIMENT.value, attempt,
                                  outcome="abort",
                                  notes=f"failed after {max_attempts} attempts;"
                                        f" last exit {result.exit_code}")
            s.experiment_result = result
            return
        # Engineer revises against the failed result.
        run.storage.end_stage(s.run_id, Stage.EXPERIMENT.value, attempt,
                              outcome="iterate",
                              notes=f"exit {result.exit_code}, asking engineer to fix")
        revise_user = (
            f"The previous run failed. Here's the result:\n\n"
            f"{_sandbox.format_result_for_prompt(result)}\n\n"
            f"Previous code:\n```python\n{code[:6000]}\n```\n\n"
            "Fix the bug. Output the corrected full Python script in a"
            " single fenced block (```python ... ```)."
        )
        resp = _invoke(run, run.roles.engineer, user=revise_user, kind="draft")
        new_code = _sandbox.extract_python_block(resp.text)
        if new_code:
            code = new_code
            s.experiment_code = code
            run.storage.put_artifact(
                s.run_id, f"experiment_code_v{attempt+1}", code,
            )


def _stage_analysis(run: LabRun) -> None:
    """Analyst reads experiment outputs and drafts the Results section."""
    s = run.state
    if s.skip_experiment:
        return
    run.storage.start_stage(s.run_id, Stage.ANALYSIS.value, 0)
    result = s.experiment_result
    if result is None:
        run.storage.end_stage(s.run_id, Stage.ANALYSIS.value, 0,
                              outcome="advance",
                              notes="no experiment result; skipping analysis")
        return

    user_prompt = (
        f"Topic: {s.topic}\n\n"
        f"Experiment code:\n```python\n{s.experiment_code[:4000]}\n```\n\n"
        f"Run output:\n{_sandbox.format_result_for_prompt(result, max_lines=120)}\n\n"
        f"Artifacts produced (filenames only):\n"
        + "\n".join(f"  - {p.name}" for p in result.artifacts) + "\n\n"
        "Draft the Results section per your role prompt. Reference figures"
        " by their actual filenames; do not fabricate figures that"
        " weren't produced."
    )
    resp = _invoke(run, run.roles.analyst, user=user_prompt, kind="draft")
    s.results_section = resp.text
    run.storage.put_artifact(s.run_id, "results_section", resp.text)
    run.storage.end_stage(s.run_id, Stage.ANALYSIS.value, 0,
                          outcome="advance",
                          notes=f"analyst drafted {len(resp.text)} chars of results")


def _stage_drafting(run: LabRun) -> None:
    """Writer drafts the full body; reviewers + lay reader iterate.

    When experiment data is available, the Results section gets pre-filled
    by the Analyst so the Writer slots it in rather than fabricating numbers.
    """
    s = run.state
    results_block = ""
    if s.results_section:
        results_block = (
            "\n\nThe Analyst has already drafted the Results section based on"
            " actual experiment output. Use this verbatim or with light"
            " editing, but do not fabricate alternative numbers:\n\n"
            f"---\n{s.results_section}\n---\n\n"
            "Available figure files:\n"
            + "\n".join(f"  - {p.name}"
                       for p in (s.experiment_result.artifacts
                                 if s.experiment_result else []))
        )
    user_prompt = (
        f"Topic: {run.state.topic}\n"
        f"Outline:\n{s.outline}\n\n"
        f"Survey + gap:\n{s.survey_summary[:1500]}\n\n"
        + results_block +
        "\n\nDraft the full paper body in Markdown, expanding each outline"
        " section. Aim for 1500-3500 words total. Use [Author, Year]"
        " citation style and include all sources in a final ## References"
        " section.\n"
        "Tone: clear, technical, confident but not overclaiming."
    )
    _producer_with_review_loop(
        run, stage=Stage.DRAFTING,
        producer=run.roles.writer,
        user_prompt=user_prompt,
        update_artifact=lambda r: s.section_drafts.update({"full_body": r}),
        artifact_kind="draft_full",
        include_lay_reader=True,
    )


def _stage_verification(run: LabRun) -> None:
    """Run the citation verifier on the final draft."""
    s = run.state
    run.storage.start_stage(s.run_id, Stage.VERIFICATION.value, 0)
    body = s.section_drafts.get("full_body", "")
    citations = _parse_citations_from_markdown(body + "\n" + s.citations_raw)
    if not citations:
        run.storage.append_message(
            s.run_id, stage=Stage.VERIFICATION.value, round_=0,
            role=ROLE_PI, kind="note",
            content="No citations parsed from draft — skipping verification.",
        )
        run.storage.end_stage(s.run_id, Stage.VERIFICATION.value, 0,
                              outcome="advance",
                              notes="0 citations parsed")
        return
    try:
        from .verifier import verify_citations

        def _progress(i: int, n: int, status: str) -> None:
            run.storage.append_message(
                s.run_id, stage=Stage.VERIFICATION.value, round_=0,
                role="verifier", kind="note",
                content=f"[{i}/{n}] {status}",
            )

        # Hard caps — without these a slow-loris socket on arxiv / SS
        # hangs the run for tens of minutes (we observed 11 min in the
        # field). Per-citation hard timeout = 30s, full stage = 5 min.
        result = verify_citations(
            citations,
            sleep_s=3.1,
            per_citation_hard_s=30.0,
            stage_max_s=300.0,
            progress_cb=_progress,
        )
    except Exception as exc:
        run.storage.append_message(
            s.run_id, stage=Stage.VERIFICATION.value, round_=0,
            role=ROLE_PI, kind="note",
            content=f"Verifier crashed: {exc}",
        )
        run.storage.end_stage(s.run_id, Stage.VERIFICATION.value, 0,
                              outcome="advance",
                              notes="verifier_error")
        return

    summary = (
        f"Citation verification: {result.n_verified} verified,"
        f" {result.n_ambiguous} ambiguous,"
        f" {result.n_not_found} not found,"
        f" {result.n_skipped} skipped."
    )
    run.storage.put_artifact(s.run_id, "citations_verified",
                              json.dumps([_verif_to_dict(v)
                                          for v in result.verifications],
                                         indent=2, ensure_ascii=False))
    run.storage.append_message(
        s.run_id, stage=Stage.VERIFICATION.value, round_=0,
        role=ROLE_PI, kind="note", content=summary,
    )
    run.storage.end_stage(s.run_id, Stage.VERIFICATION.value, 0,
                          outcome="advance", notes=summary)


def _stage_finalization(run: LabRun) -> None:
    """Compose the final markdown report and stash it."""
    s = run.state
    run.storage.start_stage(s.run_id, Stage.FINALIZATION.value, 0)
    from .output import write_markdown_report
    final = write_markdown_report(run)
    run.storage.put_artifact(s.run_id, "report", final)
    run.storage.end_stage(s.run_id, Stage.FINALIZATION.value, 0,
                          outcome="advance",
                          notes=f"final report: {len(final)} chars")


# ── Producer + reviewer loop helper ───────────────────────────────────────


def _producer_with_review_loop(
    run: LabRun,
    *, stage: Stage,
    producer: Role,
    user_prompt: str,
    update_artifact: Callable[[str], None],
    artifact_kind: str,
    include_lay_reader: bool = False,
) -> Optional[ConvergenceDecision]:
    """Drive a stage that uses the reviewer-author convergence loop.

    Returns the final convergence decision, or None if the run was cancelled.
    """
    s = run.state
    cur_text = ""
    last_critiques: list[str] = []

    for round_idx in range(1, run.convergence.max_rounds + 1):
        run.storage.start_stage(s.run_id, stage.value, round_idx)
        s.round = round_idx

        # ── Produce / revise ───────────────────────────────────────────
        if last_critiques:
            revise_prompt = (
                f"{user_prompt}\n\n"
                "Reviewers from the previous round raised these issues; revise"
                " your draft to address them:\n"
                + "\n".join(f"- {c}" for c in last_critiques)
                + f"\n\nPrevious draft:\n{cur_text[:6000]}"
            )
            resp = _invoke(run, producer, user=revise_prompt, kind="draft")
        else:
            resp = _invoke(run, producer, user=user_prompt, kind="draft")
        cur_text = resp.text
        update_artifact(cur_text)
        run.storage.put_artifact(s.run_id, artifact_kind, cur_text)

        # ── Reviewers critique ─────────────────────────────────────────
        verdicts: list[ReviewerVerdict] = []
        for reviewer in run.roles.reviewers:
            review_user = (
                f"You are reviewer {reviewer.name}. Critique the following"
                f" draft for stage `{stage.value}` of a research paper on:\n\n"
                f"Topic: {s.topic}\n\n"
                f"Draft to review:\n{cur_text[:8000]}\n\n"
                "Output exactly this JSON envelope (no other text):\n"
                "```json\n"
                "{\n"
                '  "score": <integer 1-10>,\n'
                '  "blocking_issues": ["..."],\n'
                '  "suggestions": ["..."],\n'
                '  "overall": "<one-line summary>"\n'
                "}\n"
                "```"
            )
            r_resp = _invoke(run, reviewer, user=review_user, kind="critique")
            v = _parse_reviewer_verdict(r_resp.text, reviewer.name)
            verdicts.append(v)

        if include_lay_reader:
            lay_user = (
                f"As a non-expert reader, judge if the following draft is"
                f" understandable and well-motivated.\n\nDraft:\n{cur_text[:6000]}\n\n"
                "Output the same JSON envelope as the reviewers."
            )
            r_resp = _invoke(run, run.roles.lay_reader, user=lay_user,
                              kind="critique")
            v = _parse_reviewer_verdict(r_resp.text, "lay_reader")
            verdicts.append(v)

        # ── PI synthesizes critiques into a critique list for next round ─
        last_critiques = []
        for v in verdicts:
            last_critiques.extend(v.blocking_issues[:3])  # cap noise
        if not last_critiques:
            for v in verdicts:
                last_critiques.extend(v.suggestions[:1])

        # ── Decide ──────────────────────────────────────────────────────
        decision = decide_advance(verdicts, round_index=round_idx,
                                   config=run.convergence)
        run.storage.append_message(
            s.run_id, stage=stage.value, round_=round_idx,
            role=ROLE_PI, kind="decision",
            content=decision.reason,
            meta={"advance": decision.advance,
                  "needs_redesign": decision.needs_redesign,
                  "n_pass": sum(1 for v in verdicts if v.passes),
                  "n_total": len(verdicts)},
        )
        run.storage.end_stage(s.run_id, stage.value, round_idx,
                              outcome="advance" if decision.advance
                              else ("redesign" if decision.needs_redesign
                                    else "iterate"),
                              notes=decision.reason)

        if decision.advance:
            return decision
        if decision.needs_redesign:
            # Bail back: caller stages handle this by re-entering an
            # earlier stage. For v0 we just advance; v1 wires real back-edges.
            return decision
        # else: continue loop
    return decision


# ── LLM invocation + accounting ──────────────────────────────────────────


def _invoke(run: LabRun, role: Role, *, user: str, kind: str) -> LLMResponse:
    """Call the LLM for ``role``, log to the message bus, deduct budget.

    Cheap / small / quantised models routinely emit their full response
    twice in a single completion (degenerate sampling) — gpt-5-nano did
    this on every PI message in our baseline, doubling artifact size and
    confusing reviewers.  We sanitise the text in-band via
    :func:`_dedupe_self_repeat` before logging or returning, so every
    downstream consumer (storage, prompts, parsers) sees the clean text.
    """
    template = _safe_load_template(role)
    system = template + "\n\n---\nCurrent topic: " + run.state.topic
    resp = run.call_llm(role_name=role.name, model=role.model,
                         system=system, user=user, config=run.config)
    cleaned = _dedupe_self_repeat(resp.text)
    if cleaned != resp.text:
        # Replace text on the response object — preserve token / cost fields.
        try:
            resp = type(resp)(
                text=cleaned,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                cost_cents=resp.cost_cents,
            )
        except Exception:
            # Some LLMResponse impls may be frozen / dataclass-only;
            # fall back to mutating .text directly.
            try:
                resp.text = cleaned    # type: ignore[attr-defined]
            except Exception:
                pass
    run.storage.append_message(
        run.state.run_id, stage=run.state.stage.value,
        round_=run.state.round, role=role.name, kind=kind,
        content=cleaned,
        meta={"model": role.model,
              "tokens_in": resp.tokens_in,
              "tokens_out": resp.tokens_out,
              "cost_cents": resp.cost_cents},
    )
    if resp.tokens_in or resp.tokens_out or resp.cost_cents:
        run.storage.add_budget(
            run.state.run_id,
            tokens=resp.tokens_in + resp.tokens_out,
            cost_cents=resp.cost_cents,
        )
    return resp


def _safe_load_template(role: Role) -> str:
    try:
        return load_role_template(role)
    except FileNotFoundError:
        return f"You are {role.name}. {role.description}."


def _budget_status(run: LabRun) -> BudgetStatus:
    rec = run.storage.get_run(run.state.run_id)
    tok, cost = run.storage.get_budget(run.state.run_id)
    return BudgetStatus(
        tokens_used=tok,
        tokens_budget=(rec.budget_tokens if rec else None),
        cost_cents_used=cost,
        cost_cents_budget=(rec.budget_cost_cents if rec else None),
    )


# ── Parsing helpers ──────────────────────────────────────────────────────

_NUM_RE = re.compile(r"^\s*\d+[\.\)、]\s*(.+)$")
_CITE_BLOCK_RE = re.compile(
    r"##\s*Citations?\s*\n(.+?)(?:\n##|\Z)", re.IGNORECASE | re.DOTALL,
)


def _extract_numbered(text: str) -> list[str]:
    """Pull numbered list items, preserving order, deduping by content.

    Cheap models sometimes restart the list (1. 2. 3. … 1. 2. 3.) so a
    naïve splitlines+match doubles the items. Drop a line if its first
    80 chars (lower-cased, whitespace-collapsed) match an earlier line.
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        m = _NUM_RE.match(line)
        if not m:
            continue
        content = m.group(1).strip()
        key = " ".join(content.split()).lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(content)
    return out


def _dedupe_self_repeat(text: str) -> str:
    """Trim trailing self-repetition emitted by cheap / quantised models.

    Two patterns we've seen in production:

      1. Exact duplication:    "X" → "XX" (the model finished, then started
         again from scratch).  Detection: text[:n//2] == text[n//2:] after
         whitespace-trim, length above a sanity floor.

      2. Approximate suffix duplicate:  "X" → "X X′" where X′ has the same
         opening lines as X but with minor punctuation drift.  Detection:
         the first 200 chars of the response appear *again* later in the
         same response with at least 80% of their content intact.

    Returns the cleaned text. Never raises — on any unexpected shape we
    just return ``text`` unchanged so we never destroy a legitimate but
    weird response.
    """
    if not text:
        return text
    s = text.strip()
    n = len(s)
    if n < 80:
        return text

    # Pattern 1: exact halves match.
    half = n // 2
    for split in (half, half + 1):
        front = s[:split].rstrip()
        back  = s[n - len(front):].lstrip()
        if front and front == back:
            return front

    # Pattern 2: prefix recurs in the back half.
    prefix_len = min(200, n // 4)
    prefix_norm = " ".join(s[:prefix_len].split()).lower()
    if len(prefix_norm) < 60:
        return text
    back_half = s[half:]
    back_norm = " ".join(back_half.split()).lower()
    pos = back_norm.find(prefix_norm[:80])
    if pos >= 0:
        # Trim from the start of the duplicate back-occurrence in the
        # original (non-normalised) string. We approximate by walking
        # the back half until we've consumed the same number of
        # printable chars as the normalised match position.
        # Simple heuristic: cut at half + that approx position.
        approx_cut = half + max(0, pos - 5)
        candidate = s[:approx_cut].rstrip()
        # Sanity floor: don't trim away >70% of the response.
        if len(candidate) >= 0.3 * n:
            return candidate

    return text


def _extract_citations_block(text: str) -> str:
    m = _CITE_BLOCK_RE.search(text)
    return m.group(1).strip() if m else ""


def _find_first_rq_match(text: str, rqs: list[str]) -> Optional[str]:
    """Pick the RQ the PI's response leans toward."""
    norm = text.lower()
    best = None
    best_overlap = 0
    for rq in rqs:
        words = {w for w in rq.lower().split() if len(w) > 4}
        overlap = sum(1 for w in words if w in norm)
        if overlap > best_overlap:
            best_overlap = overlap
            best = rq
    return best


def _parse_reviewer_verdict(text: str, reviewer_id: str) -> ReviewerVerdict:
    """Extract the JSON envelope; tolerate extra prose around it."""
    blob = _extract_json_blob(text)
    if blob is None:
        return ReviewerVerdict(reviewer_id=reviewer_id, score=5,
                                blocking_issues=["unparseable critique"],
                                suggestions=[],
                                overall="parse_error")
    try:
        score = int(blob.get("score", 5))
    except Exception:
        score = 5
    blocking = list(blob.get("blocking_issues") or [])
    suggestions = list(blob.get("suggestions") or [])
    overall = str(blob.get("overall", ""))
    return ReviewerVerdict(
        reviewer_id=reviewer_id, score=score,
        blocking_issues=[str(x) for x in blocking][:10],
        suggestions=[str(x) for x in suggestions][:10],
        overall=overall,
    )


def _extract_json_blob(text: str) -> Optional[dict]:
    fenced = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL)
    cand = fenced.group(1) if fenced else None
    if not cand:
        m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", text, re.DOTALL)
        cand = m.group(0) if m else None
    if not cand:
        return None
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        try:
            return json.loads(cand.replace("'", '"'))
        except Exception:
            return None


def _parse_citations_from_markdown(text: str):
    """Pull `- Title (Authors, Year). [arXiv:NNN]` style bullets out of the
    Citations section.

    We're permissive: any bullet with a year and at least one capitalised
    word counts as a citation candidate. The verifier handles fuzziness.
    """
    from .verifier import Citation
    out: list[Citation] = []
    block = _extract_citations_block(text) or text
    for line in block.splitlines():
        line = line.strip().lstrip("-•* ").strip()
        if not line or len(line) < 8:
            continue
        # year
        ym = re.search(r"\b(19|20)\d{2}\b", line)
        year = int(ym.group(0)) if ym else None
        # arxiv id
        am = re.search(r"arXiv:?\s*(\d{4}\.\d{4,5})", line, re.IGNORECASE)
        arxiv_id = am.group(1) if am else None
        # rough split: assume "Title (Authors, Year). ..."
        title_part = line
        authors: list[str] = []
        if "(" in line and ")" in line:
            head, paren = line.split("(", 1)
            paren_inner = paren.split(")", 1)[0]
            if "," in paren_inner:
                authors_part, _ = paren_inner.rsplit(",", 1)
                authors = [a.strip() for a in re.split(r",|&|\band\b",
                                                        authors_part)
                           if a.strip()]
            title_part = head.strip(" -")
        if not title_part:
            continue
        out.append(Citation(
            key=re.sub(r"\W+", "_", title_part)[:40].lower() + str(year or ""),
            title=title_part,
            authors=authors,
            year=year,
            arxiv_id=arxiv_id,
        ))
    return out


def _verif_to_dict(v) -> dict:
    return {
        "key": v.citation.key,
        "title": v.citation.title,
        "claimed_authors": v.citation.authors,
        "status": v.status,
        "matched_title": v.matched_title,
        "matched_authors": v.matched_authors,
        "matched_url": v.matched_url,
        "source": v.source,
        "notes": v.notes,
    }
