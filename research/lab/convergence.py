"""research/lab/convergence.py — when does a stage / paper "converge"?

Two layers:

  1. Per-stage convergence — the reviewer-author loop within a single
     stage (e.g. drafting a section, designing the methodology).
     Decision is made by :func:`decide_advance` after each round.
  2. Run-level convergence — the orchestrator stops the whole run when
     the run-level budget is exhausted (tokens, USD cents, wall-time)
     OR every required artifact is "sealed" (final report written).
     This is enforced inside the orchestrator, not here.

The decision rule is intentionally simple — too clever a heuristic
risks looking-good-but-being-wrong (the problem we're trying to avoid
in the first place). Counts of unanimous-pass and blocking-issues
are what drive "advance vs another round vs force-stop".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Reviewer verdict ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReviewerVerdict:
    """One reviewer's structured verdict on a draft."""
    reviewer_id: str               # "reviewer_1", "reviewer_2", ...
    score: int                     # 1–10 numeric quality score
    blocking_issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    overall: str = ""              # one-line summary

    @property
    def passes(self) -> bool:
        """A reviewer passes when there are no blocking issues AND score ≥ 7."""
        return not self.blocking_issues and self.score >= 7


@dataclass
class ConvergenceConfig:
    """Tunable knobs for the convergence rule.

    The defaults are chosen so that:
      * a draft with 2/3 reviewers passing advances after 1-2 rounds
      * 5 rounds is the hard ceiling so a model that loves to nitpick
        cannot block a stage indefinitely
      * a draft with 0/3 passing falls into a designer-revisit path
    """
    n_reviewers_required_pass: int = 2     # of typically 3
    score_pass_threshold: int = 7
    max_rounds: int = 5
    abort_after_n_rounds_with_zero_pass: int = 3


@dataclass
class ConvergenceDecision:
    advance: bool
    reason: str
    needs_redesign: bool = False           # signal back-edge in the stage graph
    rounds_remaining: int = 0


def decide_advance(verdicts: list[ReviewerVerdict],
                   *, round_index: int,
                   config: Optional[ConvergenceConfig] = None,
                   ) -> ConvergenceDecision:
    """Given this round's reviewer verdicts, decide if the stage advances.

    ``round_index`` is 1-based: round 1 is the first reviewer pass.
    """
    cfg = config or ConvergenceConfig()
    n_pass = sum(1 for v in verdicts if v.passes)
    n_total = len(verdicts) or 1
    rounds_remaining = max(0, cfg.max_rounds - round_index)

    # Hard ceiling: out of rounds → force-advance with a noted compromise.
    if round_index >= cfg.max_rounds:
        return ConvergenceDecision(
            advance=True,
            reason=f"hit max_rounds={cfg.max_rounds}; advancing with"
                   f" {n_pass}/{n_total} reviewers passing",
            rounds_remaining=0,
        )

    # Strong-pass: required quorum reached.
    if n_pass >= cfg.n_reviewers_required_pass:
        return ConvergenceDecision(
            advance=True,
            reason=f"{n_pass}/{n_total} reviewers pass (≥ {cfg.n_reviewers_required_pass} required)",
            rounds_remaining=rounds_remaining,
        )

    # Total stall — early bail with redesign flag so the orchestrator can
    # send us back to an earlier stage (designer / outline) rather than
    # wasting more rounds on a fundamentally broken draft.
    if (n_pass == 0
            and round_index >= cfg.abort_after_n_rounds_with_zero_pass):
        return ConvergenceDecision(
            advance=False,
            needs_redesign=True,
            reason=f"{round_index} rounds with 0/{n_total} passing — needs redesign",
            rounds_remaining=rounds_remaining,
        )

    return ConvergenceDecision(
        advance=False,
        reason=f"{n_pass}/{n_total} pass; iterate (round {round_index + 1}"
               f" of max {cfg.max_rounds})",
        rounds_remaining=rounds_remaining,
    )


# ── Budget enforcement ────────────────────────────────────────────────────


@dataclass
class BudgetStatus:
    tokens_used: int
    tokens_budget: Optional[int]
    cost_cents_used: int
    cost_cents_budget: Optional[int]

    def fraction_used(self) -> float:
        """Return max fractional consumption across both axes (0..1+)."""
        frac_tokens = (self.tokens_used / self.tokens_budget
                        if self.tokens_budget else 0.0)
        frac_cost = (self.cost_cents_used / self.cost_cents_budget
                      if self.cost_cents_budget else 0.0)
        return max(frac_tokens, frac_cost)

    @property
    def exceeded(self) -> bool:
        if self.tokens_budget and self.tokens_used >= self.tokens_budget:
            return True
        if self.cost_cents_budget and self.cost_cents_used >= self.cost_cents_budget:
            return True
        return False
