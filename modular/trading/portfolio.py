"""
portfolio.py — mean-variance portfolio optimizer (Markowitz, no cvxpy).

Given a list of candidates with expected returns + recent OHLCV history,
compute the long-only weights that maximise Sharpe under:

  - sum(weights) <= 1.0           (cash allowed; never leveraged)
  - 0 <= weight_i <= max_weight   (single-name cap)
  - optional sector caps          (max % per sector)

Uses scipy.optimize.minimize (SLSQP) which is plenty fast for the
~10-20-name universe a single user will have. Returns are estimated
from log-returns of the supplied price series (caller decides whether
to override with LLM-derived expected returns).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

# numpy is part of the [trading] extra; gate the eager import so that
# `pip install .` (no extras) still ships an importable wheel — the
# tests/test_packaging.py contract asserts every package in the wheel
# imports cleanly without optional deps.  Callers like ``optimize()``
# raise the original ImportError on first use if numpy is missing.
try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — exercised only on lean installs
    np = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 — annotation-only, real binding above


@dataclass
class Candidate:
    symbol: str
    closes: list[float]                 # OHLCV close history
    sector: str | None = None
    expected_return: float | None = None  # annualised, override; None = use realised mean


@dataclass
class OptimizationResult:
    weights:        dict[str, float]
    expected_return: float
    expected_vol:    float
    sharpe:          float
    diagnostics:     dict[str, Any]


def _log_returns(closes: list[float]) -> np.ndarray:
    arr = np.asarray(closes, dtype=float)
    return np.diff(np.log(arr[arr > 0]))


def optimize(
    candidates: list[Candidate],
    max_weight: float = 0.20,
    risk_free_annual: float = 0.04,
    bars_per_year: int = 252,
    sector_caps: dict[str, float] | None = None,
) -> OptimizationResult:
    """Run mean-variance optimization. Long-only, never leveraged.

    Returns the OptimizationResult with weights summing to ≤ 1.0
    (uninvested cash if MV concludes that's optimal — possible when
    correlations are extreme).
    """
    if not candidates:
        return OptimizationResult({}, 0.0, 0.0, 0.0, {"reason": "no candidates"})
    syms = [c.symbol for c in candidates]
    n = len(candidates)

    # Build aligned log-return matrix. Trim to common-min length so
    # cov matrix is well-defined.
    rets_per_sym = [_log_returns(c.closes) for c in candidates]
    if any(len(r) < 30 for r in rets_per_sym):
        return OptimizationResult({s: 0.0 for s in syms}, 0.0, 0.0, 0.0,
                                  {"reason": "insufficient history (need 30+ bars)"})

    L = min(len(r) for r in rets_per_sym)
    rets_matrix = np.array([r[-L:] for r in rets_per_sym])  # shape (n, L)

    mu_realised = rets_matrix.mean(axis=1) * bars_per_year
    if any(c.expected_return is not None for c in candidates):
        mu = np.array([
            c.expected_return if c.expected_return is not None else mu_realised[i]
            for i, c in enumerate(candidates)
        ])
    else:
        mu = mu_realised

    cov = np.cov(rets_matrix) * bars_per_year
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    rf = risk_free_annual

    # Negative Sharpe — minimize this.
    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(np.dot(w, mu))
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            return 0.0
        return -(port_ret - rf) / math.sqrt(port_var)

    bounds = [(0.0, max_weight) for _ in range(n)]
    constraints = [{"type": "ineq", "fun": lambda w: 1.0 - sum(w)}]

    if sector_caps:
        # Map symbol -> sector -> indices belonging to that sector
        sec_idx: dict[str, list[int]] = {}
        for i, c in enumerate(candidates):
            if c.sector:
                sec_idx.setdefault(c.sector, []).append(i)
        for sec, idx_list in sec_idx.items():
            cap = sector_caps.get(sec)
            if cap is None:
                continue
            constraints.append({
                "type": "ineq",
                "fun": (lambda w, idx=idx_list, cap=cap: cap - sum(w[i] for i in idx)),
            })

    # Scipy lazy-import: only need it here, keeps top-level import light.
    from scipy.optimize import minimize

    x0 = np.full(n, min(max_weight, 1.0 / n))
    res = minimize(
        neg_sharpe, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-7},
    )

    if not res.success:
        # Fall back to equal-weight inside the cap
        w = np.full(n, min(max_weight, 1.0 / n))
    else:
        w = np.clip(res.x, 0.0, max_weight)
        # Re-project sum constraint if SLSQP gave us a slightly invalid solution
        if w.sum() > 1.0:
            w = w / w.sum()

    port_ret = float(np.dot(w, mu))
    port_var = float(w @ cov @ w)
    port_vol = math.sqrt(max(port_var, 0.0))
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0.0

    return OptimizationResult(
        weights={s: round(float(wi), 4) for s, wi in zip(syms, w)},
        expected_return=round(port_ret, 4),
        expected_vol=round(port_vol, 4),
        sharpe=round(sharpe, 3),
        diagnostics={
            "n_candidates":     n,
            "bars_used":        L,
            "max_weight":       max_weight,
            "sector_caps":      sector_caps,
            "scipy_success":    bool(res.success),
            "scipy_message":    str(res.message),
        },
    )


def render_optimization_report(result: OptimizationResult) -> str:
    """Format the result as a markdown report."""
    lines = ["# Portfolio Optimization (Mean-Variance, Long-Only)"]
    lines.append("")
    if not result.weights:
        lines.append(result.diagnostics.get("reason", "Optimization failed."))
        return "\n".join(lines)

    invested = sum(result.weights.values())
    cash = max(0.0, 1.0 - invested)

    lines.append(f"**Expected annual return**: {result.expected_return * 100:+.2f}%")
    lines.append(f"**Expected annual vol**:    {result.expected_vol * 100:.2f}%")
    lines.append(f"**Sharpe**:                 {result.sharpe:+.3f}")
    lines.append(f"**Invested**:               {invested * 100:.1f}%   (cash {cash * 100:.1f}%)")
    lines.append("")

    lines.append("## Target weights")
    lines.append("| Symbol | Weight |")
    lines.append("|---|---:|")
    for sym, w in sorted(result.weights.items(), key=lambda kv: -kv[1]):
        if w < 0.001:
            continue
        lines.append(f"| {sym} | {w * 100:.1f}% |")

    diag = result.diagnostics
    lines.append("")
    lines.append(
        f"_Diagnostics: {diag['n_candidates']} candidates, "
        f"{diag['bars_used']} bars used, single-name cap "
        f"{diag['max_weight'] * 100:.0f}%, scipy={diag['scipy_message']}_"
    )

    return "\n".join(lines)
