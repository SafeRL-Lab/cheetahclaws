"""
ranker.py — composite "what's worth investing in NOW" score.

Given a candidate symbol set (from /trading discover, watchlist, or a
user-supplied list), compute a composite score that blends:

  - Quant factors      : momentum + quality (from factors.py)
  - Discovery sources  : insider clusters + earnings beats + sector tailwinds
  - Calibration weight : if the agent has had this symbol/sector right before,
                         boost; if it's been wrong, dampen

Output is a ranked table the user can use as their "top of mind" list
to actually run /trading analyze on. This is NOT a recommendation
engine — it's a triage tool that says "of these 100 names, focus on
these 10".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from . import factors as factor_mod
from . import paper_trader, calibration
from .discover import orchestrator as disc_orch
from .universe import resolve_universe


@dataclass
class RankRow:
    symbol:           str
    aggregate_score:  float
    factor_score:     float = 0.0
    discovery_score:  float = 0.0
    calibration_adj:  float = 0.0
    reasons:          list[str] = field(default_factory=list)
    sectors:          list[str] = field(default_factory=list)


def rank(
    universe: str | None = "sp100",
    symbols: Iterable[str] | None = None,
    use_discovery: bool = True,
    use_calibration: bool = True,
    top_n: int = 15,
    progress_cb=None,
) -> list[RankRow]:
    """Compute the ranked list."""
    syms = resolve_universe(universe, symbols)

    # Step 1: factor scores
    rows = factor_mod.scan_universe(syms, progress_cb=progress_cb)
    factor_mod.score(rows, weights={"momentum": 0.5, "quality": 0.5,
                                    "low_vol": 0.0, "value": 0.0})
    factor_by_sym = {r.symbol: r for r in rows}

    # Step 2: discovery (optional, slower)
    discovery_by_sym: dict[str, dict] = {}
    if use_discovery:
        # Don't re-run sector rotation since it's a separate universe;
        # just run the symbol-targeted scanners.
        try:
            disc_result = disc_orch.run(
                sources=["insider", "earnings", "momentum-quality"],
                universe=universe, symbols=symbols,
                progress_cb=progress_cb,
            )
            for entry in disc_result["ranked"]:
                discovery_by_sym[entry["symbol"]] = entry
        except Exception:
            pass

    # Step 3: calibration weighting (optional)
    calibration_by_sector: dict[str, float] = {}
    if use_calibration:
        try:
            stats = calibration.compute_calibration()
            # Per-sector adjustment: closed trades > 5 in that sector
            # determines whether to up- or down-weight.
            # For the simple v1 we use the overall mean as a global tilt.
            overall = stats.get("overall", {})
            mean_ret = overall.get("mean_return_pct", 0.0)
            global_tilt = max(-0.1, min(0.1, mean_ret / 10.0))  # ±10pp at most
            calibration_by_sector["__global__"] = global_tilt
        except Exception:
            pass

    # Step 4: combine
    out: list[RankRow] = []
    for sym in syms:
        f = factor_by_sym.get(sym)
        d = discovery_by_sym.get(sym)
        if f is None and d is None:
            continue

        factor_score = f.composite_score if f and f.composite_score is not None else 0.0
        discovery_score = d["aggregate_score"] if d else 0.0
        cal_adj = calibration_by_sector.get("__global__", 0.0)

        # Aggregate: 50% factors, 30% discovery, calibration is a small tilt
        agg = 0.5 * factor_score + 0.3 * discovery_score + cal_adj

        reasons = []
        if f and f.composite_score is not None:
            reasons.append(
                f"Factor {f.composite_score:.2f} (mom {f.momentum_score or 0:.2f}, "
                f"qual {f.quality_score or 0:.2f})"
            )
        if d:
            srcs = " · ".join(set(d["sources"]))
            reasons.append(f"Discovery [{srcs}] score {d['aggregate_score']:.2f}")
        if cal_adj != 0:
            reasons.append(f"Calibration tilt {cal_adj:+.2f}")

        sectors = []
        if f and f.sector:
            sectors.append(f.sector)

        out.append(RankRow(
            symbol=sym,
            aggregate_score=agg,
            factor_score=factor_score,
            discovery_score=discovery_score,
            calibration_adj=cal_adj,
            reasons=reasons,
            sectors=sectors,
        ))

    out.sort(key=lambda r: -r.aggregate_score)
    return out[:top_n]


def render_rank_report(rows: list[RankRow], top: int = 15) -> str:
    """Markdown ranked report."""
    if not rows:
        return "_No candidates ranked. Try a smaller universe or different symbols._"

    lines = [f"# Investment Ranking — top {min(top, len(rows))} of {len(rows)}", ""]
    lines.append("| # | Symbol | Score | Factor | Discovery | Sector | Why |")
    lines.append("|---:|---|---:|---:|---:|---|---|")
    for i, r in enumerate(rows[:top], 1):
        why = "; ".join(r.reasons)[:120]
        sectors = ", ".join(r.sectors) or "—"
        lines.append(
            f"| {i} | **{r.symbol}** | {r.aggregate_score:.3f} | "
            f"{r.factor_score:.2f} | {r.discovery_score:.2f} | {sectors} | {why} |"
        )
    lines.append("")
    lines.append("> Combine: 50% factor (momentum + quality), 30% discovery "
                 "(insider/earnings/momentum-quality), tilt by historical calibration.")
    lines.append("> **Use this as a triage list to decide which names to "
                 "`/trading analyze` next**, not as a recommendation in itself.")
    return "\n".join(lines)
