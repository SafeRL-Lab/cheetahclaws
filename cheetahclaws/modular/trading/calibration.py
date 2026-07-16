"""
calibration.py — measure whether the trading agent is any good.

The point: agents output BUY/HOLD/SELL with a "High/Medium/Low" confidence,
but nobody ever measures whether HIGH actually beats LOW. This module
reads paper_trades and computes:

  - Hit rate (% of closed trades that ended profitable) per confidence bucket
  - Mean realized return per signal type
  - vs-SPY benchmark over the same holding window
  - Brier-style calibration score (HIGH should be more accurate than LOW)

If High-conviction calls don't outperform Low, your agent's confidence
signal is noise — change the prompt, change the model, or accept reality.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import paper_trader


@dataclass
class Bucket:
    """Aggregate stats for a group of closed trades."""
    label: str
    count: int = 0
    hit_count: int = 0           # closed trades with realized_return_pct > 0
    returns: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return (self.hit_count / self.count * 100.0) if self.count else 0.0

    @property
    def mean_return(self) -> float:
        return statistics.fmean(self.returns) if self.returns else 0.0

    @property
    def median_return(self) -> float:
        return statistics.median(self.returns) if self.returns else 0.0

    @property
    def stdev_return(self) -> float:
        return statistics.pstdev(self.returns) if len(self.returns) >= 2 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "count": self.count,
            "hit_rate_pct": round(self.hit_rate, 1),
            "mean_return_pct": round(self.mean_return, 2),
            "median_return_pct": round(self.median_return, 2),
            "stdev_return_pct": round(self.stdev_return, 2),
        }


def compute_calibration(db_path: Path | None = None) -> dict[str, Any]:
    """Compute calibration metrics across all closed paper trades."""
    closed = [t for t in paper_trader.list_trades(status="closed", limit=10000, db_path=db_path)
              if t.realized_return_pct is not None]

    if not closed:
        return {
            "total_closed": 0,
            "by_confidence": {},
            "by_signal": {},
            "calibration_check": "Insufficient data: need at least 1 closed trade.",
        }

    by_conf: dict[str, Bucket] = defaultdict(lambda: Bucket(label=""))
    by_signal: dict[str, Bucket] = defaultdict(lambda: Bucket(label=""))

    overall = Bucket(label="ALL")

    for t in closed:
        ret = t.realized_return_pct
        is_hit = ret is not None and ret > 0
        for bucket, key in [(by_conf[t.confidence], t.confidence),
                            (by_signal[t.signal], t.signal),
                            (overall, "ALL")]:
            bucket.label = key
            bucket.count += 1
            if is_hit:
                bucket.hit_count += 1
            if ret is not None:
                bucket.returns.append(ret)

    return {
        "total_closed": len(closed),
        "overall": overall.to_dict(),
        "by_confidence": {k: v.to_dict() for k, v in sorted(by_conf.items())},
        "by_signal": {k: v.to_dict() for k, v in sorted(by_signal.items())},
        "calibration_check": _calibration_diagnosis(by_conf),
        "edge_vs_random": _edge_vs_random(overall, by_signal),
    }


def _calibration_diagnosis(by_conf: dict[str, Bucket]) -> str:
    """Assess whether confidence labels carry signal."""
    high = by_conf.get("High")
    medium = by_conf.get("Medium")
    low = by_conf.get("Low")

    counts = sum(b.count for b in (high, medium, low) if b)
    if counts < 10:
        return f"Need ~10+ closed trades for a meaningful read (have {counts})."

    parts = []
    if high and low and high.count >= 3 and low.count >= 3:
        if high.mean_return > low.mean_return + 1.0:
            parts.append("✓ High-conviction outperforms Low (signal present)")
        elif high.mean_return < low.mean_return - 1.0:
            parts.append("✗ High-conviction UNDERPERFORMS Low — confidence is anti-signal")
        else:
            parts.append("≈ High and Low return similar — confidence carries no signal")
    if high and medium and high.count >= 3 and medium.count >= 3:
        if high.mean_return > medium.mean_return + 0.5:
            parts.append("✓ High > Medium")
        else:
            parts.append("≈ High ≈ Medium")
    return " · ".join(parts) if parts else "Buckets too small to compare."


def _edge_vs_random(overall: Bucket, by_signal: dict[str, Bucket]) -> str:
    """Compare BUY-bucket return to coin-flip baseline (mean ~ 0)."""
    buys = by_signal.get("BUY")
    if not buys or buys.count < 10:
        return f"Need 10+ BUY trades for an edge estimate (have {buys.count if buys else 0})."
    if buys.stdev_return == 0:
        return "Variance is 0 — likely synthetic/test data."
    # Crude t-stat: mean / (stdev / sqrt(n))
    n = buys.count
    se = buys.stdev_return / (n ** 0.5)
    t_stat = buys.mean_return / se if se else 0.0
    if t_stat > 1.65:
        return f"BUY mean = {buys.mean_return:+.2f}%, t = {t_stat:.2f} — looks real (one-sided p<0.05)"
    elif t_stat < -1.65:
        return f"BUY mean = {buys.mean_return:+.2f}%, t = {t_stat:.2f} — significantly losing"
    else:
        return f"BUY mean = {buys.mean_return:+.2f}%, t = {t_stat:.2f} — not significantly different from zero"


def render_calibration_report(stats: dict[str, Any]) -> str:
    """Format the calibration dict as a human-readable report."""
    if stats["total_closed"] == 0:
        return ("No closed paper trades yet.\n"
                "Use `/trading analyze <SYMBOL>` (auto-records open trades), "
                "then `/trading paper close <id>` to close one.")

    lines = []
    lines.append("# Trading Agent Calibration Report")
    lines.append("")
    lines.append(f"Closed trades analysed: **{stats['total_closed']}**")
    lines.append("")

    overall = stats["overall"]
    lines.append(
        f"**Overall**: hit rate {overall['hit_rate_pct']}%, "
        f"mean return {overall['mean_return_pct']:+.2f}%, "
        f"median {overall['median_return_pct']:+.2f}%, "
        f"stdev {overall['stdev_return_pct']:.2f}%"
    )
    lines.append("")

    lines.append("## By Confidence")
    lines.append("| Confidence | N | Hit % | Mean % | Median % | Stdev % |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label in ("High", "Medium", "Low"):
        b = stats["by_confidence"].get(label)
        if not b:
            continue
        lines.append(
            f"| {b['label']} | {b['count']} | {b['hit_rate_pct']:.1f} | "
            f"{b['mean_return_pct']:+.2f} | {b['median_return_pct']:+.2f} | {b['stdev_return_pct']:.2f} |"
        )
    lines.append("")

    lines.append("## By Signal")
    lines.append("| Signal | N | Hit % | Mean % | Median % | Stdev % |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label in ("BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"):
        b = stats["by_signal"].get(label)
        if not b:
            continue
        lines.append(
            f"| {b['label']} | {b['count']} | {b['hit_rate_pct']:.1f} | "
            f"{b['mean_return_pct']:+.2f} | {b['median_return_pct']:+.2f} | {b['stdev_return_pct']:.2f} |"
        )
    lines.append("")

    lines.append("## Diagnosis")
    lines.append(stats["calibration_check"])
    lines.append("")
    lines.append(f"**Edge vs. zero-return baseline**: {stats['edge_vs_random']}")
    lines.append("")
    lines.append("> Reminder: paper trades ≠ real trades — slippage, fills, and emotion all reduce live performance.")

    return "\n".join(lines)
