"""
trends.py — Google Trends search interest for a ticker / company name.

Soft-fails when `pytrends` isn't installed — the analyze pipeline
keeps working without it.

What it surfaces:
  - 30-day search trend (trending up / down / flat)
  - Spike detection (today vs. 90-day median)

This is a noisy, retail-flavoured signal — a Reddit pump shows up here
before it shows up in price. Useful as a contrarian filter, not a
momentum trigger.
"""
from __future__ import annotations

import statistics
from typing import Any


def fetch_interest(symbol: str, lookback_days: int = 90) -> dict[str, Any]:
    """Return Google Trends data, or {} on missing dep / failure."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        return {}

    try:
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(4, 8))
        pytrends.build_payload([symbol], timeframe=f"today {lookback_days}-d")
        df = pytrends.interest_over_time()
    except Exception:
        return {}

    if df is None or df.empty:
        return {}

    series = df[symbol].tolist()
    if len(series) < 7:
        return {}

    return {
        "series":  series,
        "latest":  series[-1],
        "median":  statistics.median(series),
        "p90":     sorted(series)[int(len(series) * 0.9)],
        "p10":     sorted(series)[int(len(series) * 0.1)],
        "trend_7d": series[-1] - (series[-7] if len(series) >= 7 else series[0]),
    }


def render_trends_block(symbol: str, lookback_days: int = 90) -> str:
    """Markdown block on Google search interest. Empty when unavailable."""
    info = fetch_interest(symbol, lookback_days)
    if not info:
        return ""

    latest = info["latest"]
    median = info["median"]
    p90 = info["p90"]
    trend7 = info["trend_7d"]

    if latest > p90 * 1.05:
        regime = "SPIKE — public attention surge"
    elif latest > median * 1.5:
        regime = "Elevated"
    elif latest < median * 0.5:
        regime = "Quiet"
    else:
        regime = "Normal"

    direction = "+" if trend7 > 5 else "-" if trend7 < -5 else "≈"

    lines = [
        f"## Google Trends ({symbol})",
        f"- Search interest: **{regime}** (latest {latest}, median {median:.0f}, "
        f"p90 {p90}, 7-day {direction}{abs(trend7)})",
    ]
    if regime.startswith("SPIKE"):
        lines.append(
            "- ⚠ Retail attention spikes precede mean-reversion more often than "
            "trend-continuation. Treat as a fade signal unless backed by a "
            "real catalyst (earnings beat, M&A, etc.)."
        )
    return "\n".join(lines)
