"""
discover/anomaly.py — detect tickers showing unusual market behavior.

Three orthogonal anomaly types per ticker:
  - volume_spike    : today's volume / 90-day median volume
  - price_gap       : abs(today_open - prior_close) / prior_close
  - vol_z           : 5d realised vol z-score against 90d window

Each has its own threshold; an Anomaly is emitted when any threshold is
crossed. Scoring is severity-based: higher z = higher score.

This is the "市场异常" feature the user asked for. It runs in two modes:
  - one-shot: scan a list of symbols once and return a report
  - watchlist: iterate over /trading watch list (default)

Soft-fails per ticker: a single bad symbol doesn't break the scan.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from .types import Discovery
from ..data import fetchers


# ── Thresholds (tunable via scan(thresholds=...)) ─────────────────────────

DEFAULT_THRESHOLDS: dict[str, float] = {
    "volume_spike_ratio":  2.0,    # today vol / 90d median > X
    "price_gap_pct":       3.0,    # |open - prior close| / prior close > X%
    "vol_z_score":         2.0,    # 5d realised vol vs 90d, z-score
    "min_history_bars":    91,     # need ≥ this many bars for vol calc
}


def _scan_one(symbol: str, thresholds: dict[str, float]) -> list[Discovery]:
    """Anomaly checks for one ticker. Returns 0+ Discovery rows."""
    result = fetchers.fetch_market_data(symbol, interval="1d")
    if result.get("error") or not result.get("data"):
        return []
    rows = result["data"]
    if len(rows) < int(thresholds["min_history_bars"]):
        return []

    closes  = [r["close"]  for r in rows]
    opens   = [r["open"]   for r in rows]
    volumes = [r["volume"] for r in rows]

    out: list[Discovery] = []

    # 1. Volume spike: today vs 90d median
    recent_vol = volumes[-1]
    median_90d = sorted(volumes[-91:-1])[len(volumes[-91:-1]) // 2]
    if median_90d > 0:
        ratio = recent_vol / median_90d
        if ratio >= thresholds["volume_spike_ratio"]:
            out.append(Discovery(
                symbol=symbol, source="anomaly",
                score=min(1.0, (ratio - 1) / 5.0),
                reason=f"Volume spike: {ratio:.1f}× 90d median",
                details={
                    "type": "volume_spike",
                    "ratio": round(ratio, 2),
                    "today_volume": int(recent_vol),
                    "median_90d_volume": int(median_90d),
                },
            ))

    # 2. Price gap: today open vs yesterday close
    today_open = opens[-1]
    prior_close = closes[-2]
    if prior_close > 0:
        gap_pct = abs(today_open - prior_close) / prior_close * 100.0
        if gap_pct >= thresholds["price_gap_pct"]:
            direction = "up" if today_open > prior_close else "down"
            out.append(Discovery(
                symbol=symbol, source="anomaly",
                score=min(1.0, gap_pct / 10.0),
                reason=f"Price gap {direction} {gap_pct:.1f}% at open",
                details={
                    "type": "price_gap",
                    "direction": direction,
                    "gap_pct": round(gap_pct, 2),
                    "prior_close": prior_close,
                    "today_open": today_open,
                },
            ))

    # 3. Vol z-score: 5d realised vs 90d distribution
    log_rets = []
    for i in range(1, len(closes)):
        if closes[i] > 0 and closes[i - 1] > 0:
            log_rets.append(math.log(closes[i] / closes[i - 1]))

    if len(log_rets) >= 90:
        recent_5 = log_rets[-5:]
        baseline_90 = log_rets[-90:-5]

        def _stdev(xs):
            mean = sum(xs) / len(xs)
            v = sum((x - mean) ** 2 for x in xs) / len(xs)
            return math.sqrt(v) if v > 0 else 0.0

        sigma_5d = _stdev(recent_5)
        rolling_sigmas = [
            _stdev(log_rets[i:i + 5])
            for i in range(0, len(baseline_90) - 5, 5)
        ]
        if rolling_sigmas:
            mean_sigma = sum(rolling_sigmas) / len(rolling_sigmas)
            sigma_of_sigma = _stdev(rolling_sigmas)
            if sigma_of_sigma > 0:
                z = (sigma_5d - mean_sigma) / sigma_of_sigma
                if z >= thresholds["vol_z_score"]:
                    out.append(Discovery(
                        symbol=symbol, source="anomaly",
                        score=min(1.0, z / 5.0),
                        reason=f"Volatility spike: 5d realised vol z={z:.2f} vs 90d",
                        details={
                            "type": "vol_spike",
                            "z_score": round(z, 2),
                            "sigma_5d": round(sigma_5d, 4),
                            "mean_sigma_baseline": round(mean_sigma, 4),
                        },
                    ))

    return out


def scan(symbols: Iterable[str],
         thresholds: dict[str, float] | None = None,
         max_workers: int = 4,
         progress_cb=None) -> list[Discovery]:
    """Run anomaly scan on a list of symbols. Soft-fails per ticker."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    syms = [s.upper().strip() for s in symbols if s.strip()]
    if not syms:
        return []

    out: list[Discovery] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_scan_one, s, th): s for s in syms}
        done = 0
        for fut in as_completed(futures):
            sym = futures[fut]
            done += 1
            if progress_cb:
                progress_cb(done, len(syms), sym)
            try:
                hits = fut.result()
            except Exception:
                hits = []
            out.extend(hits)

    out.sort(key=lambda d: -d.score)
    return out


def render_anomaly_report(hits: list[Discovery]) -> str:
    """Markdown report grouped by anomaly type."""
    if not hits:
        return "_No anomalies detected._"

    lines = [f"# Market Anomalies ({len(hits)} hits)", ""]
    by_type: dict[str, list[Discovery]] = {}
    for h in hits:
        t = h.details.get("type", "unknown")
        by_type.setdefault(t, []).append(h)

    for t, items in by_type.items():
        lines.append(f"## {t.replace('_', ' ').title()} ({len(items)})")
        for h in sorted(items, key=lambda x: -x.score):
            lines.append(f"- **{h.symbol}** — {h.reason} _(score {h.score:.2f})_")
        lines.append("")
    return "\n".join(lines)
