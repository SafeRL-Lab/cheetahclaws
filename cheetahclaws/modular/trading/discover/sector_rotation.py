"""
discover/sector_rotation.py — find leading sectors and surface their
top-holding stocks.

Methodology:
  1. Compute 1-month and 3-month returns for each SPDR Sector ETF
  2. Rank by 1m return (1m and 3m must both be positive)
  3. Take top N sectors and surface their top holdings
  4. Score each holding by ETF return + position weight (top holdings ≈ heavier)

This is the simplest version of "sector rotation". It captures
intermediate-term style/sector trends (1-3 months) which the academic
literature shows are more reliable than intraday rotation.
"""
from __future__ import annotations

from typing import Iterable

from .types import Discovery
from ..data import fetchers
from ..universe import SECTOR_ETFS, ETF_TO_SECTOR, SECTOR_TOP_HOLDINGS


def scan(top_sectors: int = 2,
         top_per_sector: int = 5,
         progress_cb=None) -> list[Discovery]:
    """Return top holdings of leading sector ETFs."""
    sector_perf: list[tuple[str, str, float, float]] = []  # (sector, etf, ret_1m, ret_3m)

    for i, (sector, etf) in enumerate(SECTOR_ETFS.items()):
        if progress_cb:
            progress_cb(i + 1, len(SECTOR_ETFS), etf)
        result = fetchers.fetch_market_data(etf, interval="1d")
        if result.get("error") or not result.get("data"):
            continue
        rows = result["data"]
        if len(rows) < 70:
            continue
        closes = [r["close"] for r in rows]
        latest = closes[-1]
        # 1m ≈ 21 trading days, 3m ≈ 63
        ret_1m = (latest / closes[-22] - 1.0) if len(closes) >= 22 else 0.0
        ret_3m = (latest / closes[-64] - 1.0) if len(closes) >= 64 else 0.0
        sector_perf.append((sector, etf, ret_1m, ret_3m))

    # Filter to sectors positive on both windows, then sort by 1m return
    positive = [s for s in sector_perf if s[2] > 0 and s[3] > 0]
    positive.sort(key=lambda s: -s[2])
    leaders = positive[:top_sectors]

    if not leaders:
        return []

    out: list[Discovery] = []
    seen: set[str] = set()
    for sector, etf, ret_1m, ret_3m in leaders:
        holdings = SECTOR_TOP_HOLDINGS.get(etf, [])
        for rank, sym in enumerate(holdings[:top_per_sector]):
            if sym in seen:
                continue
            seen.add(sym)
            # Score: ETF 1m return + holding rank bonus (top-of-list weighted heavier)
            rank_bonus = (top_per_sector - rank) / top_per_sector * 0.2
            score = min(1.0, max(0.0, ret_1m * 5.0 + rank_bonus))
            out.append(Discovery(
                symbol=sym, source="sector",
                score=score,
                reason=f"#{rank+1} in {sector} (ETF {etf}: 1m {ret_1m*100:+.1f}%, 3m {ret_3m*100:+.1f}%)",
                details={
                    "sector": sector,
                    "etf": etf,
                    "etf_ret_1m_pct": round(ret_1m * 100, 2),
                    "etf_ret_3m_pct": round(ret_3m * 100, 2),
                    "rank_in_sector": rank + 1,
                },
            ))
    return out
