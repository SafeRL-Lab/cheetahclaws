"""
discover/insider_cluster.py — find tickers with clusters of recent
Form 4 (officer / 10%-holder) filings.

We don't (yet) parse Form 4 XML to distinguish buys from sales. The
heuristic is: a *cluster* of filings — multiple insiders filing within a
short window — is informative regardless of direction:

  - Multiple buys = strong bullish signal
  - Multiple sales = bearish (planned tax events spread over time
    are normally staggered, not clustered)
  - Mixed = internal disagreement

The user can click through to the SEC URLs we surface to verify
direction in seconds. We document this caveat clearly in the output.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from .types import Discovery
from ..alt_data import insider as ins
from ..universe import resolve_universe


def _scan_one(symbol: str, days: int) -> tuple[str, list[dict]]:
    """Fetch Form 4 filings for one ticker. Returns (sym, list of filings)."""
    filings = ins.fetch_recent_insider_filings(symbol, days=days, max_filings=20)
    # SEC fair-use buffer
    time.sleep(0.12)
    return symbol, filings


def scan(
    universe: str | None = "sp100",
    symbols: Iterable[str] | None = None,
    days: int = 30,
    min_cluster_size: int = 3,
    top_n: int = 25,
    max_workers: int = 4,
    progress_cb=None,
) -> list[Discovery]:
    """Return tickers with ≥ `min_cluster_size` Form 4 filings in last `days` days."""
    syms = resolve_universe(universe, symbols)

    out: list[Discovery] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_scan_one, s, days): s for s in syms}
        done = 0
        for fut in as_completed(futures):
            sym = futures[fut]
            done += 1
            if progress_cb:
                progress_cb(done, len(syms), sym)
            try:
                sym, filings = fut.result()
            except Exception:
                continue
            if len(filings) < min_cluster_size:
                continue
            # Score: more filings = higher score, capped
            score_val = min(1.0, len(filings) / 8.0)
            urls = [f["primary_doc_url"] for f in filings[:5]]
            dates = [f["filed_date"] for f in filings[:5]]
            reason = (
                f"{len(filings)} Form 4 filing(s) in last {days} days "
                f"(verify direction via URLs in details)"
            )
            out.append(Discovery(
                symbol=sym, source="insider",
                score=score_val, reason=reason,
                details={
                    "filing_count": len(filings),
                    "days": days,
                    "recent_dates": dates,
                    "urls": urls,
                    "caveat": ("Direction not parsed from Form 4 XML. "
                               "Click URLs to verify buys vs sales."),
                },
            ))

    out.sort(key=lambda d: -d.score)
    return out[:top_n]
