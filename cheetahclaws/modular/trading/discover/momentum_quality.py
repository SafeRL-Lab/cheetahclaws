"""
discover/momentum_quality.py — factor-intersection discovery.

Surfaces tickers that score high on BOTH momentum AND quality. This is
the most defensible factor combination in academic literature: pure
momentum can be high-vol meme stocks, pure quality is often slow-grower
boomer stocks; the intersection finds compounders with positive drift.
"""
from __future__ import annotations

from typing import Iterable

from .types import Discovery
from .. import factors as factor_mod
from ..universe import resolve_universe


def scan(
    universe: str | None = "sp100",
    symbols: Iterable[str] | None = None,
    top_n: int = 15,
    min_momentum: float = 0.5,
    min_quality:  float = 0.5,
    progress_cb=None,
) -> list[Discovery]:
    """Return tickers scoring above thresholds on both momentum AND quality.

    `top_n` clamps how many we return.
    """
    syms = resolve_universe(universe, symbols)
    rows = factor_mod.scan_universe(syms, progress_cb=progress_cb)
    factor_mod.score(rows, weights={"momentum": 0.5, "quality": 0.5,
                                    "low_vol": 0.0, "value": 0.0})

    candidates = []
    for r in rows:
        if (r.momentum_score is not None and r.quality_score is not None
            and r.momentum_score >= min_momentum
            and r.quality_score >= min_quality
            and r.composite_score is not None):
            reason = (
                f"Momentum {r.momentum_score:.2f}, Quality {r.quality_score:.2f}, "
                f"6m return {(r.ret_6m or 0)*100:+.1f}%"
            )
            candidates.append(Discovery(
                symbol=r.symbol,
                source="momentum-quality",
                score=r.composite_score,
                reason=reason,
                details={
                    "momentum_score": r.momentum_score,
                    "quality_score": r.quality_score,
                    "ret_6m": r.ret_6m,
                    "roe": r.roe,
                    "sector": r.sector,
                    "price": r.price,
                },
            ))

    candidates.sort(key=lambda d: -d.score)
    return candidates[:top_n]
