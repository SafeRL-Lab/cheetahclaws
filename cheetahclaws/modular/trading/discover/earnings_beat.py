"""
discover/earnings_beat.py — find tickers that just reported a positive
earnings surprise AND held up post-print (continuation, not pop-and-fade).

Signal: stocks that beat consensus EPS by ≥10% AND closed up on the
report day AND haven't given back the gain in the next 1-2 sessions.
This is "post-earnings drift" — well-documented in academic literature
(Bernard-Thomas 1989) and still empirically tradable.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .types import Discovery
from ..universe import resolve_universe


def _score_one(symbol: str, lookback_days: int = 14) -> Discovery | None:
    """Check if symbol had a recent positive earnings surprise + continuation."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        tk = yf.Ticker(symbol)

        # Earnings dates (with actual + estimate)
        df = None
        try:
            df = tk.get_earnings_dates(limit=4)
        except Exception:
            df = getattr(tk, "earnings_dates", None)
        if df is None or len(df) == 0:
            return None

        # Find the most recent past earnings within lookback window
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=lookback_days)
        past_recent = []
        for idx, row in df.iterrows():
            try:
                ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if cutoff <= ts <= now:
                    past_recent.append((ts, row))
            except Exception:
                continue
        if not past_recent:
            return None
        past_recent.sort(key=lambda x: -x[0].timestamp())
        ts, row = past_recent[0]

        eps_actual = _safe(row.get("EPS Estimate"))   # column naming varies
        eps_estim  = _safe(row.get("EPS Estimate"))
        # The columns are often "EPS Estimate" + "Reported EPS" + "Surprise(%)"
        actual = _safe(row.get("Reported EPS"))
        estim  = _safe(row.get("EPS Estimate"))
        surprise_pct = _safe(row.get("Surprise(%)"))

        if surprise_pct is None and actual is not None and estim and estim != 0:
            surprise_pct = (actual - estim) / abs(estim) * 100.0

        if surprise_pct is None or surprise_pct < 10.0:
            return None  # require ≥10% beat

        # Now check post-print price action
        hist = tk.history(period="1mo", interval="1d", auto_adjust=False)
        if hist is None or len(hist) < 5:
            return None
        closes = hist["Close"].dropna()
        # Find bar at/after the earnings date
        idxs = [i for i, dt in enumerate(closes.index)
                if dt.to_pydatetime().date() >= ts.date()]
        if not idxs or idxs[0] >= len(closes) - 1:
            return None
        i_earn = idxs[0]
        prior_close = float(closes.iloc[i_earn - 1]) if i_earn > 0 else None
        if prior_close is None or prior_close <= 0:
            return None
        latest_close = float(closes.iloc[-1])
        post_return = (latest_close - prior_close) / prior_close * 100.0

        # Need: positive post-print AND not faded back to flat
        if post_return < 1.0:
            return None

        score = min(1.0, (surprise_pct / 100.0 + post_return / 50.0))
        reason = (
            f"Beat by {surprise_pct:.1f}% on {ts.date()}; "
            f"+{post_return:.1f}% since"
        )
        return Discovery(
            symbol=symbol, source="earnings",
            score=score, reason=reason,
            details={
                "surprise_pct": round(surprise_pct, 2),
                "post_print_return_pct": round(post_return, 2),
                "earnings_date": ts.date().isoformat(),
                "reported_eps": actual,
                "estimate_eps": estim,
            },
        )
    except Exception:
        return None


def _safe(x):
    if x is None:
        return None
    try:
        v = float(x)
        import math
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def scan(
    universe: str | None = "sp100",
    symbols: Iterable[str] | None = None,
    lookback_days: int = 14,
    top_n: int = 15,
    max_workers: int = 4,
    progress_cb=None,
) -> list[Discovery]:
    """Return tickers that beat earnings by ≥10% in last `lookback_days`
    AND have held up post-print."""
    syms = resolve_universe(universe, symbols)
    out: list[Discovery] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_one, s, lookback_days): s for s in syms}
        done = 0
        for fut in as_completed(futures):
            sym = futures[fut]
            done += 1
            if progress_cb:
                progress_cb(done, len(syms), sym)
            try:
                d = fut.result()
            except Exception:
                d = None
            if d:
                out.append(d)
    out.sort(key=lambda d: -d.score)
    return out[:top_n]
