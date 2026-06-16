"""
earnings.py — detect upcoming earnings dates so the agent can flag event risk.

Buying a stock 2 days before earnings is gambling, not investing. yfinance's
Ticker.calendar / Ticker.earnings_dates expose the upcoming reporting date.
We surface it as a ⚠️ warning in the analyze prompt when the report is
within 7 days.

Soft-fail throughout: if yfinance is missing or the ticker is non-equity
(crypto, A-share via akshare, etc.), return an empty notice rather than
breaking the analysis.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def upcoming_earnings(symbol: str) -> dict[str, Any]:
    """Return dict with next earnings date + days_until, or empty dict if unknown.

    Returns:
        {"date": "2026-05-28", "days_until": 5, "session": "amc"} on hit
        {"error": "..."} on hard failure
        {} when there's simply no upcoming report (e.g., crypto)
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "yfinance not installed"}

    try:
        tk = yf.Ticker(symbol)
        # earnings_dates is a DataFrame; use it first since calendar is
        # often empty for US tickers in newer yfinance versions.
        df = None
        try:
            df = tk.get_earnings_dates(limit=8)
        except Exception:
            df = getattr(tk, "earnings_dates", None)

        if df is None or len(df) == 0:
            return {}

        # Index is timezone-aware datetimes (UTC). Find the soonest future row.
        now = datetime.now(timezone.utc)
        future = []
        for idx, row in df.iterrows():
            try:
                ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > now:
                    future.append((ts, row))
            except Exception:
                continue

        if not future:
            return {}

        future.sort(key=lambda x: x[0])
        ts, row = future[0]
        days = (ts - now).days

        # session inference: yfinance has 'Earnings Call Time' on some tickers
        session = ""
        try:
            ect = row.get("Earnings Call Time")
            if isinstance(ect, str) and ect:
                # "AMC" = after market close, "BMO" = before market open
                session = ect.upper().strip()
        except Exception:
            pass

        return {
            "date": ts.date().isoformat(),
            "days_until": max(days, 0),
            "session": session or "TBD",
        }
    except Exception as e:
        return {"error": f"earnings lookup failed: {type(e).__name__}: {e}"}


def render_earnings_warning(symbol: str, threshold_days: int = 7) -> str:
    """Markdown block flagging upcoming earnings, or empty string if none/clear."""
    info = upcoming_earnings(symbol)
    if info.get("error") or not info:
        return ""

    days = info.get("days_until")
    if days is None:
        return ""

    if days > threshold_days:
        # Still useful to mention so the agent doesn't accidentally recommend
        # a position that straddles earnings.
        return (
            f"## ⚠️ Earnings Calendar\n"
            f"- {symbol} reports on **{info['date']}** ({days} days away, session: {info['session']}).\n"
            f"- A {days}-day position window will straddle this event — factor "
            f"into your time horizon and stop placement.\n"
        )

    return (
        f"## 🚨 EARNINGS RISK — {info['date']} ({days} days)\n"
        f"- {symbol} reports on **{info['date']}**, session: {info['session']}.\n"
        f"- Expected move on report day: typically 3-8% (sometimes 15%+).\n"
        f"- **Strong recommendation**: do not open a directional position now "
        f"unless the thesis explicitly bets on earnings. Either size 50% smaller, "
        f"buy after the print, or use defined-risk options instead of equity.\n"
    )
