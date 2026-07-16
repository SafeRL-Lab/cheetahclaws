"""
macro.py — pull and format macro context for the trading prompt.

A single stock's 60-70% return variance is explained by market beta. The
old /trading analyze pipeline ignored this entirely — it told the model
to BUY on technical strength even when SPY was breaking 200d. This
module fetches SPY/QQQ/VIX/^TNX and produces a 5-7 line "regime" block
that gets prepended to the analyst prompt.

Cached per-process via lru_cache(maxsize=1) with a 30-min TTL — most
sessions analyse multiple stocks back-to-back; we don't need to re-pull
SPY for every one.
"""
from __future__ import annotations

import time
from typing import Any

from .data import fetchers, indicators


_CACHE: dict[str, Any] = {"data": None, "fetched_at": 0.0}
_TTL_SEC = 30 * 60  # 30 minutes


def _fetch_macro_block(force: bool = False) -> dict[str, Any]:
    """Fetch SPY/QQQ/VIX/TNX. Cached for _TTL_SEC."""
    now = time.time()
    if not force and _CACHE["data"] and (now - _CACHE["fetched_at"]) < _TTL_SEC:
        return _CACHE["data"]

    out: dict[str, Any] = {}
    for sym, label in [("SPY", "spy"), ("QQQ", "qqq"), ("^VIX", "vix"), ("^TNX", "tnx")]:
        result = fetchers.fetch_market_data(sym, interval="1d")
        if result.get("error"):
            out[label] = {"error": result["error"]}
            continue
        rows = result.get("data", [])
        if len(rows) < 200:
            out[label] = {"error": f"insufficient history ({len(rows)} bars)"}
            continue

        closes = [r["close"] for r in rows]
        latest = closes[-1]
        sma50 = indicators.sma(closes, 50)
        sma200 = indicators.sma(closes, 200)
        sma50_last = sma50[-1] if sma50 and sma50[-1] is not None else latest
        sma200_last = sma200[-1] if sma200 and sma200[-1] is not None else latest

        # 30-day percentile of latest value (for VIX regime detection)
        recent = closes[-30:]
        rank = sum(1 for x in recent if x <= latest)
        pct = rank / len(recent) * 100.0

        out[label] = {
            "symbol": sym,
            "price": round(latest, 2),
            "sma50": round(sma50_last, 2),
            "sma200": round(sma200_last, 2),
            "above_sma50": latest > sma50_last,
            "above_sma200": latest > sma200_last,
            "pct_30d": round(pct, 1),  # where price sits in last-30-day range
        }

    _CACHE["data"] = out
    _CACHE["fetched_at"] = now
    return out


def render_macro_context() -> str:
    """Return a markdown block summarising the macro regime."""
    data = _fetch_macro_block()

    if all(v.get("error") for v in data.values()):
        return "## Macro Context\n_Macro data unavailable (network or yfinance issue)._"

    lines = ["## Macro Context (US market regime)"]

    spy = data.get("spy", {})
    if not spy.get("error"):
        regime = "RISK-ON" if spy["above_sma200"] and spy["above_sma50"] else \
                 "RISK-OFF" if not spy["above_sma200"] else "TRANSITION"
        lines.append(
            f"- **SPY** {spy['price']} — {'above' if spy['above_sma200'] else 'below'} 200d "
            f"({spy['sma200']}), {'above' if spy['above_sma50'] else 'below'} 50d "
            f"({spy['sma50']}) → **{regime}**"
        )
    qqq = data.get("qqq", {})
    if not qqq.get("error"):
        rel = "leading" if qqq["above_sma50"] and qqq["above_sma200"] else \
              "lagging" if not qqq["above_sma50"] else "mixed"
        lines.append(
            f"- **QQQ** {qqq['price']} — tech sector {rel} (50d/200d "
            f"{'>'  if qqq['above_sma50'] else '<'}/"
            f"{'>'  if qqq['above_sma200'] else '<'})"
        )
    vix = data.get("vix", {})
    if not vix.get("error"):
        # VIX < 15 = complacent, 15-20 = normal, 20-30 = stressed, >30 = panic.
        v = vix["price"]
        regime = ("COMPLACENT" if v < 15 else
                  "NORMAL" if v < 20 else
                  "STRESSED" if v < 30 else "PANIC")
        lines.append(f"- **VIX** {v} — volatility **{regime}** ({vix['pct_30d']:.0f}-percentile of last 30 days)")
    tnx = data.get("tnx", {})
    if not tnx.get("error"):
        # TNX is 10y yield × 10 (convention)
        yld = tnx["price"] / 10.0
        lines.append(f"- **10y Treasury yield** {yld:.2f}% — discount-rate headwind for long-duration assets")

    if len(lines) == 1:
        return ""

    lines.append("")
    lines.append("**How to use this**: do not BUY a single stock when SPY is "
                 "RISK-OFF and VIX is STRESSED unless the thesis is a defensive "
                 "name. Wait for confirmation.")
    return "\n".join(lines)


def clear_cache() -> None:
    """Force the next render_macro_context() to refetch."""
    _CACHE["data"] = None
    _CACHE["fetched_at"] = 0.0
