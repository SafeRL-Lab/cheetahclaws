"""
factors.py — classical quant factor scoring.

Per-ticker factors (all in [0, 1] after normalisation, higher = better):
  - momentum_score  : 6m return + 50d>200d trend confirmation
  - quality_score   : ROE + low debt + healthy operating margin
  - low_vol_score   : -log(90d realised stdev), normalised
  - value_score     : earnings/price + healthy ROE (avoids value traps)
  - composite       : weighted blend (default = momentum + quality)

Implementation notes:
  - 24h disk cache at ~/.cheetahclaws/trading/factors_cache.json
  - Fundamentals via yfinance .info; gracefully degrades to NaN
  - Concurrent fetch via ThreadPoolExecutor (workers=4) to keep scan
    time on S&P 100 under ~90s
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


_CACHE_PATH = Path.home() / ".cheetahclaws" / "trading" / "factors_cache.json"
_CACHE_TTL_SEC = 24 * 3600  # 24 hours


@dataclass
class FactorRow:
    symbol: str
    price:           float | None = None
    sma50:           float | None = None
    sma200:          float | None = None
    ret_6m:          float | None = None
    realised_vol_90d: float | None = None
    pe:              float | None = None
    roe:             float | None = None
    debt_to_equity:  float | None = None
    operating_margin: float | None = None
    market_cap:      float | None = None
    sector:          str | None = None
    fetched_at:      float | None = None
    error:           str | None = None

    # Computed scores (filled by score()).
    momentum_score:  float | None = None
    quality_score:   float | None = None
    low_vol_score:   float | None = None
    value_score:     float | None = None
    composite_score: float | None = None


# ── Cache I/O ─────────────────────────────────────────────────────────────

def _load_cache() -> dict[str, dict[str, Any]]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict[str, dict[str, Any]]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, default=str))
    tmp.replace(_CACHE_PATH)


def _cache_fresh(entry: dict[str, Any]) -> bool:
    fetched = entry.get("fetched_at")
    if not fetched:
        return False
    try:
        return (time.time() - float(fetched)) < _CACHE_TTL_SEC
    except (TypeError, ValueError):
        return False


def clear_cache() -> None:
    if _CACHE_PATH.exists():
        _CACHE_PATH.unlink()


# ── Per-ticker fetch ──────────────────────────────────────────────────────

def _fetch_one(symbol: str) -> FactorRow:
    """Fetch raw factor data for one ticker. Returns row with error set on failure."""
    try:
        import yfinance as yf
    except ImportError:
        return FactorRow(symbol=symbol, error="yfinance not installed")

    row = FactorRow(symbol=symbol, fetched_at=time.time())
    try:
        tk = yf.Ticker(symbol)

        # Price history for momentum + vol + SMA
        hist = tk.history(period="1y", interval="1d", auto_adjust=False)
        if hist is None or len(hist) < 30:
            row.error = f"insufficient history ({len(hist) if hist is not None else 0} bars)"
            return row

        closes = hist["Close"].dropna().tolist()
        n = len(closes)
        row.price = float(closes[-1])

        if n >= 50:
            row.sma50 = float(sum(closes[-50:]) / 50.0)
        if n >= 200:
            row.sma200 = float(sum(closes[-200:]) / 200.0)

        # 6-month return (≈126 trading days)
        if n >= 130:
            row.ret_6m = float(closes[-1] / closes[-126] - 1.0)
        elif n >= 30:
            # Fallback to whatever we have
            row.ret_6m = float(closes[-1] / closes[0] - 1.0)

        # 90d realised vol (daily log returns, annualised)
        if n >= 91:
            recent = closes[-91:]
            log_rets = [math.log(recent[i] / recent[i - 1])
                        for i in range(1, len(recent))
                        if recent[i] > 0 and recent[i - 1] > 0]
            if log_rets:
                mean_r = sum(log_rets) / len(log_rets)
                var = sum((r - mean_r) ** 2 for r in log_rets) / len(log_rets)
                row.realised_vol_90d = math.sqrt(var) * math.sqrt(252)

        # Fundamentals via .info — slow + sometimes incomplete.
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        row.pe              = _f(info.get("trailingPE"))
        row.roe             = _f(info.get("returnOnEquity"))
        row.debt_to_equity  = _f(info.get("debtToEquity"))
        row.operating_margin = _f(info.get("operatingMargins"))
        row.market_cap      = _f(info.get("marketCap"))
        row.sector          = info.get("sector") or None
    except Exception as e:
        row.error = f"{type(e).__name__}: {e}"
    return row


def _f(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


# ── Public: scan a universe ──────────────────────────────────────────────

def scan_universe(
    symbols: list[str],
    use_cache: bool = True,
    max_workers: int = 4,
    progress_cb=None,
) -> list[FactorRow]:
    """Fetch factor data for every symbol. Returns list of FactorRow.

    `progress_cb(done, total, symbol)` is called per ticker if provided.
    """
    cache = _load_cache() if use_cache else {}
    rows: list[FactorRow] = []
    todo: list[str] = []

    for sym in symbols:
        entry = cache.get(sym)
        if entry and _cache_fresh(entry):
            row = FactorRow(**{k: v for k, v in entry.items()
                                if k in FactorRow.__dataclass_fields__})
            rows.append(row)
        else:
            todo.append(sym)

    if todo:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_one, sym): sym for sym in todo}
            done = 0
            for fut in as_completed(futures):
                sym = futures[fut]
                done += 1
                try:
                    row = fut.result()
                except Exception as e:
                    row = FactorRow(symbol=sym, error=f"future-error: {e}",
                                    fetched_at=time.time())
                rows.append(row)
                cache[sym] = asdict(row)
                if progress_cb:
                    progress_cb(done, len(todo), sym)

        _save_cache(cache)

    # Sort to preserve input order
    order = {s: i for i, s in enumerate(symbols)}
    rows.sort(key=lambda r: order.get(r.symbol, 99999))
    return rows


# ── Scoring ───────────────────────────────────────────────────────────────

def _safe_normalize(values: list[float | None]) -> list[float | None]:
    """Min-max normalise to [0, 1] across non-None values."""
    nums = [v for v in values if v is not None and not math.isnan(v)]
    if not nums:
        return [None] * len(values)
    lo, hi = min(nums), max(nums)
    if hi - lo < 1e-9:
        return [0.5 if v is not None else None for v in values]
    return [(v - lo) / (hi - lo) if v is not None else None for v in values]


def score(rows: list[FactorRow], weights: dict[str, float] | None = None) -> list[FactorRow]:
    """Compute factor scores in-place.

    Default weights: momentum 0.4 + quality 0.4 + low_vol 0.2 (no value).
    Override by passing weights dict.
    """
    weights = weights or {"momentum": 0.4, "quality": 0.4, "low_vol": 0.2, "value": 0.0}

    # Momentum: 6m return, with bonus if price > 50d > 200d
    mom_raw = []
    for r in rows:
        if r.ret_6m is None:
            mom_raw.append(None)
            continue
        boost = 0.0
        if (r.price is not None and r.sma50 is not None and r.sma200 is not None
            and r.price > r.sma50 > r.sma200):
            boost = 0.05  # 5pp bonus for confirmed uptrend
        mom_raw.append(r.ret_6m + boost)

    # Quality: ROE - 0.3*debt/equity + 2*operating_margin
    # ROE is fractional (0.18 = 18%); D/E from yfinance is in %
    qual_raw = []
    for r in rows:
        if r.roe is None and r.operating_margin is None:
            qual_raw.append(None)
            continue
        roe = r.roe or 0.0
        de  = (r.debt_to_equity or 0.0) / 100.0  # yfinance returns % so /100
        om  = r.operating_margin or 0.0
        qual_raw.append(roe - 0.3 * de + 2.0 * om)

    # Low vol: -log(vol). Negate so lower vol = higher score.
    vol_raw = []
    for r in rows:
        if r.realised_vol_90d is None or r.realised_vol_90d <= 0:
            vol_raw.append(None)
            continue
        vol_raw.append(-math.log(r.realised_vol_90d))

    # Value: earnings yield (1/PE), penalised when ROE is low (value trap)
    val_raw = []
    for r in rows:
        if r.pe is None or r.pe <= 0:
            val_raw.append(None)
            continue
        ey = 1.0 / r.pe
        roe = r.roe or 0.0
        # If ROE < 5%, treat as trap and penalise
        trap_penalty = -0.5 if roe < 0.05 else 0.0
        val_raw.append(ey + 0.1 * roe + trap_penalty)

    mom_n  = _safe_normalize(mom_raw)
    qual_n = _safe_normalize(qual_raw)
    vol_n  = _safe_normalize(vol_raw)
    val_n  = _safe_normalize(val_raw)

    for i, r in enumerate(rows):
        r.momentum_score  = mom_n[i]
        r.quality_score   = qual_n[i]
        r.low_vol_score   = vol_n[i]
        r.value_score     = val_n[i]

        parts = []
        for k, n in (("momentum", mom_n[i]), ("quality", qual_n[i]),
                     ("low_vol", vol_n[i]),  ("value", val_n[i])):
            w = weights.get(k, 0.0)
            if w > 0 and n is not None:
                parts.append(w * n)
        # Re-normalise by sum of weights of available components
        used_weights = sum(weights.get(k, 0.0)
                           for k, n in (("momentum", mom_n[i]),
                                        ("quality", qual_n[i]),
                                        ("low_vol", vol_n[i]),
                                        ("value", val_n[i]))
                           if weights.get(k, 0.0) > 0 and n is not None)
        r.composite_score = (sum(parts) / used_weights) if used_weights > 0 else None

    return rows


def render_factor_table(rows: list[FactorRow], top: int = 25) -> str:
    """Markdown table of top-N tickers by composite score."""
    scored = [r for r in rows if r.composite_score is not None]
    scored.sort(key=lambda r: -r.composite_score)
    lines = [
        f"# Factor Scores (top {min(top, len(scored))} of {len(rows)})",
        "",
        "| # | Symbol | Composite | Momentum | Quality | LowVol | 6M Ret | ROE | D/E |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(scored[:top], 1):
        def fmt(v, pct=False):
            if v is None:
                return "—"
            return f"{v*100:.1f}%" if pct else f"{v:.3f}"
        lines.append(
            f"| {i} | {r.symbol} | {r.composite_score:.3f} | "
            f"{fmt(r.momentum_score)} | {fmt(r.quality_score)} | {fmt(r.low_vol_score)} | "
            f"{fmt(r.ret_6m, pct=True)} | {fmt(r.roe, pct=True)} | "
            f"{r.debt_to_equity or '—':.0f}{'' if r.debt_to_equity is None else '%'} |"
        )
    skipped = sum(1 for r in rows if r.composite_score is None)
    if skipped:
        lines.append("")
        lines.append(f"_Skipped {skipped} ticker(s) with insufficient data._")
    return "\n".join(lines)
