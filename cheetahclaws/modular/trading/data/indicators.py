"""
Technical indicators for trading analysis.

All functions operate on lists of price dicts (OHLCV format from fetchers)
and return lists of computed values aligned with input length.

No external dependencies — pure Python + math stdlib.
"""
from __future__ import annotations

import math
from typing import Sequence


# ── Moving Averages ────────────────────────────────────────────────────────

def sma(closes: Sequence[float], period: int) -> list[float | None]:
    """Simple Moving Average."""
    result: list[float | None] = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        result[i] = sum(closes[i - period + 1:i + 1]) / period
    return result


def ema(closes: Sequence[float], period: int) -> list[float | None]:
    """Exponential Moving Average."""
    result: list[float | None] = [None] * len(closes)
    if len(closes) < period:
        return result
    k = 2.0 / (period + 1)
    # Seed with SMA
    result[period - 1] = sum(closes[:period]) / period
    for i in range(period, len(closes)):
        result[i] = closes[i] * k + result[i - 1] * (1 - k)
    return result


def wma(closes: Sequence[float], period: int) -> list[float | None]:
    """Weighted Moving Average (linear weights)."""
    result: list[float | None] = [None] * len(closes)
    denom = period * (period + 1) / 2
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        result[i] = sum(w * v for w, v in zip(range(1, period + 1), window)) / denom
    return result


# ── MACD ───────────────────────────────────────────────────────────────────

def macd(
    closes: Sequence[float],
    fast: int = 12, slow: int = 26, signal_period: int = 9,
) -> dict[str, list[float | None]]:
    """MACD (Moving Average Convergence Divergence).

    Returns:
        {"macd": [...], "signal": [...], "histogram": [...]}
    """
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)

    macd_line: list[float | None] = [None] * len(closes)
    for i in range(len(closes)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]

    # Signal line: EMA of MACD values
    macd_values = [v for v in macd_line if v is not None]
    signal_line: list[float | None] = [None] * len(closes)
    if len(macd_values) >= signal_period:
        sig_ema = ema(macd_values, signal_period)
        offset = len(closes) - len(macd_values)
        for i, v in enumerate(sig_ema):
            if v is not None:
                signal_line[offset + i] = v

    histogram: list[float | None] = [None] * len(closes)
    for i in range(len(closes)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram[i] = macd_line[i] - signal_line[i]

    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


# ── RSI ────────────────────────────────────────────────────────────────────

def rsi(closes: Sequence[float], period: int = 14) -> list[float | None]:
    """Relative Strength Index (Wilder's smoothing)."""
    result: list[float | None] = [None] * len(closes)
    if len(closes) < period + 1:
        return result

    # Initial average gain/loss
    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100 - 100 / (1 + rs)

    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gain = max(diff, 0)
        loss = max(-diff, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - 100 / (1 + rs)

    return result


# ── Bollinger Bands ────────────────────────────────────────────────────────

def bollinger_bands(
    closes: Sequence[float], period: int = 20, num_std: float = 2.0
) -> dict[str, list[float | None]]:
    """Bollinger Bands.

    Returns:
        {"upper": [...], "middle": [...], "lower": [...], "bandwidth": [...]}
    """
    middle = sma(closes, period)
    upper: list[float | None] = [None] * len(closes)
    lower: list[float | None] = [None] * len(closes)
    bandwidth: list[float | None] = [None] * len(closes)

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        mean = middle[i]
        std = math.sqrt(sum((x - mean) ** 2 for x in window) / period)
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std
        bandwidth[i] = (upper[i] - lower[i]) / mean if mean else 0

    return {"upper": upper, "middle": middle, "lower": lower, "bandwidth": bandwidth}


# ── ATR (Average True Range) ──────────────────────────────────────────────

def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[float | None]:
    """Average True Range (Wilder's smoothing)."""
    n = len(closes)
    result: list[float | None] = [None] * n
    if n < 2:
        return result

    tr_values = [highs[0] - lows[0]]
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_values.append(tr)

    if n < period:
        return result

    # Initial ATR = simple average of first `period` TRs
    atr_val = sum(tr_values[:period]) / period
    result[period - 1] = atr_val

    for i in range(period, n):
        atr_val = (atr_val * (period - 1) + tr_values[i]) / period
        result[i] = atr_val

    return result


# ── VWAP ───────────────────────────────────────────────────────────────────

def vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
) -> list[float | None]:
    """Volume Weighted Average Price (cumulative)."""
    result: list[float | None] = [None] * len(closes)
    cum_vol = 0.0
    cum_tp_vol = 0.0
    for i in range(len(closes)):
        tp = (highs[i] + lows[i] + closes[i]) / 3
        cum_vol += volumes[i]
        cum_tp_vol += tp * volumes[i]
        result[i] = cum_tp_vol / cum_vol if cum_vol > 0 else None
    return result


# ── OBV (On-Balance Volume) ───────────────────────────────────────────────

def obv(closes: Sequence[float], volumes: Sequence[float]) -> list[float]:
    """On-Balance Volume."""
    result = [0.0] * len(closes)
    if not closes:
        return result
    result[0] = float(volumes[0])
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            result[i] = result[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            result[i] = result[i - 1] - volumes[i]
        else:
            result[i] = result[i - 1]
    return result


# ── ADX (Average Directional Index) ───────────────────────────────────────

def adx(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> dict[str, list[float | None]]:
    """Average Directional Index.

    Returns:
        {"adx": [...], "plus_di": [...], "minus_di": [...]}
    """
    n = len(closes)
    adx_vals: list[float | None] = [None] * n
    plus_di: list[float | None] = [None] * n
    minus_di: list[float | None] = [None] * n

    if n < period + 1:
        return {"adx": adx_vals, "plus_di": plus_di, "minus_di": minus_di}

    # True Range, +DM, -DM
    tr_list = []
    plus_dm = []
    minus_dm = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_list.append(tr)
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0)
        minus_dm.append(down if down > up and down > 0 else 0)

    # Smoothed sums (Wilder's)
    atr_sum = sum(tr_list[:period])
    pdm_sum = sum(plus_dm[:period])
    mdm_sum = sum(minus_dm[:period])

    dx_list = []
    for i in range(period - 1, len(tr_list)):
        if i == period - 1:
            pass  # use initial sums
        else:
            atr_sum = atr_sum - atr_sum / period + tr_list[i]
            pdm_sum = pdm_sum - pdm_sum / period + plus_dm[i]
            mdm_sum = mdm_sum - mdm_sum / period + minus_dm[i]

        pdi = 100 * pdm_sum / atr_sum if atr_sum else 0
        mdi = 100 * mdm_sum / atr_sum if atr_sum else 0
        idx = i + 1  # offset by 1 since tr_list starts at index 1
        plus_di[idx] = round(pdi, 2)
        minus_di[idx] = round(mdi, 2)

        di_sum = pdi + mdi
        dx = 100 * abs(pdi - mdi) / di_sum if di_sum else 0
        dx_list.append(dx)

    # ADX = smoothed DX
    if len(dx_list) >= period:
        adx_val = sum(dx_list[:period]) / period
        adx_vals[2 * period] = round(adx_val, 2)
        for i in range(period, len(dx_list)):
            adx_val = (adx_val * (period - 1) + dx_list[i]) / period
            idx = i + period + 1
            if idx < n:
                adx_vals[idx] = round(adx_val, 2)

    return {"adx": adx_vals, "plus_di": plus_di, "minus_di": minus_di}


# ── Stochastic Oscillator ─────────────────────────────────────────────────

def stochastic(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    k_period: int = 14,
    d_period: int = 3,
) -> dict[str, list[float | None]]:
    """Stochastic Oscillator (%K and %D).

    Returns:
        {"k": [...], "d": [...]}
    """
    n = len(closes)
    k_vals: list[float | None] = [None] * n

    for i in range(k_period - 1, n):
        h_window = highs[i - k_period + 1:i + 1]
        l_window = lows[i - k_period + 1:i + 1]
        highest = max(h_window)
        lowest = min(l_window)
        if highest == lowest:
            k_vals[i] = 50.0
        else:
            k_vals[i] = 100 * (closes[i] - lowest) / (highest - lowest)

    # %D = SMA of %K
    k_numbers = [v for v in k_vals if v is not None]
    d_vals: list[float | None] = [None] * n
    if len(k_numbers) >= d_period:
        d_sma = sma(k_numbers, d_period)
        offset = n - len(k_numbers)
        for i, v in enumerate(d_sma):
            if v is not None and offset + i < n:
                d_vals[offset + i] = v

    return {"k": k_vals, "d": d_vals}


# ── Convenience: compute all indicators at once ───────────────────────────

def compute_all(data: list[dict], periods: dict | None = None) -> dict:
    """Compute all indicators from OHLCV data.

    Args:
        data: list of {"date", "open", "high", "low", "close", "volume"} dicts
        periods: override default periods, e.g. {"sma": [20, 50, 200], "rsi": 14}

    Returns:
        Dict of indicator name → values list.
    """
    if not data:
        return {}

    p = periods or {}
    closes = [d["close"] for d in data]
    highs = [d["high"] for d in data]
    lows = [d["low"] for d in data]
    volumes = [d.get("volume", 0) for d in data]

    result = {}

    # Moving averages
    for period in p.get("sma", [20, 50, 200]):
        result[f"sma_{period}"] = sma(closes, period)
    for period in p.get("ema", [12, 26]):
        result[f"ema_{period}"] = ema(closes, period)

    # Momentum
    result["rsi"] = rsi(closes, p.get("rsi", 14))
    result["macd"] = macd(closes)
    result["stochastic"] = stochastic(highs, lows, closes)

    # Volatility
    result["bollinger"] = bollinger_bands(closes, p.get("bb_period", 20))
    result["atr"] = atr(highs, lows, closes, p.get("atr", 14))

    # Volume
    result["obv"] = obv(closes, volumes)
    result["vwap"] = vwap(highs, lows, closes, volumes)

    # Trend
    result["adx"] = adx(highs, lows, closes, p.get("adx", 14))

    return result


def format_indicators_report(data: list[dict], indicators: dict) -> str:
    """Format indicators into a human-readable report (last values)."""
    if not data:
        return "No data available."

    lines = []
    last_idx = len(data) - 1
    price = data[last_idx]["close"]
    lines.append(f"Current Price: ${price:,.4f}")
    lines.append(f"Date: {data[last_idx]['date']}")
    lines.append("")

    # SMAs
    lines.append("## Moving Averages")
    for key in sorted(k for k in indicators if k.startswith("sma_")):
        val = indicators[key][last_idx]
        period = key.split("_")[1]
        status = "ABOVE" if val and price > val else "BELOW"
        lines.append(f"  SMA({period}): {val:,.4f} — Price {status}" if val else f"  SMA({period}): N/A")
    for key in sorted(k for k in indicators if k.startswith("ema_")):
        val = indicators[key][last_idx]
        period = key.split("_")[1]
        lines.append(f"  EMA({period}): {val:,.4f}" if val else f"  EMA({period}): N/A")

    # RSI
    lines.append("")
    lines.append("## Momentum")
    rsi_val = indicators.get("rsi", [None] * (last_idx + 1))[last_idx]
    if rsi_val is not None:
        zone = "OVERBOUGHT" if rsi_val > 70 else "OVERSOLD" if rsi_val < 30 else "NEUTRAL"
        lines.append(f"  RSI(14): {rsi_val:.2f} — {zone}")

    # MACD
    macd_data = indicators.get("macd", {})
    macd_val = macd_data.get("macd", [None] * (last_idx + 1))[last_idx]
    sig_val = macd_data.get("signal", [None] * (last_idx + 1))[last_idx]
    hist_val = macd_data.get("histogram", [None] * (last_idx + 1))[last_idx]
    if macd_val is not None:
        trend = "BULLISH" if hist_val and hist_val > 0 else "BEARISH"
        lines.append(f"  MACD: {macd_val:.4f} | Signal: {sig_val:.4f} | Histogram: {hist_val:.4f} — {trend}")

    # Stochastic
    stoch = indicators.get("stochastic", {})
    k_val = stoch.get("k", [None] * (last_idx + 1))[last_idx]
    d_val = stoch.get("d", [None] * (last_idx + 1))[last_idx]
    if k_val is not None:
        zone = "OVERBOUGHT" if k_val > 80 else "OVERSOLD" if k_val < 20 else "NEUTRAL"
        lines.append(f"  Stochastic %K: {k_val:.2f} | %D: {d_val:.2f}" if d_val else f"  Stochastic %K: {k_val:.2f}")

    # Bollinger Bands
    lines.append("")
    lines.append("## Volatility")
    bb = indicators.get("bollinger", {})
    bb_upper = bb.get("upper", [None] * (last_idx + 1))[last_idx]
    bb_lower = bb.get("lower", [None] * (last_idx + 1))[last_idx]
    bb_bw = bb.get("bandwidth", [None] * (last_idx + 1))[last_idx]
    if bb_upper is not None:
        pos = "ABOVE UPPER" if price > bb_upper else "BELOW LOWER" if price < bb_lower else "WITHIN BANDS"
        lines.append(f"  Bollinger: Upper={bb_upper:.4f} Lower={bb_lower:.4f} BW={bb_bw:.4f} — {pos}")

    atr_val = indicators.get("atr", [None] * (last_idx + 1))[last_idx]
    if atr_val is not None:
        lines.append(f"  ATR(14): {atr_val:.4f} ({atr_val/price*100:.2f}% of price)")

    # Volume
    lines.append("")
    lines.append("## Volume")
    obv_val = indicators.get("obv", [0] * (last_idx + 1))[last_idx]
    lines.append(f"  OBV: {obv_val:,.0f}")
    vwap_val = indicators.get("vwap", [None] * (last_idx + 1))[last_idx]
    if vwap_val:
        lines.append(f"  VWAP: {vwap_val:,.4f}")

    # ADX
    lines.append("")
    lines.append("## Trend Strength")
    adx_data = indicators.get("adx", {})
    adx_val = adx_data.get("adx", [None] * (last_idx + 1))[last_idx]
    if adx_val is not None:
        strength = "STRONG" if adx_val > 25 else "WEAK"
        lines.append(f"  ADX(14): {adx_val:.2f} — {strength} trend")
        pdi = adx_data.get("plus_di", [None] * (last_idx + 1))[last_idx]
        mdi = adx_data.get("minus_di", [None] * (last_idx + 1))[last_idx]
        if pdi is not None and mdi is not None:
            direction = "BULLISH (+DI > -DI)" if pdi > mdi else "BEARISH (-DI > +DI)"
            lines.append(f"  +DI: {pdi:.2f} | -DI: {mdi:.2f} — {direction}")

    return "\n".join(lines)
