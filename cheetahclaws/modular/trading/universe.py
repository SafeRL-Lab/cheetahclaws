"""
universe.py — curated symbol universes + sector ETF helpers.

S&P 100 (OEX) is hardcoded since constituents change ~5%/year and we
prefer reproducibility over freshness. For a full S&P 500 scan, the
caller can pass `--universe sp500` and we'll lazily fetch from yfinance
(slower; subject to rate limits).
"""
from __future__ import annotations

from typing import Iterable


# S&P 100 (OEX) constituents as of 2026-Q2. ~7-8% drift per year — refresh
# quarterly or when a major index event hits (split / acquisition).
SP100: list[str] = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX",
    "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM",
    "INTC", "INTU", "ISRG", "JNJ", "JPM", "KHC", "KMI", "KO", "LIN", "LLY",
    "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO",
    "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE",
    "PG", "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SLB", "SO", "SPG",
    "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS", "USB",
    "V", "VZ", "WBA", "WFC", "WMT", "XOM",
]

# Sector ETFs (SPDR Select). Used by sector_rotation discovery to pick
# the leading sector(s) and then surface their top holdings.
SECTOR_ETFS: dict[str, str] = {
    "Technology":             "XLK",
    "Financials":             "XLF",
    "Healthcare":             "XLV",
    "ConsumerDiscretionary":  "XLY",
    "ConsumerStaples":        "XLP",
    "Energy":                 "XLE",
    "Industrials":            "XLI",
    "Materials":              "XLB",
    "Utilities":              "XLU",
    "RealEstate":             "XLRE",
    "Communication":          "XLC",
}

# Reverse map for "given a sector ETF, what sector?"
ETF_TO_SECTOR: dict[str, str] = {v: k for k, v in SECTOR_ETFS.items()}

# Top-10 holdings of each sector ETF as of 2026-Q2 (curated snapshot —
# yfinance .funds_data / .info doesn't reliably expose these for all
# tickers, so we hardcode for predictability).
SECTOR_TOP_HOLDINGS: dict[str, list[str]] = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "ACN", "CSCO"],
    "XLF": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "C"],
    "XLV": ["LLY", "JNJ", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "AMGN", "ISRG"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "BKNG", "TJX", "LOW", "NKE", "SBUX", "CMG"],
    "XLP": ["COST", "WMT", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "TGT"],
    "XLE": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "WMB"],
    "XLI": ["GE", "CAT", "RTX", "HON", "UNP", "BA", "LMT", "DE", "ETN", "UPS"],
    "XLB": ["LIN", "SHW", "APD", "ECL", "FCX", "NEM", "DOW", "PPG", "DD", "MLM"],
    "XLU": ["NEE", "SO", "DUK", "CEG", "AEP", "SRE", "EXC", "D", "XEL", "PCG"],
    "XLRE": ["PLD", "AMT", "EQIX", "WELL", "SPG", "DLR", "PSA", "O", "CCI", "EXR"],
    "XLC": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "TMUS", "VZ", "T", "EA", "CHTR"],
}


# Universe presets — name → list of symbols
PRESETS: dict[str, list[str]] = {
    "sp100":   SP100,
    "sectors": list(SECTOR_ETFS.values()),
}


def resolve_universe(name: str | None,
                     custom: Iterable[str] | None = None) -> list[str]:
    """Resolve a universe name to a list of symbols.

    Order of precedence:
      1. custom list (if provided)
      2. preset name (sp100 / sectors)
      3. default = SP100
    """
    if custom:
        return [s.upper().strip() for s in custom if s.strip()]
    if name and name.lower() in PRESETS:
        return list(PRESETS[name.lower()])
    return list(SP100)


def fetch_sp500_dynamic() -> list[str]:
    """Lazy fetch of S&P 500 constituents via Wikipedia (best-effort).

    Returns SP100 fallback on any failure — Wikipedia is rate-limited and
    occasionally blocks user agents. Cache the result manually if you
    need it more than once per session.
    """
    try:
        import urllib.request
        import re
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(
            url, headers={"User-Agent": "cheetahclaws-trading/3.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode("utf-8", errors="ignore")
        # The first table column has tickers in <a> tags inside <td>.
        # Match `<td><a href="/wiki/...">SYM</a>` rough pattern.
        symbols = re.findall(r'<td>\s*<a [^>]*>([A-Z]{1,5}(?:\.[A-Z])?)</a>\s*</td>', html)
        # First column of S&P 500 table — dedupe + uppercase
        out = list(dict.fromkeys(s.upper() for s in symbols))
        if len(out) >= 400:  # sanity: real S&P 500 is ~503
            return out
        return SP100
    except Exception:
        return SP100
