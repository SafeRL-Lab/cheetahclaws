"""
Market data fetchers with automatic fallback chains.

Supported sources:
  - yfinance  (US/HK equities, no API key needed)
  - coingecko (crypto, no API key needed)
  - akshare   (A-shares, US, HK, futures, forex — optional)

Each fetcher returns a standardised dict:
    {
        "symbol": str,
        "source": str,
        "data": list[dict],   # OHLCV rows
        "info": dict,         # current price, name, etc.
        "error": str | None,
    }

Fallback chains by market:
    us_equity:  [yfinance]
    hk_equity:  [yfinance]
    crypto:     [coingecko, yfinance]
    a_share:    [akshare, yfinance]
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Any


# ── Market detection ───────────────────────────────────────────────────────

def detect_market(symbol: str) -> str:
    """Detect market type from symbol format."""
    s = symbol.upper().strip()
    # Crypto: BTC, ETH, BTC-USDT, BTCUSDT
    if s in _CRYPTO_IDS or re.match(r"^[A-Z]{2,10}-USDT$", s) or re.match(r"^[A-Z]{2,10}USDT$", s):
        return "crypto"
    # A-share: 6-digit codes (000001.SZ, 600519.SH)
    if re.match(r"^\d{6}\.(SZ|SH|SS)$", s):
        return "a_share"
    # HK: 4-5 digit codes (0700.HK, 9988.HK)
    if re.match(r"^\d{4,5}\.HK$", s):
        return "hk_equity"
    # Default: US equity
    return "us_equity"


# ── Fallback chains ───────────────────────────────────────────────────────

FALLBACK_CHAINS: dict[str, list[str]] = {
    "us_equity": ["yfinance"],
    "hk_equity": ["yfinance"],
    "crypto":    ["coingecko", "yfinance"],
    "a_share":   ["akshare", "yfinance"],
}

_FETCHERS: dict[str, Any] = {}  # populated lazily


def _get_fetcher(source: str):
    """Return fetcher function for source name."""
    return {
        "yfinance":  fetch_yfinance,
        "coingecko": fetch_coingecko,
        "akshare":   fetch_akshare,
    }.get(source)


# ── Public API ─────────────────────────────────────────────────────────────

def fetch_market_data(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "1d",
    source: str = "auto",
) -> dict:
    """Fetch OHLCV data with automatic fallback.

    Args:
        symbol: ticker (AAPL, BTC, 000001.SZ, 0700.HK, etc.)
        start_date: YYYY-MM-DD (default: 1 year ago)
        end_date: YYYY-MM-DD (default: today)
        interval: 1d, 1h, 5m, etc.
        source: auto | yfinance | coingecko | akshare

    Returns:
        Standardised result dict.
    """
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    market = detect_market(symbol)

    if source != "auto":
        fetcher = _get_fetcher(source)
        if fetcher:
            return fetcher(symbol, start_date, end_date, interval)
        return {"symbol": symbol, "source": source, "data": [], "info": {},
                "error": f"Unknown source: {source}"}

    chain = FALLBACK_CHAINS.get(market, ["yfinance"])
    last_error = None
    for src in chain:
        fetcher = _get_fetcher(src)
        if not fetcher:
            continue
        try:
            result = fetcher(symbol, start_date, end_date, interval)
            if result.get("data") or result.get("info"):
                return result
            last_error = result.get("error", "No data returned")
        except Exception as e:
            last_error = str(e)

    return {"symbol": symbol, "source": "none", "data": [], "info": {},
            "error": f"All sources failed for {symbol}: {last_error}"}


def fetch_current_price(symbol: str) -> dict:
    """Fetch current price and basic info (lightweight)."""
    market = detect_market(symbol)
    if market == "crypto":
        return _fetch_crypto_price(symbol)
    return _fetch_stock_price(symbol)


# ── yfinance fetcher ───────────────────────────────────────────────────────

def fetch_yfinance(
    symbol: str, start_date: str, end_date: str, interval: str = "1d"
) -> dict:
    """Fetch OHLCV from Yahoo Finance via yfinance library."""
    try:
        import yfinance as yf
    except ImportError:
        return {"symbol": symbol, "source": "yfinance", "data": [], "info": {},
                "error": "yfinance not installed. Run: pip install yfinance"}

    ticker = _normalize_yf_symbol(symbol)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start_date, end=end_date, interval=interval)
        if hist.empty:
            return {"symbol": symbol, "source": "yfinance", "data": [], "info": {},
                    "error": f"No data for {ticker}"}

        rows = []
        for idx, row in hist.iterrows():
            rows.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(float(row.get("Open", 0)), 4),
                "high": round(float(row.get("High", 0)), 4),
                "low": round(float(row.get("Low", 0)), 4),
                "close": round(float(row.get("Close", 0)), 4),
                "volume": int(row.get("Volume", 0)),
            })

        info_data = {}
        try:
            fi = t.fast_info
            info_data = {
                "name": getattr(fi, "short_name", ticker),
                "price": round(float(getattr(fi, "last_price", rows[-1]["close"])), 4),
                "market_cap": getattr(fi, "market_cap", None),
                "currency": getattr(fi, "currency", "USD"),
            }
        except Exception:
            if rows:
                info_data = {"name": ticker, "price": rows[-1]["close"]}

        return {"symbol": symbol, "source": "yfinance", "data": rows,
                "info": info_data, "error": None}
    except Exception as e:
        return {"symbol": symbol, "source": "yfinance", "data": [], "info": {},
                "error": f"yfinance error: {e}"}


def _normalize_yf_symbol(symbol: str) -> str:
    """Convert symbol to yfinance format."""
    s = symbol.upper().strip()
    # Already valid yfinance ticker
    if re.match(r"^[A-Z]{1,5}$", s):
        return s
    # HK: 0700.HK -> 0700.HK (already correct)
    if s.endswith(".HK"):
        code = s.replace(".HK", "")
        return code.zfill(4) + ".HK"
    # A-share: 000001.SZ -> 000001.SZ (yfinance supports this)
    if re.match(r"^\d{6}\.(SZ|SH|SS)$", s):
        return s.replace(".SS", ".SH")
    # Remove .US suffix
    if s.endswith(".US"):
        return s.replace(".US", "")
    return s


# ── CoinGecko fetcher ──────────────────────────────────────────────────────

_CRYPTO_IDS: dict[str, str] = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "ADA": "cardano",
    "DOGE": "dogecoin", "DOT": "polkadot", "AVAX": "avalanche-2",
    "MATIC": "matic-network", "LINK": "chainlink", "UNI": "uniswap",
    "LTC": "litecoin", "ATOM": "cosmos", "NEAR": "near",
    "ARB": "arbitrum", "OP": "optimism", "APT": "aptos",
    "SUI": "sui", "SEI": "sei-network",
}


def _normalize_crypto_symbol(symbol: str) -> str:
    """Extract base symbol: BTC-USDT -> BTC, BTCUSDT -> BTC."""
    s = symbol.upper().strip()
    s = re.sub(r"[-/]?USDT$", "", s)
    s = re.sub(r"[-/]?USD$", "", s)
    return s


def fetch_coingecko(
    symbol: str, start_date: str, end_date: str, interval: str = "1d"
) -> dict:
    """Fetch crypto price history from CoinGecko (free, no key)."""
    base = _normalize_crypto_symbol(symbol)
    coin_id = _CRYPTO_IDS.get(base)
    if not coin_id:
        return {"symbol": symbol, "source": "coingecko", "data": [], "info": {},
                "error": f"Unknown crypto: {base}. Supported: {', '.join(sorted(_CRYPTO_IDS))}"}

    try:
        # Calculate days for history
        d_start = datetime.strptime(start_date, "%Y-%m-%d")
        d_end = datetime.strptime(end_date, "%Y-%m-%d")
        days = max(1, (d_end - d_start).days)
        if days > 365:
            days = 365  # CoinGecko free tier limit

        url = (
            f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            f"/market_chart?vs_currency=usd&days={days}"
        )
        data = _http_get_json(url)
        if not data or "prices" not in data:
            return {"symbol": symbol, "source": "coingecko", "data": [], "info": {},
                    "error": "CoinGecko returned no price data"}

        rows = []
        prices = data["prices"]
        for i, (ts, price) in enumerate(prices):
            dt = datetime.utcfromtimestamp(ts / 1000)
            rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "open": round(price, 4),
                "high": round(price, 4),
                "low": round(price, 4),
                "close": round(price, 4),
                "volume": 0,
            })

        # Current info
        info_url = (
            f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            f"?localization=false&tickers=false&community_data=false&developer_data=false"
        )
        info_data = {}
        try:
            coin_info = _http_get_json(info_url)
            md = coin_info.get("market_data", {})
            info_data = {
                "name": coin_info.get("name", base),
                "price": md.get("current_price", {}).get("usd", 0),
                "market_cap": md.get("market_cap", {}).get("usd", 0),
                "price_change_24h": md.get("price_change_percentage_24h", 0),
                "price_change_7d": md.get("price_change_percentage_7d", 0),
                "price_change_30d": md.get("price_change_percentage_30d", 0),
                "volume_24h": md.get("total_volume", {}).get("usd", 0),
                "ath": md.get("ath", {}).get("usd", 0),
                "currency": "USD",
            }
        except Exception:
            if rows:
                info_data = {"name": base, "price": rows[-1]["close"]}

        return {"symbol": symbol, "source": "coingecko", "data": rows,
                "info": info_data, "error": None}
    except Exception as e:
        return {"symbol": symbol, "source": "coingecko", "data": [], "info": {},
                "error": f"CoinGecko error: {e}"}


# ── AKShare fetcher (optional) ─────────────────────────────────────────────

def fetch_akshare(
    symbol: str, start_date: str, end_date: str, interval: str = "1d"
) -> dict:
    """Fetch data from AKShare (A-shares, US, HK, futures, forex)."""
    try:
        import akshare as ak  # type: ignore
    except ImportError:
        return {"symbol": symbol, "source": "akshare", "data": [], "info": {},
                "error": "akshare not installed. Run: pip install akshare"}

    s = symbol.upper().strip()
    try:
        # A-share
        if re.match(r"^\d{6}\.(SZ|SH|SS)$", s):
            code = s.split(".")[0]
            df = ak.stock_zh_a_hist(
                symbol=code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
            rows = []
            for _, row in df.iterrows():
                rows.append({
                    "date": str(row.get("日期", "")),
                    "open": float(row.get("开盘", 0)),
                    "high": float(row.get("最高", 0)),
                    "low": float(row.get("最低", 0)),
                    "close": float(row.get("收盘", 0)),
                    "volume": int(row.get("成交量", 0)),
                })
            return {"symbol": symbol, "source": "akshare", "data": rows,
                    "info": {"name": code, "price": rows[-1]["close"] if rows else 0},
                    "error": None}
        else:
            return {"symbol": symbol, "source": "akshare", "data": [], "info": {},
                    "error": f"AKShare: unsupported symbol format: {s}"}
    except Exception as e:
        return {"symbol": symbol, "source": "akshare", "data": [], "info": {},
                "error": f"AKShare error: {e}"}


# ── Stock price (Yahoo Finance API, no library needed) ─────────────────────

def _fetch_stock_price(symbol: str) -> dict:
    """Fetch current stock price via Yahoo Finance API (no library needed)."""
    ticker = _normalize_yf_symbol(symbol)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=5d"
    try:
        data = _http_get_json(url)
        result = data.get("chart", {}).get("result", [{}])[0]
        meta = result.get("meta", {})
        price = meta.get("regularMarketPrice", 0)
        prev = meta.get("chartPreviousClose", price)
        change_pct = ((price - prev) / prev * 100) if prev else 0

        quotes = result.get("indicators", {}).get("quote", [{}])[0]
        volumes = quotes.get("volume", [0])
        volume = volumes[-1] if volumes else 0

        return {
            "symbol": symbol,
            "price": round(price, 4),
            "change_pct": round(change_pct, 2),
            "volume": volume,
            "exchange": meta.get("exchangeName", ""),
            "currency": meta.get("currency", "USD"),
            "name": meta.get("shortName", ticker),
            "error": None,
        }
    except Exception as e:
        return {"symbol": symbol, "price": 0, "error": str(e)}


def _fetch_crypto_price(symbol: str) -> dict:
    """Fetch current crypto price from CoinGecko."""
    base = _normalize_crypto_symbol(symbol)
    coin_id = _CRYPTO_IDS.get(base)
    if not coin_id:
        return {"symbol": symbol, "price": 0, "error": f"Unknown crypto: {base}"}

    url = (
        f"https://api.coingecko.com/api/v3/simple/price"
        f"?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
        f"&include_market_cap=true&include_24hr_vol=true"
    )
    try:
        data = _http_get_json(url)
        info = data.get(coin_id, {})
        return {
            "symbol": symbol,
            "price": info.get("usd", 0),
            "change_pct": round(info.get("usd_24h_change", 0), 2),
            "market_cap": info.get("usd_market_cap", 0),
            "volume_24h": info.get("usd_24h_vol", 0),
            "name": base,
            "error": None,
        }
    except Exception as e:
        return {"symbol": symbol, "price": 0, "error": str(e)}


# ── Fundamentals ───────────────────────────────────────────────────────────

def fetch_fundamentals(symbol: str) -> dict:
    """Fetch fundamental data (P/E, EPS, revenue, etc.) via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return {"symbol": symbol, "error": "yfinance not installed"}

    ticker = _normalize_yf_symbol(symbol)
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return {
            "symbol": symbol,
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "eps": info.get("trailingEps", None),
            "revenue": info.get("totalRevenue", 0),
            "profit_margin": info.get("profitMargins", None),
            "roe": info.get("returnOnEquity", None),
            "debt_to_equity": info.get("debtToEquity", None),
            "dividend_yield": info.get("dividendYield", None),
            "beta": info.get("beta", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "avg_volume": info.get("averageVolume", 0),
            "error": None,
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


# ── News ───────────────────────────────────────────────────────────────────

def fetch_news(symbol: str, limit: int = 10) -> dict:
    """Fetch recent news for a symbol via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return {"symbol": symbol, "news": [], "error": "yfinance not installed"}

    ticker = _normalize_yf_symbol(symbol)
    try:
        t = yf.Ticker(ticker)
        news_items = []
        for item in (t.news or [])[:limit]:
            news_items.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "published": item.get("providerPublishTime", ""),
                "type": item.get("type", ""),
            })
        return {"symbol": symbol, "news": news_items, "error": None}
    except Exception as e:
        return {"symbol": symbol, "news": [], "error": str(e)}


# ── HTTP helpers ───────────────────────────────────────────────────────────

def _http_get_json(url: str, timeout: int = 15) -> dict:
    """Simple HTTP GET returning parsed JSON."""
    req = urllib.request.Request(url, headers={"User-Agent": "CheetahClaws/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())
