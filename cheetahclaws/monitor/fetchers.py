"""
monitor/fetchers.py — Data fetchers for each subscription topic.

All fetchers return a RawData dict:
  {
    "topic":   str,
    "source":  str,           # human-readable source name
    "items":   list[dict],    # raw items (title, url, summary, date, ...)
    "error":   str | None,    # None on success
  }

No external dependencies beyond stdlib — uses urllib.request + xml.etree.
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any


# ── HTTP helper ────────────────────────────────────────────────────────────

def _get(url: str, headers: dict | None = None, timeout: int = 15) -> str | None:
    req = urllib.request.Request(url, headers={
        "User-Agent": "CheetahClaws-Monitor/1.0",
        **(headers or {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = "utf-8"
            ct = resp.headers.get_content_charset()
            if ct:
                charset = ct
            return resp.read().decode(charset, errors="replace")
    except Exception:
        return None


def _get_json(url: str, headers: dict | None = None) -> Any | None:
    text = _get(url, headers)
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _parse_rss(xml_text: str) -> list[dict]:
    """Parse RSS/Atom feed XML. Returns list of {title, url, summary, date}."""
    items = []
    try:
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom",
              "dc": "http://purl.org/dc/elements/1.1/"}

        # Try RSS 2.0
        for item in root.iter("item"):
            def _t(tag):
                el = item.find(tag)
                return (el.text or "").strip() if el is not None else ""
            items.append({
                "title":   _t("title"),
                "url":     _t("link"),
                "summary": re.sub(r"<[^>]+>", "", _t("description"))[:300],
                "date":    _t("pubDate"),
            })

        # Try Atom
        for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
            def _at(tag):
                el = entry.find(f"atom:{tag}", ns)
                return (el.text or "").strip() if el is not None else ""
            link_el = entry.find("atom:link", ns)
            url = (link_el.get("href") or "") if link_el is not None else ""
            summary_el = entry.find("atom:summary", ns) or entry.find("atom:content", ns)
            summary = re.sub(r"<[^>]+>", "", (summary_el.text or "") if summary_el is not None else "")[:300]
            items.append({
                "title":   _at("title"),
                "url":     url,
                "summary": summary,
                "date":    _at("updated") or _at("published"),
            })
    except Exception:
        pass
    return items[:30]


# ── ai_research ────────────────────────────────────────────────────────────

def _arxiv_api_search(query: str, max_results: int = 15) -> list[dict]:
    """Use arxiv API to search for recent papers."""
    import urllib.parse
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"https://export.arxiv.org/api/query?{params}"
    xml = _get(url)
    if not xml:
        return []
    return _parse_rss(xml)


def fetch_ai_research() -> dict:
    """Fetch latest AI papers from arxiv cs.AI + cs.LG RSS feeds (or API on weekends)."""
    items = []
    errors = []

    for category in ("cs.AI", "cs.LG", "cs.CL"):
        url = f"https://export.arxiv.org/rss/{category}"
        xml = _get(url)
        if xml:
            raw = _parse_rss(xml)
            for r in raw:
                r["category"] = category
            items.extend(raw[:10])
        else:
            errors.append(category)

    # On weekends arxiv RSS is empty — fall back to API search
    if not items:
        for query in ("cat:cs.AI", "cat:cs.LG", "cat:cs.CL"):
            raw = _arxiv_api_search(query, max_results=8)
            cat = query.split(":")[1]
            for r in raw:
                r["category"] = cat
            items.extend(raw)

    return {
        "topic": "ai_research",
        "source": "arxiv (cs.AI, cs.LG, cs.CL)",
        "items": items[:25],
        "error": f"Failed to fetch: {', '.join(errors)}" if errors and not items else None,
    }


# ── stock_TICKER ───────────────────────────────────────────────────────────

def fetch_stock(ticker: str) -> dict:
    """Fetch stock price + recent data from Yahoo Finance (no API key)."""
    ticker = ticker.upper()
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&range=5d"
    )
    data = _get_json(url)
    if not data:
        return {"topic": f"stock_{ticker}", "source": "Yahoo Finance",
                "items": [], "error": f"Failed to fetch {ticker} data"}

    try:
        result = data["chart"]["result"][0]
        meta = result["meta"]
        price = meta.get("regularMarketPrice", 0)
        prev_close = meta.get("chartPreviousClose", meta.get("previousClose", price))
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0
        currency = meta.get("currency", "USD")
        name = meta.get("longName") or meta.get("shortName") or ticker
        volume = meta.get("regularMarketVolume", 0)

        items = [{
            "title": f"{name} ({ticker}): {currency} {price:.2f}  ({change:+.2f}, {change_pct:+.2f}%)",
            "url": f"https://finance.yahoo.com/quote/{ticker}",
            "summary": (
                f"Price: {price:.2f} {currency} | "
                f"Change: {change:+.2f} ({change_pct:+.2f}%) | "
                f"Volume: {volume:,} | "
                f"Exchange: {meta.get('exchangeName', 'N/A')}"
            ),
            "date": datetime.now(timezone.utc).isoformat(),
        }]

        # Add recent closes
        timestamps = result.get("timestamp", [])
        closes = (result.get("indicators", {})
                       .get("quote", [{}])[0]
                       .get("close", []))
        history = []
        for ts, c in zip(timestamps[-5:], closes[-5:]):
            if c:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                history.append(f"{dt}: {c:.2f}")
        if history:
            items.append({
                "title": f"{ticker} 5-day price history",
                "url": "",
                "summary": " | ".join(history),
                "date": "",
            })

        return {"topic": f"stock_{ticker}", "source": "Yahoo Finance",
                "items": items, "error": None}
    except (KeyError, IndexError, TypeError) as e:
        return {"topic": f"stock_{ticker}", "source": "Yahoo Finance",
                "items": [], "error": f"Parse error: {e}"}


# ── crypto_SYMBOL ──────────────────────────────────────────────────────────

_COIN_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "ADA": "cardano",
    "DOGE": "dogecoin", "DOT": "polkadot", "AVAX": "avalanche-2",
    "MATIC": "matic-network", "LINK": "chainlink", "UNI": "uniswap",
    "LTC": "litecoin", "ATOM": "cosmos", "NEAR": "near",
}


def fetch_crypto(symbol: str) -> dict:
    """Fetch crypto price/market data via CoinGecko public API (no key needed)."""
    symbol = symbol.upper()
    coin_id = _COIN_IDS.get(symbol, symbol.lower())

    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        f"?localization=false&tickers=false&community_data=false&developer_data=false"
    )
    data = _get_json(url)
    if not data:
        # fallback: simple price endpoint
        url2 = (
            f"https://api.coingecko.com/api/v3/simple/price"
            f"?ids={coin_id}&vs_currencies=usd"
            f"&include_24hr_change=true&include_market_cap=true&include_24hr_vol=true"
        )
        data2 = _get_json(url2)
        if data2 and coin_id in data2:
            d = data2[coin_id]
            price = d.get("usd", 0)
            change = d.get("usd_24h_change", 0)
            mcap = d.get("usd_market_cap", 0)
            vol = d.get("usd_24h_vol", 0)
            items = [{
                "title": f"{symbol}: ${price:,.2f}  ({change:+.2f}% 24h)",
                "url": f"https://www.coingecko.com/en/coins/{coin_id}",
                "summary": (
                    f"Price: ${price:,.2f} | 24h change: {change:+.2f}% | "
                    f"Market cap: ${mcap:,.0f} | Volume 24h: ${vol:,.0f}"
                ),
                "date": datetime.now(timezone.utc).isoformat(),
            }]
            return {"topic": f"crypto_{symbol}", "source": "CoinGecko",
                    "items": items, "error": None}
        return {"topic": f"crypto_{symbol}", "source": "CoinGecko",
                "items": [], "error": f"Failed to fetch {symbol} data"}

    try:
        name = data.get("name", symbol)
        md = data.get("market_data", {})
        price = md.get("current_price", {}).get("usd", 0)
        change_24h = md.get("price_change_percentage_24h", 0) or 0
        change_7d = md.get("price_change_percentage_7d", 0) or 0
        change_30d = md.get("price_change_percentage_30d", 0) or 0
        mcap = md.get("market_cap", {}).get("usd", 0)
        vol = md.get("total_volume", {}).get("usd", 0)
        ath = md.get("ath", {}).get("usd", 0)
        ath_change = md.get("ath_change_percentage", {}).get("usd", 0) or 0
        rank = data.get("market_cap_rank", "N/A")
        desc_raw = data.get("description", {}).get("en", "")
        desc = re.sub(r"<[^>]+>", "", desc_raw)[:200]

        items = [
            {
                "title": f"{name} ({symbol}): ${price:,.2f}  ({change_24h:+.2f}% 24h)",
                "url": f"https://www.coingecko.com/en/coins/{coin_id}",
                "summary": (
                    f"Price: ${price:,.2f} | 24h: {change_24h:+.2f}% | "
                    f"7d: {change_7d:+.2f}% | 30d: {change_30d:+.2f}% | "
                    f"Market cap: ${mcap:,.0f} | Volume: ${vol:,.0f} | "
                    f"Rank: #{rank} | ATH: ${ath:,.2f} ({ath_change:+.2f}%)"
                ),
                "date": datetime.now(timezone.utc).isoformat(),
            }
        ]
        if desc:
            items.append({"title": "About", "url": "", "summary": desc, "date": ""})

        return {"topic": f"crypto_{symbol}", "source": "CoinGecko",
                "items": items, "error": None}
    except Exception as e:
        return {"topic": f"crypto_{symbol}", "source": "CoinGecko",
                "items": [], "error": f"Parse error: {e}"}


# ── world_news ─────────────────────────────────────────────────────────────

_NEWS_FEEDS = [
    ("Reuters World", "https://feeds.reuters.com/reuters/worldNews"),
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("The Guardian World", "https://www.theguardian.com/world/rss"),
    ("AP News", "https://feeds.apnews.com/rss/topnews"),
]


def fetch_world_news() -> dict:
    """Fetch top world news from multiple RSS feeds."""
    items = []
    errors = []

    for name, url in _NEWS_FEEDS:
        xml = _get(url)
        if xml:
            raw = _parse_rss(xml)
            for r in raw[:5]:
                r["source_name"] = name
            items.extend(raw[:5])
        else:
            errors.append(name)
        if len(items) >= 20:
            break

    return {
        "topic": "world_news",
        "source": "Reuters / BBC / Guardian / AP",
        "items": items[:20],
        "error": f"Failed: {', '.join(errors)}" if errors and not items else None,
    }


# ── custom:QUERY ───────────────────────────────────────────────────────────

def fetch_custom(query: str) -> dict:
    """Fetch results for a custom query via DuckDuckGo Instant Answer API."""
    import urllib.parse
    encoded = urllib.parse.quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_redirect=1&no_html=1"
    data = _get_json(url)

    items = []
    if data:
        abstract = data.get("AbstractText", "")
        if abstract:
            items.append({
                "title": data.get("Heading", query),
                "url": data.get("AbstractURL", ""),
                "summary": abstract[:400],
                "date": datetime.now(timezone.utc).isoformat(),
            })
        for rt in data.get("RelatedTopics", [])[:8]:
            if isinstance(rt, dict) and rt.get("Text"):
                items.append({
                    "title": rt.get("Text", "")[:80],
                    "url": rt.get("FirstURL", ""),
                    "summary": rt.get("Text", "")[:300],
                    "date": "",
                })

    return {
        "topic": f"custom:{query}",
        "source": f"DuckDuckGo search: {query}",
        "items": items,
        "error": None if items else "No results found",
    }


# ── research:QUERY — multi-source research brief ───────────────────────────

def fetch_research(query: str, time_range_token: str = "7d") -> dict:
    """Run the full /research pipeline for this query, return formatted brief.

    Used by `research:<topic>` subscriptions for weekly trend tracking.
    Default window is last 7 days (a `/monitor` subscription running
    weekly covers the week's new material).
    """
    try:
        from cheetahclaws.research import research, build_time_range
        from cheetahclaws.research.synthesizer import (
            format_heat_table, format_publication_trend,
            format_publication_sparkline, render_citations,
        )
    except ImportError as e:
        return {"topic": f"research:{query}", "source": "research",
                "items": [], "error": f"research module unavailable: {e}"}

    try:
        tr = build_time_range(range_token=time_range_token)
    except ValueError:
        tr = build_time_range(range_token="7d")

    try:
        brief = research(
            topic=query, time_range=tr,
            synthesize=False,       # monitor summarizer does its own pass
            use_cache=True,
            limit=8,                # tighter per-source for digest use
        )
    except Exception as e:
        return {"topic": f"research:{query}", "source": "research",
                "items": [],
                "error": f"research run failed: {type(e).__name__}: {e}"}

    # Flatten to monitor's item shape: {title, url, summary, date}
    items = []
    for r in brief.results[:25]:
        items.append({
            "title": f"[{r.source}] {r.title}"[:200],
            "url": r.url,
            "summary": (
                (f"{r.engagement_label} · " if r.engagement_label else "")
                + (r.snippet or "")
            )[:400],
            "date": r.published,
        })

    # Attach heat table + sparkline as an extra digest item for the summarizer
    spark = format_publication_sparkline(brief, buckets=12)
    heat = format_heat_table(brief)
    if items and heat:
        items.insert(0, {
            "title": f"Cross-platform heat ({tr.label})",
            "url": "",
            "summary": (spark + "\n\n" + heat)[:1200],
            "date": "",
        })

    return {
        "topic": f"research:{query}",
        "source": f"research pipeline ({tr.label})",
        "items": items,
        "error": None if items else "No results found across 17 sources",
    }


# ── Dispatch ───────────────────────────────────────────────────────────────

def fetch(topic: str) -> dict:
    """Fetch raw data for a given topic string."""
    if topic == "ai_research":
        return fetch_ai_research()
    if topic.startswith("stock_"):
        return fetch_stock(topic[6:])
    if topic.startswith("crypto_"):
        return fetch_crypto(topic[7:])
    if topic == "world_news":
        return fetch_world_news()
    if topic.startswith("custom:"):
        return fetch_custom(topic[7:])
    if topic.startswith("research:"):
        # research:<topic>  OR  research:<range>:<topic>  (e.g. research:30d:LLM)
        body = topic[9:]
        maybe_range, _, rest = body.partition(":")
        from cheetahclaws.research.time_range import _PRESET_DAYS
        if maybe_range in _PRESET_DAYS and rest:
            data = fetch_research(rest, time_range_token=maybe_range)
        else:
            data = fetch_research(body, time_range_token="7d")
        # Preserve the original subscription topic string so the
        # scheduler/notifier routes reports back to the right sub.
        data["topic"] = topic
        return data
    return {"topic": topic, "source": "unknown", "items": [],
            "error": f"Unknown topic type: {topic}"}
