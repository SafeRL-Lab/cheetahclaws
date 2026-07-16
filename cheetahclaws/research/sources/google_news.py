"""Google News — via the public RSS feed. No key, multi-language."""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from ..http import DEFAULT_TIMEOUT, DEFAULT_UA
from ..types import Result
from . import SourceSpec, register

_ENDPOINT = "https://news.google.com/rss/search"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    # hl/gl/ceid: language/region. Default en-US; overridable via config.
    hl = (config or {}).get("google_news_hl", "en-US")
    gl = (config or {}).get("google_news_gl", "US")
    ceid = (config or {}).get("google_news_ceid", "US:en")
    # Google News supports `when:<period>` (e.g. 1d, 7d, 30d, 1y) and
    # `after:YYYY-MM-DD before:YYYY-MM-DD` operators in the query itself.
    q = query
    if time_range and time_range.is_bounded:
        if time_range.since:
            q += f" after:{time_range.since.strftime('%Y-%m-%d')}"
        if time_range.until:
            q += f" before:{time_range.until.strftime('%Y-%m-%d')}"
    params = {"q": q, "hl": hl, "gl": gl, "ceid": ceid}
    url = f"{_ENDPOINT}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
        body = resp.read().decode("utf-8", errors="replace")

    # Parse RSS 2.0
    root = ET.fromstring(body)
    channel = root.find("channel")
    if channel is None:
        return []

    out: list[Result] = []
    for item in channel.findall("item")[:limit]:
        title = _text(item, "title")
        link = _text(item, "link")
        pub = _text(item, "pubDate")
        desc = _text(item, "description")
        source_el = item.find("source")
        source_name = source_el.text if source_el is not None and source_el.text else ""

        # Google News description is wrapped HTML; strip tags
        desc_plain = re.sub(r"<[^>]+>", " ", desc)
        desc_plain = re.sub(r"\s+", " ", desc_plain).strip()[:400]

        if not title or not link:
            continue
        out.append(Result(
            source="google_news",
            title=title,
            url=link,
            snippet=desc_plain,
            author=source_name,
            published=pub,
            domain="news",
            engagement_label="news" + (f" · {source_name}" if source_name else ""),
        ))
    return out


def _text(node, path: str) -> str:
    el = node.find(path)
    return (el.text or "").strip() if el is not None and el.text else ""


register(SourceSpec(
    name="google_news",
    domains=["news", "web"],
    tier="free",
    search=search,
    description="Google News RSS — multilingual news aggregation",
))
