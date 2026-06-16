"""arXiv — academic preprints via the public Atom feed API.

No API key. Rate limit: recommended 1 req / 3s; we stay well under by
issuing a single request per search.
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from ..http import DEFAULT_TIMEOUT, DEFAULT_UA
from ..types import Result
from . import SourceSpec, register

_ENDPOINT = "http://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    # arXiv supports submittedDate:[YYYYMMDDHHMM+TO+YYYYMMDDHHMM] in the query
    search_q = f"all:{query}"
    if time_range and time_range.is_bounded:
        since = (time_range.since.strftime("%Y%m%d0000") if time_range.since else "197001010000")
        until = (time_range.until.strftime("%Y%m%d2359") if time_range.until else "999912312359")
        search_q = f"({search_q})+AND+submittedDate:[{since}+TO+{until}]"

    params = {
        "search_query": search_q,
        "start": 0,
        "max_results": min(limit, 50),
        "sortBy": "submittedDate" if (time_range and time_range.is_bounded) else "relevance",
        "sortOrder": "descending",
    }
    url = f"{_ENDPOINT}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
        body = resp.read().decode("utf-8", errors="replace")

    root = ET.fromstring(body)
    out: list[Result] = []
    for entry in root.findall("a:entry", _NS):
        title = _text(entry, "a:title")
        summary = _text(entry, "a:summary")
        published = _text(entry, "a:published")
        link = ""
        for lnk in entry.findall("a:link", _NS):
            if lnk.get("rel") == "alternate" or lnk.get("type") == "text/html":
                link = lnk.get("href", "")
                break
        if not link:
            link = _text(entry, "a:id")
        authors = [_text(a, "a:name") for a in entry.findall("a:author", _NS)]
        author_str = ", ".join(a for a in authors[:3] if a)
        if len(authors) > 3:
            author_str += f", +{len(authors) - 3} more"

        out.append(Result(
            source="arxiv",
            title=re.sub(r"\s+", " ", title).strip(),
            url=link,
            snippet=re.sub(r"\s+", " ", summary).strip()[:600],
            author=author_str,
            published=published,
            domain="academic",
            engagement_label="preprint",
        ))
    return out


def _text(node, path: str) -> str:
    el = node.find(path, _NS)
    return (el.text or "").strip() if el is not None and el.text else ""


register(SourceSpec(
    name="arxiv",
    domains=["academic"],
    tier="free",
    search=search,
    description="arXiv preprints via public Atom feed",
))
