"""StackOverflow via the Stack Exchange API. Free, no key.

Without a key: 300 req/day per IP. With STACKEXCHANGE_KEY: 10000 req/day
(the key is a "client key" — it's not a secret, only raises the limit).
"""
from __future__ import annotations

import html
import os

from ..http import get
from ..types import Result
from . import SourceSpec, register

_ENDPOINT = "https://api.stackexchange.com/2.3/search/advanced"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    params = {
        "q": query,
        "site": "stackoverflow",
        "pagesize": min(limit, 50),
        "order": "desc",
        "sort": "relevance",
        "filter": "withbody",  # include body text
    }
    key = (
        (config or {}).get("stackexchange_key")
        or os.environ.get("STACKEXCHANGE_KEY")
    )
    if key:
        params["key"] = key
    if time_range and time_range.is_bounded:
        if time_range.since:
            params["fromdate"] = int(time_range.since.timestamp())
        if time_range.until:
            params["todate"] = int(time_range.until.timestamp())

    data = get(_ENDPOINT, params=params)

    out: list[Result] = []
    for item in data.get("items") or []:
        title = html.unescape(item.get("title") or "")
        if not title:
            continue
        url = item.get("link") or ""
        if not url:
            continue
        score = int(item.get("score") or 0)
        answers = int(item.get("answer_count") or 0)
        views = int(item.get("view_count") or 0)
        body = html.unescape(item.get("body") or "")[:400]

        # Strip crude HTML from body
        import re as _re
        body = _re.sub(r"<[^>]+>", " ", body)
        body = _re.sub(r"\s+", " ", body).strip()

        owner = (item.get("owner") or {}).get("display_name", "")
        creation = item.get("last_activity_date") or item.get("creation_date")
        published = ""
        if creation:
            from datetime import datetime, timezone
            try:
                published = datetime.fromtimestamp(
                    int(creation), tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, OSError):
                pass

        engagement = score * 10 + answers * 5 + (views // 100)

        out.append(Result(
            source="stackoverflow",
            title=title,
            url=url,
            snippet=body,
            author=owner,
            published=published,
            engagement_raw=engagement,
            engagement_label=f"score {score} · {answers} answers · {views:,} views",
            domain="tech",
            extra={"score": score, "answers": answers, "views": views,
                   "is_answered": bool(item.get("is_answered"))},
        ))
    return out


register(SourceSpec(
    name="stackoverflow",
    domains=["tech"],
    tier="free",
    search=search,
    description="StackOverflow questions — score, answers, view counts",
))
