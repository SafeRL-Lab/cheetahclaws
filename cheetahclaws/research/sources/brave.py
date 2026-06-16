"""Brave Search API — web + news. Free tier: 2000 req/month.

Needs BRAVE_API_KEY. Silently skips without a key.
"""
from __future__ import annotations

import os

from ..http import get
from ..types import Result
from . import SourceSkipped, SourceSpec, register

_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    key = (
        (config or {}).get("brave_api_key")
        or os.environ.get("BRAVE_API_KEY")
    )
    if not key:
        raise SourceSkipped("BRAVE_API_KEY not set")

    headers = {
        "X-Subscription-Token": key,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }
    params = {
        "q": query,
        "count": min(limit, 20),
        "safesearch": "moderate",
    }
    # Brave freshness: pd (past day), pw (past week), pm (past month), py (past year)
    if time_range and time_range.is_bounded and time_range.since:
        from datetime import datetime, timezone
        delta_days = (datetime.now(timezone.utc) - time_range.since).days
        if delta_days <= 1:   params["freshness"] = "pd"
        elif delta_days <= 7: params["freshness"] = "pw"
        elif delta_days <= 31: params["freshness"] = "pm"
        elif delta_days <= 366: params["freshness"] = "py"

    data = get(_ENDPOINT, params=params, headers=headers)

    out: list[Result] = []
    for item in (data.get("web") or {}).get("results") or []:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        if not title or not url:
            continue
        description = (item.get("description") or "")[:500]
        # Brave snips include HTML like <strong> — strip
        import re as _re
        description = _re.sub(r"<[^>]+>", "", description).strip()
        age = item.get("age") or ""

        out.append(Result(
            source="brave",
            title=title,
            url=url,
            snippet=description,
            published=item.get("page_age") or "",
            domain="web",
            engagement_label=age or "web",
            extra={"language": item.get("language", "")},
        ))
    return out


register(SourceSpec(
    name="brave",
    domains=["web", "news", "tech", "finance"],
    tier="optional",
    search=search,
    requires_env=["BRAVE_API_KEY"],
    description="Brave Search — web results (optional, BRAVE_API_KEY)",
))
