"""Tavily Search API — production-grade web search. Free tier: 1000 req/month.

Needs TAVILY_API_KEY. Silently skips itself without a key.
"""
from __future__ import annotations

import os

from ..http import post_json
from ..types import Result
from . import SourceSkipped, SourceSpec, register

_ENDPOINT = "https://api.tavily.com/search"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    key = (
        (config or {}).get("tavily_api_key")
        or os.environ.get("TAVILY_API_KEY")
    )
    if not key:
        raise SourceSkipped("TAVILY_API_KEY not set")

    payload = {
        "api_key": key,
        "query": query,
        "max_results": min(limit, 20),
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
    }
    if time_range and time_range.is_bounded:
        if time_range.since:
            payload["start_published_date"] = time_range.since.strftime("%Y-%m-%d")
        if time_range.until:
            payload["end_published_date"] = time_range.until.strftime("%Y-%m-%d")
    data = post_json(_ENDPOINT, payload)

    out: list[Result] = []
    for item in data.get("results") or []:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        if not title or not url:
            continue
        content = (item.get("content") or "")[:500]
        score = float(item.get("score") or 0.0)
        published = item.get("published_date") or ""
        out.append(Result(
            source="tavily",
            title=title,
            url=url,
            snippet=content,
            published=published,
            engagement_raw=int(score * 100),
            engagement_label=f"relevance {score:.2f}",
            domain="web",
        ))
    return out


register(SourceSpec(
    name="tavily",
    domains=["web", "news", "tech", "finance", "academic"],
    tier="optional",
    search=search,
    requires_env=["TAVILY_API_KEY"],
    description="Tavily AI search — production web search (optional, TAVILY_API_KEY)",
))
