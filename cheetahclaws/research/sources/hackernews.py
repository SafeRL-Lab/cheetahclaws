"""HackerNews via Algolia search API. Free, no key."""
from __future__ import annotations

from datetime import datetime, timezone

from ..http import get
from ..types import Result
from . import SourceSpec, register

_ENDPOINT = "http://hn.algolia.com/api/v1/search"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    # `search` ranks by relevance; `search_by_date` ranks by recency. We
    # use relevance — the ranker layer re-sorts with our own engagement
    # formula anyway.
    params = {
        "query": query,
        "hitsPerPage": min(limit, 50),
        "tags": "(story,comment)",
    }
    if time_range and time_range.is_bounded:
        filters = []
        if time_range.since:
            filters.append(f"created_at_i>{int(time_range.since.timestamp())}")
        if time_range.until:
            filters.append(f"created_at_i<{int(time_range.until.timestamp())}")
        params["numericFilters"] = ",".join(filters)

    data = get(_ENDPOINT, params=params)

    out: list[Result] = []
    for hit in data.get("hits") or []:
        title = (
            hit.get("title")
            or hit.get("story_title")
            or (hit.get("comment_text") or "")[:120]
        )
        if not title:
            continue
        obj_id = hit.get("objectID")
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={obj_id}"
        points = int(hit.get("points") or 0)
        num_comments = int(hit.get("num_comments") or 0)
        author = hit.get("author") or ""
        created_at = hit.get("created_at") or ""

        # Engagement = points + 0.5×comments (comments are softer signal)
        engagement = points + (num_comments // 2)

        snippet_parts = []
        if hit.get("story_text"):
            snippet_parts.append(hit["story_text"][:300])
        elif hit.get("comment_text"):
            snippet_parts.append(hit["comment_text"][:300])
        snippet = " ".join(snippet_parts)

        out.append(Result(
            source="hackernews",
            title=title,
            url=url,
            snippet=snippet,
            author=author,
            published=created_at,
            engagement_raw=engagement,
            engagement_label=f"{points} pts · {num_comments} comments",
            domain="social",
            extra={
                "hn_discussion": f"https://news.ycombinator.com/item?id={obj_id}",
                "points": points,
                "comments": num_comments,
            },
        ))
    return out


register(SourceSpec(
    name="hackernews",
    domains=["tech", "social", "news"],
    tier="free",
    search=search,
    description="HackerNews (Algolia search) — points + comment counts",
))
