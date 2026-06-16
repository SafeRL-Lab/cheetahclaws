"""Reddit public JSON — searches across all of Reddit for the query.

No auth, no key. Reddit's public JSON endpoint needs a distinctive
User-Agent (the default `python-urllib/...` gets blocked); the shared
DEFAULT_UA handles that. Rate-limited to ~60 req/min per IP.
"""
from __future__ import annotations

from datetime import datetime, timezone

from ..http import get
from ..types import Result
from . import SourceSpec, register

_ENDPOINT = "https://www.reddit.com/search.json"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    # Reddit's public search accepts t ∈ {hour, day, week, month, year, all}
    t = _reddit_t_for(time_range)
    data = get(_ENDPOINT, params={
        "q": query,
        "limit": min(limit, 50),
        "sort": "relevance",
        "t": t,
        "raw_json": 1,
    })

    out: list[Result] = []
    for child in (data.get("data") or {}).get("children") or []:
        post = child.get("data") or {}
        title = post.get("title") or ""
        if not title:
            continue
        subreddit = post.get("subreddit") or ""
        url = f"https://www.reddit.com{post.get('permalink') or ''}"
        upvotes = int(post.get("ups") or 0)
        comments = int(post.get("num_comments") or 0)
        selftext = (post.get("selftext") or "")[:400]
        author = post.get("author") or ""
        created = post.get("created_utc")
        if created:
            try:
                published = datetime.fromtimestamp(
                    float(created), tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, OSError):
                published = ""
        else:
            published = ""

        out.append(Result(
            source="reddit",
            title=f"r/{subreddit}: {title}",
            url=url,
            snippet=selftext,
            author=f"u/{author}" if author else "",
            published=published,
            engagement_raw=upvotes + (comments // 2),
            engagement_label=f"{upvotes:,} upvotes · {comments} comments",
            domain="social",
            extra={"subreddit": subreddit, "upvotes": upvotes, "comments": comments},
        ))
    return out


def _reddit_t_for(time_range) -> str:
    """Map an arbitrary TimeRange to Reddit's closest native t={hour|day|week|
    month|year|all}. Use small epsilons so that `parse_range("7d")`
    (which will be a hair over 7 days by the time we get here) maps
    cleanly to 'week'."""
    if not time_range or not time_range.is_bounded or not time_range.since:
        return "month"
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    days = (now - time_range.since).total_seconds() / 86400.0
    if days <= 0.05:  return "hour"
    if days <= 1.1:   return "day"
    if days <= 7.5:   return "week"
    if days <= 31.5:  return "month"
    if days <= 366:   return "year"
    return "all"


register(SourceSpec(
    name="reddit",
    domains=["social", "news"],
    tier="free",
    search=search,
    description="Reddit site-wide search — t auto-maps to user's --range",
))
