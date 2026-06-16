"""Twitter / X via the v2 recent-search API.

Needs `X_API_BEARER_TOKEN` (or `TWITTER_BEARER_TOKEN`). The free / Basic
tier of the X API covers this endpoint with a low monthly cap; use
sparingly and rely on the 24h cache. Silently skips without a key.

Engagement signal: likes + retweets + replies + quotes.
"""
from __future__ import annotations

import os

from ..http import get
from ..types import Result
from . import SourceSkipped, SourceSpec, register

_ENDPOINT = "https://api.twitter.com/2/tweets/search/recent"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    token = (
        (config or {}).get("x_api_bearer_token")
        or os.environ.get("X_API_BEARER_TOKEN")
        or os.environ.get("TWITTER_BEARER_TOKEN")
    )
    if not token:
        raise SourceSkipped(
            "X_API_BEARER_TOKEN / TWITTER_BEARER_TOKEN not set"
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    # X v2 recent search restricts to last 7 days on free/basic tiers.
    # We request public_metrics explicitly so we get the engagement signal
    # without a separate lookup.
    params = {
        "query": f"{query} -is:retweet lang:en",
        "max_results": min(max(10, limit), 100),
        "tweet.fields": "public_metrics,created_at,author_id,lang",
        "expansions": "author_id",
        "user.fields": "username,name,verified",
    }
    if time_range and time_range.is_bounded:
        # X v2 needs ISO 8601 with seconds, UTC
        if time_range.since:
            params["start_time"] = time_range.since.strftime("%Y-%m-%dT%H:%M:%SZ")
        if time_range.until:
            params["end_time"] = time_range.until.strftime("%Y-%m-%dT%H:%M:%SZ")
    data = get(_ENDPOINT, params=params, headers=headers)

    users_by_id: dict[str, dict] = {}
    for u in (data.get("includes") or {}).get("users") or []:
        if u.get("id"):
            users_by_id[u["id"]] = u

    out: list[Result] = []
    for t in data.get("data") or []:
        text = (t.get("text") or "").strip()
        tid = t.get("id")
        author_id = t.get("author_id") or ""
        metrics = t.get("public_metrics") or {}
        likes = int(metrics.get("like_count") or 0)
        retweets = int(metrics.get("retweet_count") or 0)
        replies = int(metrics.get("reply_count") or 0)
        quotes = int(metrics.get("quote_count") or 0)

        user = users_by_id.get(author_id, {})
        uname = user.get("username", "")
        display = user.get("name", uname)
        url = f"https://x.com/{uname}/status/{tid}" if uname and tid else ""
        if not url:
            continue

        # First-line of tweet as "title"; full text as snippet.
        first_line = text.split("\n", 1)[0][:140]
        engagement = likes + retweets * 3 + replies + quotes * 2

        out.append(Result(
            source="twitter",
            title=first_line,
            url=url,
            snippet=text[:500],
            author=f"@{uname}" if uname else display,
            published=t.get("created_at") or "",
            engagement_raw=engagement,
            engagement_label=(
                f"{likes:,} ❤ · {retweets:,} ↻ · {replies:,} 💬"
                + (f" · {quotes:,} quotes" if quotes else "")
            ),
            domain="social",
            extra={"likes": likes, "retweets": retweets, "replies": replies,
                   "quotes": quotes, "verified": bool(user.get("verified"))},
        ))
    return out


register(SourceSpec(
    name="twitter",
    domains=["social", "news"],
    tier="optional",
    search=search,
    requires_env=["X_API_BEARER_TOKEN"],
    description="X / Twitter v2 recent search (7d window) · needs X_API_BEARER_TOKEN",
))
