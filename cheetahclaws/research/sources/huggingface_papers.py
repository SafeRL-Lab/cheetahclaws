"""HuggingFace Papers — curated daily papers with upvotes + comments.

Uses the public `/api/daily_papers` endpoint that powers huggingface.co/papers.
Returns a list of the most recently curated papers; we client-side filter
by topic substring. No key required. Optional `HF_TOKEN` in headers does
not affect this endpoint but is included for consistency.
"""
from __future__ import annotations

import os

from ..http import get
from ..types import Result
from . import SourceSpec, register

_ENDPOINT = "https://huggingface.co/api/daily_papers"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    headers = {}
    token = (
        (config or {}).get("hf_token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Endpoint returns the last ~50 curated papers. We pull them and
    # client-side filter by topic — mirrors our Polymarket approach.
    data = get(_ENDPOINT, headers=headers or None)
    if not isinstance(data, list):
        return []

    q_terms = [t.lower() for t in query.split() if len(t) > 2]
    if not q_terms:
        q_terms = [query.lower().strip()]

    out: list[Result] = []
    for item in data:
        paper = item.get("paper") or {}
        title = (paper.get("title") or item.get("title") or "").strip()
        if not title:
            continue

        summary = paper.get("summary") or item.get("summary") or ""
        hay = (title + " " + summary).lower()
        if not any(t in hay for t in q_terms):
            continue

        # Time filter — client-side on publishedAt
        if time_range and time_range.is_bounded:
            published_str = (paper.get("publishedAt")
                             or paper.get("submittedOnDailyAt")
                             or item.get("publishedAt") or "")
            if published_str:
                from datetime import datetime, timezone
                try:
                    pub_dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    if time_range.since and pub_dt < time_range.since:
                        continue
                    if time_range.until and pub_dt > time_range.until:
                        continue
                except ValueError:
                    pass

        arxiv_id = paper.get("id") or ""
        upvotes = int(paper.get("upvotes") or 0)
        num_comments = int(item.get("numComments") or 0)
        authors = [a.get("name", "") for a in (paper.get("authors") or [])]
        author_str = ", ".join(a for a in authors[:3] if a)
        if len(authors) > 3:
            author_str += f", +{len(authors) - 3} more"

        discussion_id = paper.get("discussionId") or ""
        url = f"https://huggingface.co/papers/{arxiv_id}" if arxiv_id else ""
        if not url:
            continue

        published = (
            paper.get("publishedAt")
            or paper.get("submittedOnDailyAt")
            or item.get("publishedAt")
            or ""
        )

        out.append(Result(
            source="huggingface",
            title=title,
            url=url,
            snippet=summary.strip().replace("\n", " ")[:600],
            author=author_str,
            published=published,
            engagement_raw=upvotes + num_comments,
            engagement_label=f"{upvotes} upvotes · {num_comments} comments",
            domain="academic",
            extra={
                "arxiv_id": arxiv_id,
                "upvotes": upvotes,
                "comments": num_comments,
                "discussion_id": discussion_id,
                "project_page": paper.get("projectPage") or "",
            },
        ))
        if len(out) >= limit:
            break

    return out


register(SourceSpec(
    name="huggingface",
    domains=["academic", "tech"],
    tier="free",
    search=search,
    description="HuggingFace Papers — curated papers with upvotes + comments",
))
