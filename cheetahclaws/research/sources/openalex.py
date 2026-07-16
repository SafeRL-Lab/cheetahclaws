"""OpenAlex — fully open academic graph, 250M+ works.

Free up to 100k req/day (polite pool). Provides a `mailto=` polite
identifier via OPENALEX_EMAIL or the cheetahclaws config `research_email`;
this grants higher rate limits. No hard auth required.
"""
from __future__ import annotations

import os

from ..http import get
from ..types import Result
from . import SourceSpec, register

_ENDPOINT = "https://api.openalex.org/works"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    params = {
        "search": query,
        "per-page": min(limit, 50),
        "sort": "relevance_score:desc",
    }
    email = (
        (config or {}).get("research_email")
        or os.environ.get("OPENALEX_EMAIL")
    )
    if email:
        params["mailto"] = email
    if time_range and time_range.is_bounded:
        filters = []
        if time_range.since:
            filters.append(f"from_publication_date:{time_range.since.strftime('%Y-%m-%d')}")
        if time_range.until:
            filters.append(f"to_publication_date:{time_range.until.strftime('%Y-%m-%d')}")
        params["filter"] = ",".join(filters)

    data = get(_ENDPOINT, params=params)

    out: list[Result] = []
    for work in data.get("results") or []:
        title = (work.get("title") or work.get("display_name") or "").strip()
        if not title:
            continue
        url = (
            work.get("doi")
            or (work.get("primary_location") or {}).get("landing_page_url")
            or work.get("id")
            or ""
        )
        if not url:
            continue

        authors = [
            (a.get("author") or {}).get("display_name", "")
            for a in (work.get("authorships") or [])
        ]
        author_str = ", ".join(a for a in authors[:3] if a)
        if len(authors) > 3:
            author_str += f", +{len(authors) - 3} more"

        citations = int(work.get("cited_by_count") or 0)
        year = work.get("publication_year")
        published = f"{year}-01-01" if year else (work.get("publication_date") or "")

        abstract_idx = work.get("abstract_inverted_index") or {}
        abstract = _reconstruct_abstract(abstract_idx)[:600]

        venue = (
            (work.get("primary_location") or {}).get("source") or {}
        ).get("display_name", "")

        out.append(Result(
            source="openalex",
            title=title,
            url=url,
            snippet=abstract,
            author=author_str,
            published=published,
            engagement_raw=citations,
            engagement_label=f"{citations:,} citations",
            domain="academic",
            extra={"venue": venue},
        ))
    return out


def _reconstruct_abstract(idx: dict) -> str:
    """OpenAlex ships abstracts as a {word: [positions]} inverted index."""
    if not idx:
        return ""
    words_at: dict[int, str] = {}
    for word, positions in idx.items():
        for p in positions:
            words_at[p] = word
    if not words_at:
        return ""
    ordered = [words_at[i] for i in sorted(words_at.keys())]
    return " ".join(ordered)


register(SourceSpec(
    name="openalex",
    domains=["academic"],
    tier="free",
    search=search,
    description="OpenAlex — 250M+ open academic works with citation counts",
))
