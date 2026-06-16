"""Semantic Scholar — academic search with citation counts + official TL;DRs.

No key required; rate-limited to 100 req/5min (shared pool). Optional
SEMANTIC_SCHOLAR_API_KEY (sometimes called S2_API_KEY) raises the limit.
"""
from __future__ import annotations

import os

from ..http import get
from ..types import Result
from . import SourceSpec, register

_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,abstract,url,year,citationCount,influentialCitationCount,authors,tldr,externalIds,openAccessPdf"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    headers = {}
    key = (
        (config or {}).get("semantic_scholar_api_key")
        or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        or os.environ.get("S2_API_KEY")
    )
    if key:
        headers["x-api-key"] = key

    params = {
        "query": query,
        "limit": min(limit, 50),
        "fields": _FIELDS,
    }
    if time_range and time_range.is_bounded:
        lo = time_range.since.year if time_range.since else 1900
        hi = time_range.until.year if time_range.until else 9999
        params["year"] = f"{lo}-{hi}"

    data = get(_ENDPOINT, params=params, headers=headers)

    out: list[Result] = []
    for paper in data.get("data") or []:
        title = (paper.get("title") or "").strip()
        if not title:
            continue
        authors = [a.get("name", "") for a in paper.get("authors") or []]
        author_str = ", ".join(a for a in authors[:3] if a)
        if len(authors) > 3:
            author_str += f", +{len(authors) - 3} more"

        tldr = (paper.get("tldr") or {}).get("text") or ""
        abstract = paper.get("abstract") or ""
        snippet = (tldr or abstract)[:600]

        citations = int(paper.get("citationCount") or 0)
        influential = int(paper.get("influentialCitationCount") or 0)

        url = paper.get("url") or ""
        ext = paper.get("externalIds") or {}
        if not url and ext.get("DOI"):
            url = f"https://doi.org/{ext['DOI']}"
        if not url and ext.get("ArXiv"):
            url = f"https://arxiv.org/abs/{ext['ArXiv']}"
        if not url:
            continue

        year = paper.get("year")
        published = f"{year}-01-01" if year else ""

        out.append(Result(
            source="semantic_scholar",
            title=title,
            url=url,
            snippet=snippet,
            author=author_str,
            published=published,
            engagement_raw=citations,
            engagement_label=f"{citations:,} citations ({influential} influential)",
            domain="academic",
            extra={
                "influential_citations": influential,
                "open_access_pdf": (paper.get("openAccessPdf") or {}).get("url", ""),
            },
        ))
    return out


register(SourceSpec(
    name="semantic_scholar",
    domains=["academic"],
    tier="free",
    search=search,
    description="Semantic Scholar academic search with citation counts + TL;DRs",
))
