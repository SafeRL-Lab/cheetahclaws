"""alphaXiv — community discussion layer over arXiv.

alphaXiv does not expose a public full-text search API (verified
2026-04-20: probed endpoints return 500/404). What it does give us is a
stable URL pattern — `https://www.alphaxiv.org/abs/{arxiv_id}` resolves
to that paper's discussion page — so we run an arXiv search internally
and return alphaXiv discussion URLs for each hit. Users click through
for community comments and annotations.

This is intentionally a thin layer: alphaXiv's value is the discussion,
not a separate search. The Result objects we emit surface as a distinct
"platform" in the cross-platform heat table, making it easy to see
which papers have an alphaXiv discussion page.
"""
from __future__ import annotations

import re

from ..types import Result
from . import SourceSpec, register

_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/([\w\.\-]+?)(?:v\d+)?(?:/|$)")


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    # Delegate to the arxiv source for the actual paper search — inherit
    # time_range so alphaxiv naturally respects --range.
    from . import arxiv as _arxiv

    arxiv_results = _arxiv.search(query, limit, config, time_range=time_range)
    out: list[Result] = []
    for r in arxiv_results:
        m = _ARXIV_ID_RE.search(r.url or "")
        if not m:
            continue
        arxiv_id = m.group(1)
        alpha_url = f"https://www.alphaxiv.org/abs/{arxiv_id}"
        out.append(Result(
            source="alphaxiv",
            title=r.title,
            url=alpha_url,
            snippet=(r.snippet or "").strip()[:500],
            author=r.author,
            published=r.published,
            domain="academic",
            engagement_label="community discussion",
            extra={"arxiv_id": arxiv_id, "arxiv_url": r.url},
        ))
    return out


register(SourceSpec(
    name="alphaxiv",
    domains=["academic"],
    tier="free",
    search=search,
    description="alphaXiv — community discussion pages for arXiv papers",
))
