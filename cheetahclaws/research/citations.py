"""Secondary citation analysis — for a set of academic results, look up
who cited them and flag authors with large scholarly footprints.

Uses Semantic Scholar's graph API:
    /paper/search (we already hit)
    /paper/{id}/citations          — papers citing this one
    /author/{id}                   — author citation totals, h-index

Notable = author with total citationCount >= threshold (default 10000,
user-overridable). Run as a POST-processing step after the main research
run, only when we have academic results.

This adds 2-3 extra API calls per "top paper"; we cap at TOP_N_PAPERS
and MAX_CITING_PAPERS to keep latency + rate-limit impact bounded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .http import get, HttpError

_PAPER_BASE = "https://api.semanticscholar.org/graph/v1"

TOP_N_PAPERS = 3           # how many of the top-ranked papers to expand
MAX_CITING_PAPERS = 30     # per-paper cap on citations to scan
NOTABLE_THRESHOLD = 10000  # author-total citationCount to be "notable"


@dataclass
class NotableCiter:
    name: str
    author_id: str
    total_citations: int
    h_index: int
    affiliation: str = ""
    cited_papers: list[str] = field(default_factory=list)  # paper titles cited


def analyze(
    academic_results,
    threshold: int = NOTABLE_THRESHOLD,
    top_n: int = TOP_N_PAPERS,
    max_citing: int = MAX_CITING_PAPERS,
    config: dict | None = None,
) -> list[NotableCiter]:
    """Find notable authors who have cited the top academic results.

    Only looks at Semantic Scholar results (they have a paperId we can
    follow). Returns notable citers deduped by author_id, sorted by
    total_citations descending.
    """
    # Pick top-ranked Semantic Scholar results
    ss_results = [r for r in academic_results if r.source == "semantic_scholar"]
    if not ss_results:
        return []
    ss_results = ss_results[:top_n]

    headers = {}
    key = (
        (config or {}).get("semantic_scholar_api_key")
        or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        or os.environ.get("S2_API_KEY")
    )
    if key:
        headers["x-api-key"] = key

    notable: dict[str, NotableCiter] = {}

    for paper in ss_results:
        paper_id = _extract_ss_id(paper)
        if not paper_id:
            continue
        try:
            citations = _fetch_citations(paper_id, headers, limit=max_citing)
        except (HttpError, Exception):
            continue

        for citing in citations:
            citing_paper = citing.get("citingPaper") or {}
            authors = citing_paper.get("authors") or []
            for a in authors[:3]:  # typically only first few are "headline" authors
                aid = a.get("authorId")
                name = a.get("name") or ""
                if not aid or not name:
                    continue
                if aid in notable:
                    if paper.title not in notable[aid].cited_papers:
                        notable[aid].cited_papers.append(paper.title)
                    continue

                try:
                    adata = _fetch_author(aid, headers)
                except (HttpError, Exception):
                    continue

                cc = int(adata.get("citationCount") or 0)
                if cc < threshold:
                    continue
                aff_list = adata.get("affiliations") or []
                aff = aff_list[0] if aff_list else ""

                notable[aid] = NotableCiter(
                    name=adata.get("name") or name,
                    author_id=aid,
                    total_citations=cc,
                    h_index=int(adata.get("hIndex") or 0),
                    affiliation=aff,
                    cited_papers=[paper.title],
                )

    ranked = sorted(notable.values(), key=lambda n: n.total_citations, reverse=True)
    return ranked


def _extract_ss_id(result) -> str | None:
    # Semantic Scholar URLs are `https://www.semanticscholar.org/paper/<hash>/<ssid>`
    # or `/paper/<ssid>`. The last path segment is the S2 paper ID.
    url = (result.url or "").rstrip("/")
    if "semanticscholar.org/paper/" in url:
        return url.rsplit("/", 1)[-1]
    # Fallback: DOI / arXiv IDs are accepted by /paper/{id}
    if "arxiv.org/abs/" in url:
        aid = url.rsplit("/", 1)[-1].split("v")[0]
        return f"arXiv:{aid}"
    if "doi.org/" in url:
        return f"DOI:{url.split('doi.org/', 1)[1]}"
    return None


def _fetch_citations(paper_id: str, headers: dict, limit: int) -> list[dict]:
    data = get(
        f"{_PAPER_BASE}/paper/{paper_id}/citations",
        params={"limit": limit, "fields": "citingPaper.authors"},
        headers=headers or None,
    )
    return data.get("data") or []


def _fetch_author(author_id: str, headers: dict) -> dict:
    return get(
        f"{_PAPER_BASE}/author/{author_id}",
        params={"fields": "name,hIndex,citationCount,affiliations,paperCount"},
        headers=headers or None,
    )


def render_notable_section(notable: list[NotableCiter], threshold: int) -> str:
    if not notable:
        return ""
    out = [
        f"## Notable citing authors (≥{threshold:,} total citations)",
        "",
        "| Author | Affiliation | Total cites | h-index | Cited |",
        "|---|---|---|---|---|",
    ]
    for n in notable[:15]:
        cited = "; ".join(t[:40] for t in n.cited_papers[:2])
        if len(n.cited_papers) > 2:
            cited += f" (+{len(n.cited_papers) - 2} more)"
        out.append(
            f"| {n.name} | {n.affiliation or '—'} | "
            f"{n.total_citations:,} | {n.h_index} | {cited} |"
        )
    return "\n".join(out)
