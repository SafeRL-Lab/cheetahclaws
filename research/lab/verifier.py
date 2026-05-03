"""research/lab/verifier.py — citation existence + author check.

Catches the #1 failure mode of LLM-generated papers: fabricated
citations.  We verify each claimed citation against three free APIs
in priority order:

  1. arXiv API           (https://export.arxiv.org/api/query)
  2. Semantic Scholar    (https://api.semanticscholar.org/graph/v1)
  3. CrossRef            (https://api.crossref.org/works)

Net result for each citation:

  status = "verified"      — found, title + author lists match closely
  status = "ambiguous"     — found by title but author overlap < 50%
  status = "not_found"     — none of the APIs found a match
  status = "verification_skipped"  — no network / all APIs failed

We don't fail the run on any single ``not_found``; the orchestrator
counts them and decides whether to surface a warning, drop the
citation, or ask the writer to find a real one.

API rate limits are conservative:
  * arXiv: 1 request / 3 seconds (their guideline)
  * Semantic Scholar: 100 requests / 5 minutes for unauthenticated
  * CrossRef: polite pool, 50 / second

We sleep ~3 s between arXiv calls; for batch verification of N
citations, total time is O(N * 3s) wall-clock, dominated by arXiv.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator, Optional
from xml.etree import ElementTree as ET


_ARXIV_API     = "https://export.arxiv.org/api/query"
_SEM_SCHOLAR   = "https://api.semanticscholar.org/graph/v1/paper/search"
_CROSSREF_API  = "https://api.crossref.org/works"

_USER_AGENT = "cheetahclaws-research-lab/0.1 (mailto:research@example.com)"


@dataclass
class Citation:
    """A claim about a paper made by the writer."""
    key: str                      # local citation key, e.g. "vaswani2017"
    title: str                    # claimed title
    authors: list[str]            # claimed author family names
    year: Optional[int] = None    # claimed year
    venue: Optional[str] = None   # claimed venue (journal/conf)
    arxiv_id: Optional[str] = None  # if explicitly claimed
    doi: Optional[str] = None       # if explicitly claimed


@dataclass
class CitationVerification:
    citation: Citation
    status: str                   # verified|ambiguous|not_found|verification_skipped
    matched_title: Optional[str] = None
    matched_authors: list[str] = field(default_factory=list)
    matched_url: Optional[str] = None
    source: Optional[str] = None  # "arxiv" | "semantic_scholar" | "crossref"
    notes: Optional[str] = None


@dataclass
class VerifierResult:
    verifications: list[CitationVerification]
    n_verified: int
    n_ambiguous: int
    n_not_found: int
    n_skipped: int

    @property
    def fabrication_rate(self) -> float:
        total = max(1, len(self.verifications))
        return self.n_not_found / total

    def fabricated(self) -> list[CitationVerification]:
        return [v for v in self.verifications if v.status == "not_found"]


# ── Public surface ────────────────────────────────────────────────────────


def verify_citations(citations: list[Citation],
                     *, sleep_s: float = 3.1,
                     timeout_s: float = 10.0,
                     ) -> VerifierResult:
    verifs = []
    for c in citations:
        v = verify_one(c, timeout_s=timeout_s)
        verifs.append(v)
        # Be polite to arXiv even if we didn't actually call it for this one.
        time.sleep(sleep_s)
    counts = {"verified": 0, "ambiguous": 0, "not_found": 0,
              "verification_skipped": 0}
    for v in verifs:
        counts[v.status] = counts.get(v.status, 0) + 1
    return VerifierResult(
        verifications=verifs,
        n_verified=counts["verified"],
        n_ambiguous=counts["ambiguous"],
        n_not_found=counts["not_found"],
        n_skipped=counts["verification_skipped"],
    )


def verify_one(citation: Citation, *, timeout_s: float = 10.0
               ) -> CitationVerification:
    """Try the three APIs in order; return on first solid match."""
    n_skipped = 0   # count APIs that failed at the network layer

    # 1. If arxiv_id explicit — check arXiv directly
    if citation.arxiv_id:
        v = _check_arxiv_by_id(citation, timeout_s=timeout_s)
        if v.status == "verified":
            return v
        if v.status == "verification_skipped":
            n_skipped += 1

    # 2. arXiv title search
    v_arxiv = _check_arxiv(citation, timeout_s=timeout_s)
    if v_arxiv.status == "verified":
        return v_arxiv
    if v_arxiv.status == "verification_skipped":
        n_skipped += 1

    # 3. Semantic Scholar (broader coverage)
    v_sem = _check_semantic_scholar(citation, timeout_s=timeout_s)
    if v_sem.status == "verified":
        return v_sem
    if v_sem.status == "verification_skipped":
        n_skipped += 1

    # 4. CrossRef (DOI catalog; good for journal papers)
    v_cr = _check_crossref(citation, timeout_s=timeout_s)
    if v_cr.status == "verified":
        return v_cr
    if v_cr.status == "verification_skipped":
        n_skipped += 1

    # If literally everything was a network failure, propagate that —
    # we can't honestly call this "not_found".
    if n_skipped >= 3:
        return CitationVerification(citation=citation,
                                     status="verification_skipped",
                                     notes="all APIs failed at network layer")

    # Otherwise pick the best non-skipped partial we got.
    partials = [p for p in (v_sem, v_arxiv, v_cr)
                if p.status not in ("verified", "verification_skipped")]
    for p in partials:
        if p.status == "ambiguous":
            return p
    if partials:
        return partials[0]

    return CitationVerification(citation=citation, status="not_found")


# ── arXiv ─────────────────────────────────────────────────────────────────


def _check_arxiv_by_id(citation: Citation, timeout_s: float
                       ) -> CitationVerification:
    aid = citation.arxiv_id or ""
    aid = aid.replace("arxiv:", "").replace("arXiv:", "").strip()
    if not aid:
        return CitationVerification(citation=citation,
                                     status="verification_skipped",
                                     notes="empty arxiv_id")
    url = f"{_ARXIV_API}?id_list={urllib.parse.quote(aid)}&max_results=1"
    try:
        text = _http_get(url, timeout_s=timeout_s)
    except Exception as exc:
        return CitationVerification(citation=citation,
                                     status="verification_skipped",
                                     notes=f"arxiv: {exc}")
    entries = _parse_arxiv_atom(text)
    if not entries:
        return CitationVerification(citation=citation, status="not_found",
                                     source="arxiv")
    e = entries[0]
    return _score_arxiv_match(citation, e)


def _check_arxiv(citation: Citation, timeout_s: float
                 ) -> CitationVerification:
    if not citation.title:
        return CitationVerification(citation=citation,
                                     status="verification_skipped",
                                     notes="no title")
    q = urllib.parse.quote(f'ti:"{citation.title}"')
    url = f"{_ARXIV_API}?search_query={q}&max_results=3"
    try:
        text = _http_get(url, timeout_s=timeout_s)
    except Exception as exc:
        return CitationVerification(citation=citation,
                                     status="verification_skipped",
                                     notes=f"arxiv: {exc}")
    entries = _parse_arxiv_atom(text)
    if not entries:
        return CitationVerification(citation=citation, status="not_found",
                                     source="arxiv")
    best = max(entries,
               key=lambda e: _title_similarity(e.get("title", ""),
                                                 citation.title))
    return _score_arxiv_match(citation, best)


def _score_arxiv_match(citation: Citation, e: dict) -> CitationVerification:
    title_sim = _title_similarity(e.get("title", ""), citation.title)
    if title_sim < 0.55:
        return CitationVerification(citation=citation, status="not_found",
                                     source="arxiv",
                                     matched_title=e.get("title"))
    author_overlap = _author_overlap(citation.authors, e.get("authors", []))
    if author_overlap < 0.5:
        return CitationVerification(citation=citation, status="ambiguous",
                                     source="arxiv",
                                     matched_title=e.get("title"),
                                     matched_authors=e.get("authors", []),
                                     matched_url=e.get("url"),
                                     notes=f"title sim {title_sim:.2f},"
                                           f" authors overlap {author_overlap:.2f}")
    return CitationVerification(citation=citation, status="verified",
                                 source="arxiv",
                                 matched_title=e.get("title"),
                                 matched_authors=e.get("authors", []),
                                 matched_url=e.get("url"))


def _parse_arxiv_atom(xml: str) -> list[dict]:
    NS = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    out = []
    for entry in root.findall("atom:entry", NS):
        title_el = entry.find("atom:title", NS)
        link_el = entry.find("atom:id", NS)
        authors = []
        for a in entry.findall("atom:author", NS):
            name = a.find("atom:name", NS)
            if name is not None and name.text:
                authors.append(name.text.strip())
        out.append({
            "title": (title_el.text or "").strip() if title_el is not None else "",
            "authors": authors,
            "url": (link_el.text or "").strip() if link_el is not None else "",
        })
    return out


# ── Semantic Scholar ──────────────────────────────────────────────────────


def _check_semantic_scholar(citation: Citation, timeout_s: float
                             ) -> CitationVerification:
    if not citation.title:
        return CitationVerification(citation=citation,
                                     status="verification_skipped",
                                     notes="no title")
    q = urllib.parse.quote(citation.title)
    url = (f"{_SEM_SCHOLAR}?query={q}&limit=3"
           "&fields=title,authors,year,externalIds,url,venue")
    try:
        body = _http_get(url, timeout_s=timeout_s,
                         headers={"Accept": "application/json"})
    except Exception as exc:
        return CitationVerification(citation=citation,
                                     status="verification_skipped",
                                     notes=f"sem_scholar: {exc}")
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return CitationVerification(citation=citation,
                                     status="verification_skipped",
                                     notes="sem_scholar: bad json")
    papers = data.get("data") or []
    if not papers:
        return CitationVerification(citation=citation, status="not_found",
                                     source="semantic_scholar")
    best = max(
        papers,
        key=lambda p: _title_similarity(p.get("title") or "", citation.title),
    )
    title = best.get("title") or ""
    title_sim = _title_similarity(title, citation.title)
    if title_sim < 0.55:
        return CitationVerification(citation=citation, status="not_found",
                                     source="semantic_scholar",
                                     matched_title=title)
    auths = [a.get("name", "") for a in (best.get("authors") or [])]
    overlap = _author_overlap(citation.authors, auths)
    if overlap < 0.5:
        return CitationVerification(citation=citation, status="ambiguous",
                                     source="semantic_scholar",
                                     matched_title=title,
                                     matched_authors=auths,
                                     matched_url=best.get("url"),
                                     notes=f"title sim {title_sim:.2f},"
                                           f" authors overlap {overlap:.2f}")
    return CitationVerification(citation=citation, status="verified",
                                 source="semantic_scholar",
                                 matched_title=title,
                                 matched_authors=auths,
                                 matched_url=best.get("url"))


# ── CrossRef ──────────────────────────────────────────────────────────────


def _check_crossref(citation: Citation, timeout_s: float
                    ) -> CitationVerification:
    if citation.doi:
        url = f"{_CROSSREF_API}/{urllib.parse.quote(citation.doi)}"
        try:
            body = _http_get(url, timeout_s=timeout_s,
                             headers={"Accept": "application/json"})
        except Exception as exc:
            return CitationVerification(citation=citation,
                                         status="verification_skipped",
                                         notes=f"crossref: {exc}")
        try:
            data = json.loads(body).get("message") or {}
        except json.JSONDecodeError:
            return CitationVerification(citation=citation,
                                         status="not_found",
                                         source="crossref")
        return _score_crossref(citation, data)
    if not citation.title:
        return CitationVerification(citation=citation,
                                     status="verification_skipped",
                                     notes="no title")
    q = urllib.parse.quote(citation.title)
    url = f"{_CROSSREF_API}?query.title={q}&rows=3"
    try:
        body = _http_get(url, timeout_s=timeout_s,
                         headers={"Accept": "application/json"})
    except Exception as exc:
        return CitationVerification(citation=citation,
                                     status="verification_skipped",
                                     notes=f"crossref: {exc}")
    try:
        items = json.loads(body).get("message", {}).get("items") or []
    except json.JSONDecodeError:
        items = []
    if not items:
        return CitationVerification(citation=citation, status="not_found",
                                     source="crossref")
    best = max(items, key=lambda p: _title_similarity(
        " ".join(p.get("title") or []), citation.title))
    return _score_crossref(citation, best)


def _score_crossref(citation: Citation, data: dict) -> CitationVerification:
    title = " ".join(data.get("title") or [])
    title_sim = _title_similarity(title, citation.title)
    if title_sim < 0.55:
        return CitationVerification(citation=citation, status="not_found",
                                     source="crossref",
                                     matched_title=title)
    auths_raw = data.get("author") or []
    auths = [(" ".join([a.get("given") or "", a.get("family") or ""])).strip()
             for a in auths_raw]
    overlap = _author_overlap(citation.authors, auths)
    url = data.get("URL") or ""
    if overlap < 0.5:
        return CitationVerification(citation=citation, status="ambiguous",
                                     source="crossref",
                                     matched_title=title, matched_authors=auths,
                                     matched_url=url,
                                     notes=f"title sim {title_sim:.2f},"
                                           f" authors overlap {overlap:.2f}")
    return CitationVerification(citation=citation, status="verified",
                                 source="crossref",
                                 matched_title=title, matched_authors=auths,
                                 matched_url=url)


# ── Similarity helpers ────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[\W_]+", " ", text)
    return " ".join(text.split())


def _title_similarity(a: str, b: str) -> float:
    """Jaccard on word sets after normalization. Cheap and good enough."""
    if not a or not b:
        return 0.0
    sa = set(_normalize(a).split())
    sb = set(_normalize(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _author_overlap(claimed: list[str], found: list[str]) -> float:
    """Surname-based overlap; lenient because authors get formatted in
    many ways ("First Last", "Last, F.", "F. Last", just last name)."""
    if not claimed or not found:
        # If neither side asserts, don't penalize.
        return 1.0 if not claimed else 0.0
    cset = {_last_name(a) for a in claimed if a}
    fset = {_last_name(a) for a in found if a}
    cset.discard("")
    fset.discard("")
    if not cset or not fset:
        return 1.0
    return len(cset & fset) / len(cset | fset)


def _last_name(name: str) -> str:
    """Return the last whitespace-separated token, normalized."""
    name = name.strip()
    if "," in name:
        # "Vaswani, Ashish" or "Vaswani, A." — surname before comma
        return _normalize(name.split(",", 1)[0])
    parts = name.split()
    return _normalize(parts[-1] if parts else "")


# ── HTTP helper ───────────────────────────────────────────────────────────


def _http_get(url: str, *, timeout_s: float,
              headers: Optional[dict] = None) -> str:
    h = {"User-Agent": _USER_AGENT}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")
