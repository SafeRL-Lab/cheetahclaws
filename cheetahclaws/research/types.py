"""Shared data types for the research package."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Domain = Literal["academic", "tech", "finance", "news", "social", "web"]


@dataclass
class Result:
    """A normalized result from any source.

    engagement_raw is the source's native signal (HN points, GitHub stars,
    Reddit upvotes, Semantic Scholar citations…). The ranker normalizes
    these across sources into engagement_score in [0, 1].
    """
    source: str
    title: str
    url: str
    snippet: str = ""
    author: str = ""
    published: str = ""
    engagement_raw: int = 0
    engagement_label: str = ""
    domain: Domain = "web"
    extra: dict = field(default_factory=dict)
    engagement_score: float = 0.0


@dataclass
class SourceStatus:
    """Per-source execution outcome — surfaced in the brief's 'Missed' section."""
    name: str
    ok: bool
    count: int = 0
    duration_ms: int = 0
    error: str = ""
    skipped_reason: str = ""


@dataclass
class Brief:
    """A complete research brief, ready to render."""
    topic: str
    domains: list[Domain]
    results: list[Result]
    statuses: list[SourceStatus]
    synthesis: str = ""
    total_duration_ms: int = 0
    cache_hits: int = 0

    def by_domain(self) -> dict[Domain, list[Result]]:
        out: dict[Domain, list[Result]] = {}
        for r in self.results:
            out.setdefault(r.domain, []).append(r)
        for dom in out:
            out[dom].sort(key=lambda x: x.engagement_score, reverse=True)
        return out
