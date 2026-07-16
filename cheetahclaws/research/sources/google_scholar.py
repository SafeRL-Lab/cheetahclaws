"""Google Scholar — via the `scholarly` library (optional dependency).

Google Scholar has NO official API. The `scholarly` package scrapes
Scholar's public HTML and is the community-standard way to access it
programmatically. It's brittle (Scholar sometimes rate-limits and serves
CAPTCHAs), slow (each query ~5-20s), and fragile across versions — but
when it works, it's unique: no other source gives you Scholar's exact
citation count or the "cited by" linkage.

This source is `tier=optional`: it silently skips if `scholarly` isn't
installed, and skips with an explanation if the user sets
`SKIP_GOOGLE_SCHOLAR=1` to opt out even when installed.

Install: `pip install scholarly`
"""
from __future__ import annotations

import os

from ..types import Result
from . import SourceSkipped, SourceSpec, register


def search(query: str, limit: int = 10, config: dict | None = None,
           time_range=None) -> list[Result]:
    if os.environ.get("SKIP_GOOGLE_SCHOLAR"):
        raise SourceSkipped("SKIP_GOOGLE_SCHOLAR=1 set")

    try:
        from scholarly import scholarly  # type: ignore
    except ImportError:
        raise SourceSkipped(
            "scholarly package not installed — `pip install scholarly` to enable"
        )

    # scholarly doesn't natively support date-range filtering on search;
    # we apply it client-side post-fetch.
    search_iter = scholarly.search_pubs(query)

    out: list[Result] = []
    pulled = 0
    max_pull = min(limit * 2, 20)  # over-fetch so we have room to filter

    while pulled < max_pull:
        try:
            pub = next(search_iter)
        except StopIteration:
            break
        except Exception as e:
            # scholarly raises various things on rate-limit / CAPTCHA
            raise SourceSkipped(
                f"Google Scholar blocked the query ({type(e).__name__}: {str(e)[:120]}). "
                f"Wait a few minutes or set a proxy via `scholarly.use_proxy(...)`."
            )
        pulled += 1

        bib = pub.get("bib") or {}
        title = (bib.get("title") or "").strip()
        if not title:
            continue
        pub_url = pub.get("pub_url") or pub.get("eprint_url") or ""
        author_list = bib.get("author") or []
        if isinstance(author_list, list):
            author_str = ", ".join(author_list[:3])
            if len(author_list) > 3:
                author_str += f", +{len(author_list) - 3} more"
        else:
            author_str = str(author_list)
        year = bib.get("pub_year") or bib.get("year") or ""
        published = f"{year}-01-01" if str(year).isdigit() else ""

        # Client-side time filter
        if time_range and time_range.is_bounded and year and str(year).isdigit():
            y = int(year)
            if time_range.since and y < time_range.since.year:
                continue
            if time_range.until and y > time_range.until.year:
                continue

        num_citations = int(pub.get("num_citations") or 0)
        abstract = (bib.get("abstract") or "").strip()[:600]
        venue = bib.get("venue") or ""

        out.append(Result(
            source="google_scholar",
            title=title,
            url=pub_url or pub.get("url_scholarbib") or "",
            snippet=abstract,
            author=author_str,
            published=published,
            engagement_raw=num_citations,
            engagement_label=f"{num_citations:,} citations (Scholar)",
            domain="academic",
            extra={
                "venue": venue,
                "scholar_author_id": pub.get("author_id") or [],
                "cites_url": pub.get("citedby_url") or "",
            },
        ))
        if len(out) >= limit:
            break

    return out


register(SourceSpec(
    name="google_scholar",
    domains=["academic"],
    tier="optional",
    search=search,
    requires_env=["scholarly (pip install scholarly)"],
    description="Google Scholar via `scholarly` package — brittle but unique",
))
