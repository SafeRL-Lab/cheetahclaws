"""SEC EDGAR full-text search — US public company filings.

EDGAR's full-text endpoint (efts.sec.gov) requires a User-Agent that
identifies the requester. No API key, but SEC asks for polite usage.
"""
from __future__ import annotations

import os

from ..http import get
from ..types import Result
from . import SourceSpec, register

_ENDPOINT = "https://efts.sec.gov/LATEST/search-index"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    # SEC requires identification. We use the cheetahclaws-global UA with
    # an optional contact email from config.
    contact = (
        (config or {}).get("research_email")
        or os.environ.get("SEC_CONTACT_EMAIL")
        or "research@cheetahclaws.local"
    )
    headers = {
        "User-Agent": f"CheetahClaws-Research/1.0 ({contact})",
        "Accept": "application/json",
    }

    startdt = (time_range.since.strftime("%Y-%m-%d")
               if time_range and time_range.since else "2024-01-01")
    enddt = (time_range.until.strftime("%Y-%m-%d")
             if time_range and time_range.until else "2030-12-31")

    data = get(_ENDPOINT, params={
        "q": f'"{query}"',
        "dateRange": "custom",
        "startdt": startdt,
        "enddt": enddt,
        "forms": "10-K,10-Q,8-K,S-1,DEF 14A,13F-HR",
    }, headers=headers)

    out: list[Result] = []
    hits = ((data.get("hits") or {}).get("hits") or [])
    for hit in hits[:limit]:
        source = hit.get("_source") or {}
        adsh = (hit.get("_id") or "").split(":")[0]
        display = source.get("display_names") or []
        company = display[0] if display else (source.get("name") or "")
        form = source.get("form") or ""
        filed = source.get("file_date") or ""
        file_type = source.get("file_type") or ""
        cik = (source.get("ciks") or [""])[0]

        # URL to the filing index page
        adsh_clean = adsh.replace("-", "")
        url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik}&type={form}"
        ) if cik else ""

        # Sometimes a direct filing URL is available
        if source.get("adsh") and cik:
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{adsh_clean}/{adsh}-index.htm"
            )

        title = f"{company} — {form}"
        snippet = (source.get("display_names") and ", ".join(source["display_names"])) or ""
        snippet = f"{snippet} · filed {filed} · {file_type}".strip(" ·")

        out.append(Result(
            source="sec_edgar",
            title=title,
            url=url or f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}",
            snippet=snippet[:400],
            author=company,
            published=filed,
            domain="finance",
            engagement_label=form,
            extra={"form": form, "cik": cik, "accession": adsh},
        ))
    return out


register(SourceSpec(
    name="sec_edgar",
    domains=["finance"],
    tier="free",
    search=search,
    description="SEC EDGAR full-text search of 10-K/10-Q/8-K/S-1 filings",
))
