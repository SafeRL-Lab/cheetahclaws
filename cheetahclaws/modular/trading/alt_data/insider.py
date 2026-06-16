"""
insider.py — fetch SEC EDGAR Form 4 (insider transactions) for a ticker.

Why: when officers / 10%-owners buy or sell their own company stock,
that's a strong signal — they have non-public information. Public,
free, and not already priced in by quant funds at the LLM-analysis
timescale we operate on.

Implementation notes:
  - SEC requires a User-Agent identifying the requester (their stated
    fair-use policy). We send a generic "cheetahclaws/<email>" string.
  - Rate-limited to 10 requests/sec by SEC; we add a small sleep buffer.
  - Output is normalised to a few fields per filing rather than
    surfacing the full XBRL — the LLM only needs net direction + size.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Any

# SEC asks for a User-Agent header identifying the requester.
_USER_AGENT = "cheetahclaws-trading-research research@cheetahclaws.invalid"
_SEC_BASE = "https://www.sec.gov"
_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
_REQ_TIMEOUT = 8


def _http_get(url: str) -> bytes | None:
    """SEC-friendly GET. Returns bytes or None on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_REQ_TIMEOUT) as r:
            return r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def _ticker_to_cik(ticker: str) -> str | None:
    """Resolve ticker symbol to SEC CIK (10-digit zero-padded)."""
    raw = _http_get(f"{_SEC_BASE}/files/company_tickers.json")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    target = ticker.upper().strip()
    for entry in data.values():
        if entry.get("ticker", "").upper() == target:
            return str(entry["cik_str"]).zfill(10)
    return None


def fetch_recent_insider_filings(ticker: str, days: int = 90, max_filings: int = 20) -> list[dict[str, Any]]:
    """Return a list of recent Form 4 filings for `ticker`.

    Each entry has:
        {accession, filed_date, form, primary_doc_url}

    Empty list on any failure (SEC down, ticker not found, no Form 4 in window).
    """
    cik = _ticker_to_cik(ticker)
    if not cik:
        return []

    raw = _http_get(f"{_SUBMISSIONS_BASE}/CIK{cik}.json")
    time.sleep(0.12)  # SEC fair-use buffer
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms       = recent.get("form", [])
    accessions  = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    if not (forms and accessions and filing_dates):
        return []

    cutoff = (datetime.utcnow() - timedelta(days=days)).date()
    out = []
    for form, acc, dt, doc in zip(forms, accessions, filing_dates, primary_docs):
        if form not in ("4", "4/A"):
            continue
        try:
            filed = datetime.strptime(dt, "%Y-%m-%d").date()
        except ValueError:
            continue
        if filed < cutoff:
            continue
        acc_clean = acc.replace("-", "")
        out.append({
            "accession":   acc,
            "filed_date":  dt,
            "form":        form,
            "primary_doc_url": (
                f"{_SEC_BASE}/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
            ),
        })
        if len(out) >= max_filings:
            break
    return out


def render_insider_summary(ticker: str, days: int = 90) -> str:
    """Markdown block for the analyze prompt. Empty string when no data."""
    filings = fetch_recent_insider_filings(ticker, days=days)
    if not filings:
        return ""

    lines = [f"## Insider Activity ({ticker}, last {days} days)"]
    lines.append(f"- {len(filings)} Form 4 filing(s) by officers / 10%-holders")
    # Surface up to 5 most recent so the LLM can read them as URLs.
    for f in filings[:5]:
        lines.append(f"  - {f['filed_date']} ({f['form']}): {f['primary_doc_url']}")
    if len(filings) > 5:
        lines.append(f"  - ... and {len(filings) - 5} more")
    lines.append("")
    lines.append("**How to use**: cluster of buys by multiple officers within a "
                 "short window = strong signal. Sales alone are noise (taxes, "
                 "diversification). Combined buys + sells in the same week = "
                 "insider disagreement.")
    return "\n".join(lines)
