"""Polymarket — prediction market odds. Real USD volume = hard signal.

Public Gamma API (markets + events). No auth.
"""
from __future__ import annotations

from ..http import get
from ..types import Result
from . import SourceSpec, register

_ENDPOINT = "https://gamma-api.polymarket.com/markets"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    # Gamma `/markets` doesn't offer full-text search, but accepts a
    # q-like parameter via `active=true&order=volume24hr&search=<q>` on
    # some deployments. Fall back to pulling recent active markets and
    # filtering client-side by title substring.
    out: list[Result] = []

    # Pull active markets sorted by 24h volume; client-side filter for the term
    try:
        data = get(_ENDPOINT, params={
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
            "limit": 200,
        })
    except Exception:
        return out

    q = query.lower()
    terms = [t for t in q.split() if len(t) > 2]

    candidates = data if isinstance(data, list) else (data.get("markets") or [])
    for m in candidates:
        title = (m.get("question") or m.get("title") or "").strip()
        if not title:
            continue
        hay = title.lower() + " " + (m.get("description") or "").lower()
        if not any(t in hay for t in terms):
            continue

        slug = m.get("slug") or ""
        url = f"https://polymarket.com/market/{slug}" if slug else (m.get("url") or "")
        if not url:
            continue

        volume = float(m.get("volume") or m.get("volumeNum") or 0.0)
        liquidity = float(m.get("liquidity") or m.get("liquidityNum") or 0.0)

        # Outcomes + prices → extract the YES probability as a human-readable label
        outcomes = m.get("outcomes")
        outcome_prices = m.get("outcomePrices")
        odds_str = ""
        try:
            if isinstance(outcomes, str):
                import json as _j
                outcomes = _j.loads(outcomes)
            if isinstance(outcome_prices, str):
                import json as _j
                outcome_prices = _j.loads(outcome_prices)
            if outcomes and outcome_prices and len(outcomes) == len(outcome_prices):
                parts = []
                for name, price in zip(outcomes, outcome_prices):
                    try:
                        parts.append(f"{name} {float(price) * 100:.0f}%")
                    except (TypeError, ValueError):
                        continue
                odds_str = " · ".join(parts)
        except Exception:
            pass

        description = (m.get("description") or "")[:400]

        out.append(Result(
            source="polymarket",
            title=title,
            url=url,
            snippet=f"Odds: {odds_str}. {description}".strip(". "),
            author="",
            published=m.get("startDate") or m.get("createdAt") or "",
            engagement_raw=int(volume),
            engagement_label=f"${volume:,.0f} volume · ${liquidity:,.0f} liquidity",
            domain="finance",
            extra={"slug": slug, "volume_usd": volume, "liquidity_usd": liquidity,
                   "odds": odds_str},
        ))
        if len(out) >= limit:
            break

    return out


register(SourceSpec(
    name="polymarket",
    domains=["finance"],
    tier="free",
    search=search,
    description="Polymarket prediction market odds + USD volume",
))
