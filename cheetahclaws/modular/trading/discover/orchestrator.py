"""
discover/orchestrator.py — run multiple discovery scanners and merge results.

When the same ticker is flagged by multiple sources (e.g., insider
cluster + earnings beat + sector momentum), confidence is much higher.
The orchestrator aggregates per-symbol scores from all sources and
emits a final ranked list.
"""
from __future__ import annotations

from typing import Iterable

from .types import Discovery
from . import insider_cluster, earnings_beat, sector_rotation, momentum_quality


# Source weights when computing aggregate score. Tuned by perceived
# reliability — insider clusters are the strongest single signal,
# sector rotation is supporting evidence.
SOURCE_WEIGHTS: dict[str, float] = {
    "insider":          1.0,
    "earnings":         0.9,
    "momentum-quality": 0.7,
    "sector":           0.5,
    "anomaly":          0.6,
}


SCANNERS = {
    "insider":          insider_cluster.scan,
    "earnings":         earnings_beat.scan,
    "momentum-quality": momentum_quality.scan,
    "sector":           sector_rotation.scan,
}


def run(
    sources: Iterable[str] | None = None,
    universe: str | None = "sp100",
    symbols: Iterable[str] | None = None,
    top_n: int = 20,
    progress_cb=None,
) -> dict:
    """Run discovery across requested sources. Returns dict with hits + ranked.

    sources: subset of {"insider", "earnings", "momentum-quality", "sector"}
             None means run all four.
    """
    sources = list(sources) if sources else list(SCANNERS.keys())

    all_hits: list[Discovery] = []
    per_source: dict[str, list[Discovery]] = {}
    notes: list[str] = []

    for src in sources:
        if src not in SCANNERS:
            notes.append(f"Unknown source skipped: {src}")
            continue
        try:
            if src == "sector":
                hits = SCANNERS[src](progress_cb=progress_cb)
            else:
                hits = SCANNERS[src](universe=universe, symbols=symbols,
                                     progress_cb=progress_cb)
        except Exception as e:
            notes.append(f"Source {src} failed: {type(e).__name__}: {e}")
            hits = []
        per_source[src] = hits
        all_hits.extend(hits)

    # Merge by symbol — multi-source hits get aggregated score
    merged: dict[str, dict] = {}
    for h in all_hits:
        entry = merged.setdefault(h.symbol, {
            "symbol": h.symbol,
            "sources": [],
            "reasons": [],
            "details": {},
            "raw_score": 0.0,
        })
        entry["sources"].append(h.source)
        entry["reasons"].append(f"[{h.source}] {h.reason}")
        entry["details"][h.source] = h.details
        weight = SOURCE_WEIGHTS.get(h.source, 0.5)
        entry["raw_score"] += weight * h.score

    # Cross-source bonus: tickers flagged by ≥2 sources get a boost
    for sym, e in merged.items():
        n_sources = len(set(e["sources"]))
        e["n_sources"] = n_sources
        e["bonus"] = 0.5 if n_sources >= 2 else 0.0
        e["aggregate_score"] = e["raw_score"] + e["bonus"]

    ranked = sorted(merged.values(), key=lambda x: -x["aggregate_score"])

    return {
        "ranked": ranked[:top_n],
        "per_source": {src: [h.to_dict() for h in hits]
                       for src, hits in per_source.items()},
        "n_unique": len(merged),
        "n_total_hits": len(all_hits),
        "notes": notes,
    }


def render_report(result: dict, top: int = 15) -> str:
    """Markdown report of discovery results."""
    if not result["ranked"]:
        msg = "_No discoveries — try a different universe or relax thresholds._"
        if result.get("notes"):
            msg += "\n\n" + "\n".join(f"- {n}" for n in result["notes"])
        return msg

    lines = [f"# Discovery — {result['n_unique']} unique tickers, "
             f"{result['n_total_hits']} total hits"]
    lines.append("")
    lines.append("| # | Symbol | Sources | Score | Reasons |")
    lines.append("|---:|---|---|---:|---|")
    for i, e in enumerate(result["ranked"][:top], 1):
        srcs = " · ".join(set(e["sources"]))
        reason_short = "; ".join(e["reasons"])[:200]
        if len(reason_short) > 198:
            reason_short = reason_short[:197] + "…"
        lines.append(
            f"| {i} | **{e['symbol']}** | {srcs} | "
            f"{e['aggregate_score']:.2f} | {reason_short} |"
        )

    if result.get("notes"):
        lines.append("")
        lines.append("## Notes")
        for n in result["notes"]:
            lines.append(f"- {n}")

    lines.append("")
    lines.append("> Ranked by aggregate score across sources (insider 1.0, "
                 "earnings 0.9, momentum-quality 0.7, sector 0.5). Tickers "
                 "flagged by ≥2 sources get a +0.5 bonus.")
    return "\n".join(lines)
