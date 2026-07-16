"""Research tool — agent-facing entry point for multi-source web research."""
from __future__ import annotations


def _research(
    topic: str,
    domains: list[str] | None = None,
    sources: list[str] | None = None,
    limit: int = 15,
    synthesize: bool = True,
    use_cache: bool = True,
    time_range: str | None = None,
    since: str | None = None,
    until: str | None = None,
    analyze_citations: bool = False,
    citation_threshold: int = 10000,
    expand: int = 0,
    save_as: str | None = None,
    auto_save: bool = True,
    config: dict | None = None,
) -> str:
    """Run the research pipeline and return a markdown-formatted brief.

    The agent calls this when it needs current, multi-source information on
    any topic — academic papers, tech discussion, finance signals, news.
    Supports time-range filtering (`time_range="30d"`, or absolute
    `since`/`until` ISO dates), notable-citer analysis for academic
    topics, and auto-saving to `~/.cheetahclaws/research_reports/`.

    Returns a ready-to-consume markdown brief with TL;DR, cross-platform
    attention table, publication trend, per-domain findings, notable
    citing authors (if analyzed), and numbered citations. Failures on
    individual sources surface in a "Missed" footer.
    """
    from cheetahclaws.research import research, build_time_range
    from cheetahclaws.research.citations import render_notable_section
    from cheetahclaws.research.entities import render_entities_table
    from cheetahclaws.research.synthesizer import (
        format_heat_table, format_publication_trend,
        format_publication_sparkline, render_citations,
    )
    from cheetahclaws.research import reports as _reports

    try:
        tr = build_time_range(range_token=time_range, since=since, until=until)
    except ValueError as e:
        return f"Error: {e}"

    try:
        brief = research(
            topic=topic,
            domains=domains,
            sources=sources,
            limit=int(limit),
            use_cache=bool(use_cache),
            synthesize=bool(synthesize),
            time_range=tr,
            analyze_citations=bool(analyze_citations),
            citation_threshold=int(citation_threshold),
            expand=int(expand),
            config=config or {},
        )
    except Exception as e:
        return f"Error running research: {type(e).__name__}: {e}"

    if not brief.results:
        ok = [s.name for s in brief.statuses if s.ok]
        failed = [f"{s.name}: {s.error or s.skipped_reason}"
                  for s in brief.statuses if not s.ok]
        return (
            f"No results for '{topic}'"
            f"{' (' + tr.label + ')' if tr.label else ''}.\n"
            f"Queried: {', '.join(ok) or '(none succeeded)'}\n"
            f"Issues: {'; '.join(failed) if failed else '(none)'}"
        )

    spark = format_publication_sparkline(brief)
    out: list[str] = []
    out.append(f"# Research Brief: {brief.topic}")
    out.append("")
    header_bits = [
        f"Routed to {', '.join(brief.domains)}",
        f"{len(brief.results)} results from "
        f"{sum(1 for s in brief.statuses if s.ok)} sources",
        f"{brief.total_duration_ms} ms",
        f"{brief.cache_hits} cached",
    ]
    if tr.label:
        header_bits.insert(0, f"Range: **{tr.label}**")
    out.append("_" + " · ".join(header_bits) + "_")
    if spark:
        out.append("")
        out.append(f"`{spark}`")
    out.append("")

    if brief.synthesis:
        out.append(brief.synthesis)
    else:
        out.append("## Cross-platform attention")
        out.append("")
        out.append(format_heat_table(brief))
        ents = getattr(brief, "_entities", None)
        if ents is not None:
            et = render_entities_table(ents)
            if et:
                out.append("")
                out.append(et)

    # Publication trend (if dated results exist)
    trend = format_publication_trend(brief, buckets=12)
    if trend and "## Publication trend" not in (brief.synthesis or ""):
        out.append("")
        out.append(trend)

    # Notable citing authors (if citation analysis ran)
    notable = getattr(brief, "_notable_citers", []) or []
    if notable:
        out.append("")
        out.append(render_notable_section(notable, citation_threshold))

    out.append("")
    out.append(render_citations(brief))

    missed = [
        (s.name,
         s.skipped_reason or s.error or "unknown")
        for s in brief.statuses if not s.ok
    ]
    if missed:
        out.append("")
        out.append("## Missed / skipped sources")
        out.append("")
        for name, reason in missed:
            out.append(f"- **{name}** — {reason}")

    rendered = "\n".join(out)

    # Auto-save
    if auto_save:
        try:
            path = _reports.save(brief, rendered, notable=notable, also_save_as=save_as)
            rendered += f"\n\n---\n_Saved: {path}_"
            if save_as:
                rendered += f" _· also → {save_as}_"
        except OSError:
            pass

    return rendered
