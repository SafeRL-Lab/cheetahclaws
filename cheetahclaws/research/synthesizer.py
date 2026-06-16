"""Brief synthesizer — uses the active cheetahclaws model to distill results.

Takes the top N ranked results, groups by domain, and asks the model for:
  1. A 3-5 bullet TL;DR
  2. A cross-platform attention table
  3. Per-domain highlights with citations
  4. Contrarian / minority views worth flagging
  5. Open questions / gaps in coverage

Graceful fallback: if no model is available, returns a deterministic
non-LLM brief with the top results grouped by domain. The brief is
never empty as long as at least one source returned a result.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .types import Brief, Result

_MAX_RESULTS_FOR_LLM = 25
_MAX_SNIPPET_CHARS   = 300


def synthesize(brief: Brief, config: dict | None = None) -> str:
    cfg = config or {}
    model = cfg.get("model")
    if not model:
        return render_without_llm(brief)

    # Build a compact context for the LLM
    ctx_lines = [
        f"Topic: {brief.topic}",
        f"Domains routed to: {', '.join(brief.domains)}",
        f"Sources queried: {len(brief.statuses)} · OK: "
        f"{sum(1 for s in brief.statuses if s.ok)} · "
        f"skipped: {sum(1 for s in brief.statuses if s.skipped_reason)} · "
        f"failed: {sum(1 for s in brief.statuses if not s.ok and not s.skipped_reason)}",
        "",
        "# Top-ranked results (engagement-sorted):",
        "",
    ]
    for i, r in enumerate(brief.results[:_MAX_RESULTS_FOR_LLM], start=1):
        ctx_lines.append(f"[{i}] ({r.source}/{r.domain}) {r.title}")
        if r.engagement_label:
            ctx_lines.append(f"    engagement: {r.engagement_label}")
        if r.published:
            ctx_lines.append(f"    published: {r.published}")
        ctx_lines.append(f"    url: {r.url}")
        if r.snippet:
            ctx_lines.append(f"    snippet: {r.snippet[:_MAX_SNIPPET_CHARS]}")
        ctx_lines.append("")

    context = "\n".join(ctx_lines)

    system = (
        "You are a research analyst. You synthesize cross-source findings "
        "into a tight, scannable brief. You MUST cite sources inline as "
        "[N] where N is the bracket-number from the input. You NEVER invent "
        "results, numbers, URLs, or quotes. If coverage is thin or biased, "
        "you say so explicitly."
    )

    heat = format_heat_table(brief)
    # Entities — emitted as a block the model must copy verbatim (same pattern
    # as the heat table) so the structured data lands in the final brief even
    # if the model tries to be creative.
    entities_block = ""
    ents = getattr(brief, "_entities", None)
    if ents is not None:
        from .entities import render_entities_table
        ent_text = render_entities_table(ents)
        if ent_text:
            entities_block = (
                "\n\n## Top mentioned entities\n"
                "Copy the block below verbatim under this heading, then write "
                "2 sentences of analysis — which model / benchmark / org is "
                "dominating the discussion, and whether any notable player is "
                "conspicuously absent.\n\n"
                "ENTITIES BLOCK TO COPY VERBATIM:\n" + ent_text
            )

    user = f"""Below is ranked evidence on "{brief.topic}" from {len(brief.statuses)} sources.
Write a research brief with this exact structure:

## TL;DR
3-5 bullets. Each bullet must cite one or more sources inline using [N].

## Cross-platform attention
Copy the heat-table block below verbatim (do not rewrite, do not add rows).
Then write 2-3 sentences of analysis underneath: which platform has the
most signal on this topic, where the topic is notably absent, and what
that distribution implies (academic-heavy vs. social-heavy vs. news-heavy).

HEAT TABLE TO COPY VERBATIM:
{heat}
{entities_block}

## Key findings by domain
One section per domain that has material. For each bullet cite [N].

## Contrarian or minority views
2-3 bullets that run against the mainstream signal — only include if the
evidence actually shows them. If everything points one way, write
"No notable dissent in the pulled results." instead.

## Open questions / gaps
2-3 bullets on what the pulled evidence does NOT cover. Flag stale data
if everything is >30 days old, flag single-source claims as unverified.

Rules:
- Cite only what is in the input; never invent a reference.
- Quote numbers (star counts, upvotes, citations) verbatim when stating
  them — do not paraphrase "1,234 stars" as "over a thousand".
- Use Markdown. No preamble, no "Here is the brief:" — start at ## TL;DR.
- Target 500-800 words total.

---

{context}
"""

    # Stream through the provider
    try:
        from cheetahclaws.providers import stream, TextChunk, AssistantTurn
    except ImportError:
        return render_without_llm(brief)

    messages = [{"role": "user", "content": user}]
    out_parts: list[str] = []
    try:
        for ev in stream(
            model=model,
            system=system,
            messages=messages,
            tool_schemas=[],
            config={**cfg, "no_tools": True},
        ):
            if isinstance(ev, TextChunk):
                out_parts.append(ev.text)
            elif isinstance(ev, AssistantTurn):
                break
    except Exception as e:
        return render_without_llm(brief) + f"\n\n_(LLM synthesis error: {type(e).__name__}: {e})_"

    return "".join(out_parts).strip() or render_without_llm(brief)


def synthesize_comparison(
    topics: list[str],
    briefs: list[Brief],
    config: dict | None = None,
) -> str:
    """Build a side-by-side comparison brief for 2-3 topics.

    The prompt asks the model for: quick verdict, per-dimension comparison
    (academic coverage, community heat, engagement magnitudes, geography),
    shared themes, and unique strengths per topic. Always cites using
    `[A-N]` / `[B-N]` / `[C-N]` prefixed markers so readers can trace back.
    """
    cfg = config or {}
    model = cfg.get("model")
    if not model:
        return render_compare_fallback(topics, briefs)

    prefixes = ["A", "B", "C"][:len(topics)]

    # Build interleaved context: for each topic, list top-10 results with
    # a prefixed numbering scheme
    ctx_lines = [
        f"Comparing: {', '.join(f'{p}={t}' for p, t in zip(prefixes, topics))}",
        "",
    ]
    for pfx, t, brief in zip(prefixes, topics, briefs):
        ctx_lines.append(f"# {pfx}. {t}")
        ctx_lines.append(
            f"(routed: {', '.join(brief.domains)} · "
            f"{len(brief.results)} results · "
            f"{sum(1 for s in brief.statuses if s.ok)} sources OK)"
        )
        ctx_lines.append("")
        for i, r in enumerate(brief.results[:12], start=1):
            ctx_lines.append(f"[{pfx}{i}] ({r.source}/{r.domain}) {r.title}")
            if r.engagement_label:
                ctx_lines.append(f"      engagement: {r.engagement_label}")
            if r.published:
                ctx_lines.append(f"      published: {r.published}")
            ctx_lines.append(f"      url: {r.url}")
            if r.snippet:
                ctx_lines.append(f"      snippet: {r.snippet[:240]}")
        ctx_lines.append("")

    # Heat tables
    heat_lines = []
    for pfx, t, brief in zip(prefixes, topics, briefs):
        heat_lines.append(f"**{pfx}. {t}**\n\n{format_heat_table(brief)}")
    heat_md = "\n\n".join(heat_lines)

    system = (
        "You are a research analyst writing a comparative brief. You never "
        "invent results, numbers, urls, or quotes. You cite every claim as "
        "[A-N] / [B-N] / [C-N] matching the prefixed numbers in the input. "
        "You make cross-topic contrasts concrete (numbers, not adjectives)."
    )
    user = f"""Below is ranked evidence for {len(topics)} topics being compared.
Write a comparative brief with this EXACT structure:

## Verdict at a glance
One paragraph (3-4 sentences) stating the headline comparison.
Cite specifics: which topic pulls more academic papers, which one dominates
on social platforms, which one has bigger engagement peaks. Use [A-N]-style
citations throughout.

## Side-by-side heat
Copy the heat-table block below VERBATIM (do not rewrite):

{heat_md}

Then write a 2-3 sentence analysis pointing out the biggest gaps in the
tables — e.g. "A has no academic signal but dominates social; B has the
opposite distribution".

## Shared themes
2-3 bullets: topics both/all discuss, with citations from each.

## Unique strengths — {topics[0]} (A)
2-3 bullets on what A covers that the other(s) don't. [A-N] citations only.

## Unique strengths — {topics[1]} (B)
2-3 bullets on what B covers that the other(s) don't. [B-N] citations only.

{f'## Unique strengths — {topics[2]} (C){chr(10)}2-3 bullets on what C covers that the other(s) dont. [C-N] citations only.' if len(topics) == 3 else ''}

## Open questions / gaps
2-3 bullets on what would sharpen this comparison — which platforms you
WANT data from but dont have, which dimensions are undercovered.

Rules:
- Cite with [A-N] / [B-N] / [C-N] — NEVER plain [N].
- Quote numbers verbatim when stating engagement / citation / star counts.
- Start with `## Verdict at a glance`. No preamble.
- Target 500-900 words total.

---

{chr(10).join(ctx_lines)}
"""

    try:
        from cheetahclaws.providers import stream, TextChunk, AssistantTurn
    except ImportError:
        return render_compare_fallback(topics, briefs)

    messages = [{"role": "user", "content": user}]
    out_parts: list[str] = []
    try:
        for ev in stream(
            model=model, system=system, messages=messages,
            tool_schemas=[], config={**cfg, "no_tools": True},
        ):
            if isinstance(ev, TextChunk):
                out_parts.append(ev.text)
            elif isinstance(ev, AssistantTurn):
                break
    except Exception as e:
        return (render_compare_fallback(topics, briefs)
                + f"\n\n_(LLM comparison error: {type(e).__name__}: {e})_")

    return "".join(out_parts).strip() or render_compare_fallback(topics, briefs)


def render_compare_fallback(topics: list[str], briefs: list[Brief]) -> str:
    """Deterministic side-by-side rendering when no model is available."""
    out = ["## Verdict at a glance", ""]
    out.append(
        "_LLM synthesis unavailable — showing per-topic heat tables side-by-side._"
    )
    out.append("")
    prefixes = ["A", "B", "C"]
    for pfx, t, brief in zip(prefixes, topics, briefs):
        out.append(f"## {pfx}. {t}")
        out.append(
            f"_{len(brief.results)} results from "
            f"{sum(1 for s in brief.statuses if s.ok)} sources · routed to "
            f"{', '.join(brief.domains)}_"
        )
        out.append("")
        out.append(format_heat_table(brief))
        out.append("")
        ents = getattr(brief, "_entities", None)
        if ents:
            from .entities import render_entities_table
            et = render_entities_table(ents, title_prefix=f"{pfx}.")
            if et:
                out.append(et)
                out.append("")
        # Top 3 per domain
        grouped = brief.by_domain()
        for dom, rs in grouped.items():
            if not rs:
                continue
            out.append(f"### {pfx}. {dom.title()} top results")
            for i, r in enumerate(rs[:3], start=1):
                eng = f" — {r.engagement_label}" if r.engagement_label else ""
                out.append(f"- [{pfx}{i}] **{r.title}**{eng}")
                out.append(f"  {r.url}")
            out.append("")
    return "\n".join(out).strip()


def render_compare_brief(result: dict) -> str:
    """Assemble the final markdown for a compare() result."""
    topics = result["topics"]
    briefs = result["briefs"]
    comparison = result.get("comparison", "")
    duration = result.get("total_duration_ms", 0)

    out = [
        "# Comparative Research Brief",
        "",
        "_vs._ ".join(f"**{t}**" for t in topics),
        "",
        f"_{sum(len(b.results) for b in briefs)} combined results · "
        f"{duration} ms_",
        "",
    ]
    if comparison:
        out.append(comparison)
        out.append("")
    else:
        out.append(render_compare_fallback(topics, briefs))
        out.append("")

    # Citations across both/all topics
    out.append("## Citations")
    out.append("")
    for pfx, t, brief in zip(["A", "B", "C"], topics, briefs):
        if not brief.results:
            continue
        out.append(f"### {pfx}. {t}")
        for i, r in enumerate(brief.results[:12], start=1):
            eng = f" — {r.engagement_label}" if r.engagement_label else ""
            out.append(f"[{pfx}{i}] ({r.source}) {r.title}{eng}")
            out.append(f"    {r.url}")
        out.append("")
    return "\n".join(out).strip()


def render_without_llm(brief: Brief) -> str:
    """Deterministic fallback when no model is available."""
    lines = ["## TL;DR", "",
             f"_{len(brief.results)} results from {sum(1 for s in brief.statuses if s.ok)} sources — "
             f"LLM synthesis skipped, showing top results grouped by domain._",
             "",
             "## Cross-platform attention",
             "",
             format_heat_table(brief),
             ""]
    ents = getattr(brief, "_entities", None)
    if ents is not None:
        from .entities import render_entities_table
        et = render_entities_table(ents)
        if et:
            lines.append(et)
            lines.append("")
    grouped = brief.by_domain()
    for domain, rs in grouped.items():
        if not rs:
            continue
        lines.append(f"## {domain.title()}")
        lines.append("")
        for r in rs[:5]:
            eng = f" — {r.engagement_label}" if r.engagement_label else ""
            lines.append(f"- **{r.title}**{eng}")
            lines.append(f"  {r.url}")
            if r.snippet:
                lines.append(f"  {r.snippet[:200]}")
            lines.append("")
    return "\n".join(lines).strip()


def format_heat_table(brief: Brief) -> str:
    """Render a Markdown table showing per-platform attention on the topic.

    Columns: platform, result count, top engagement label, median age (days),
    domain tag. Platforms that were skipped or failed are still shown with
    a `—` in place of numbers so the reader sees where coverage is missing.
    """
    from .sources import SOURCES

    # Group results by source
    by_source: dict[str, list[Result]] = {}
    for r in brief.results:
        by_source.setdefault(r.source, []).append(r)

    status_by_name = {s.name: s for s in brief.statuses}
    all_source_names = sorted(set(list(status_by_name.keys()) + list(by_source.keys())))

    now = datetime.now(timezone.utc)
    rows: list[tuple[str, str, str, str, str]] = []

    for name in all_source_names:
        rs = by_source.get(name, [])
        st = status_by_name.get(name)
        spec = SOURCES.get(name)
        domain_tag = "/".join(spec.domains[:2]) if spec else "?"

        if rs:
            count = str(len(rs))
            top = max(rs, key=lambda r: r.engagement_raw)
            top_label = top.engagement_label or (
                f"{top.engagement_raw}" if top.engagement_raw else "—"
            )
            med_age = _median_age_days(rs, now)
            med_age_str = _fmt_age(med_age) if med_age is not None else "—"
        else:
            count = "0"
            if st and st.skipped_reason:
                top_label = f"skipped · {_abbreviate(st.skipped_reason, 40)}"
            elif st and st.error:
                top_label = f"failed · {_abbreviate(st.error, 40)}"
            else:
                top_label = "—"
            med_age_str = "—"

        rows.append((name, count, top_label, med_age_str, domain_tag))

    # Sort: active sources first (by result count desc), then skipped/failed
    rows.sort(key=lambda r: (-int(r[1]) if r[1].isdigit() else 0, r[0]))

    out = [
        "| Platform | Results | Top signal | Median age | Domain |",
        "|---|---|---|---|---|",
    ]
    for name, count, top_label, age, dom in rows:
        # Escape pipes in signal labels
        top_label = top_label.replace("|", "\\|")
        out.append(f"| {name} | {count} | {top_label} | {age} | {dom} |")
    return "\n".join(out)


def _median_age_days(results: list[Result], now: datetime) -> float | None:
    from .ranker import _parse_date
    ages: list[float] = []
    for r in results:
        if not r.published:
            continue
        dt = _parse_date(r.published)
        if dt is None:
            continue
        ages.append((now - dt).total_seconds() / 86400.0)
    if not ages:
        return None
    ages.sort()
    mid = len(ages) // 2
    if len(ages) % 2 == 1:
        return ages[mid]
    return (ages[mid - 1] + ages[mid]) / 2


def _fmt_age(days: float) -> str:
    if days < 1:
        hours = max(1, int(days * 24))
        return f"{hours}h"
    if days < 30:
        return f"{int(round(days))}d"
    if days < 365:
        return f"{int(round(days / 30))}mo"
    return f"{days / 365:.1f}y"


def _abbreviate(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n - 1] + "…"


_SPARK_BARS = "▁▂▃▄▅▆▇█"


def format_publication_trend(brief: Brief, buckets: int = 12) -> str:
    """Render a per-month bar chart of publication frequency.

    Uses ALL results with a parseable `published` date. Buckets are the
    last N months relative to now (or, if the brief's time range is
    bounded, the range's last N months).
    """
    from datetime import datetime, timezone
    from .ranker import _parse_date

    dated: list[datetime] = []
    for r in brief.results:
        dt = _parse_date(r.published) if r.published else None
        if dt:
            dated.append(dt)
    if not dated:
        return ""

    now = datetime.now(timezone.utc)
    # Build (year, month) buckets for the last N months
    bucket_keys: list[tuple[int, int]] = []
    y, m = now.year, now.month
    for _ in range(buckets):
        bucket_keys.append((y, m))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    bucket_keys.reverse()

    counts: dict[tuple[int, int], int] = {k: 0 for k in bucket_keys}
    for dt in dated:
        k = (dt.year, dt.month)
        if k in counts:
            counts[k] += 1

    max_count = max(counts.values()) if counts.values() else 0
    if max_count == 0:
        return ""

    lines = [f"## Publication trend — last {buckets} months",
             "",
             "```",
             f"Results across {len(dated)} dated items · peak month: {max_count}",
             ""]
    for (yr, mo) in bucket_keys:
        c = counts[(yr, mo)]
        bar_width = int((c / max_count) * 20)
        bar = "█" * bar_width if bar_width > 0 else ""
        lines.append(f"  {yr}-{mo:02d}  {bar:20s} {c:3d}")
    lines.append("```")
    return "\n".join(lines)


def format_publication_sparkline(brief: Brief, buckets: int = 24) -> str:
    """Compact single-line sparkline across N months (for the brief header)."""
    from datetime import datetime, timezone
    from .ranker import _parse_date

    dated: list[datetime] = []
    for r in brief.results:
        dt = _parse_date(r.published) if r.published else None
        if dt:
            dated.append(dt)
    if not dated:
        return ""

    now = datetime.now(timezone.utc)
    bucket_keys: list[tuple[int, int]] = []
    y, m = now.year, now.month
    for _ in range(buckets):
        bucket_keys.append((y, m))
        m -= 1
        if m == 0:
            m = 12; y -= 1
    bucket_keys.reverse()
    counts = [0] * buckets
    for dt in dated:
        k = (dt.year, dt.month)
        if k in bucket_keys:
            counts[bucket_keys.index(k)] += 1
    mx = max(counts) or 1
    spark = "".join(
        _SPARK_BARS[min(len(_SPARK_BARS) - 1,
                        int((c / mx) * (len(_SPARK_BARS) - 1)))]
        for c in counts
    )
    return f"{spark}  ({buckets}mo window · peak {mx})"


def render_citations(brief: Brief) -> str:
    """Emit a numbered citation list matching the [N] markers the LLM uses."""
    out = ["## Citations", ""]
    for i, r in enumerate(brief.results[:_MAX_RESULTS_FOR_LLM], start=1):
        eng = f" — {r.engagement_label}" if r.engagement_label else ""
        out.append(f"[{i}] ({r.source}) {r.title}{eng}")
        out.append(f"    {r.url}")
    return "\n".join(out)
