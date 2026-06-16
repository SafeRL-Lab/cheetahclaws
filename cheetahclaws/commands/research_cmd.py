"""`/research` and `/reports` slash commands.

Research usage:
    /research <topic>
    /research --domain academic "transformer inference efficiency"
    /research --sources arxiv,github "vLLM"
    /research --range 30d "latest AI reasoning benchmarks"
    /research --since 2024-01-01 --until 2024-06-30 "kubernetes CVEs"
    /research --citations "diffusion models"          # +notable citer analysis
    /research --citation-threshold 50000 "RLHF"
    /research --save-as ~/reports/my.md "topic"
    /research --no-cache --limit 30 "topic"
    /research list-sources

Reports usage:
    /reports              — list recent saved reports
    /reports list         — same
    /reports open 3       — print saved report #3
    /reports open 2024-04-20_143015-nvidia-earnings  — open by stem
    /reports delete 3     — delete report #3
    /reports path 3       — print the markdown file path
"""
from __future__ import annotations

import shlex
import time

from cheetahclaws.ui.render import clr, info, ok, warn, err

_VALID_DOMAINS = {"academic", "tech", "finance", "news", "social", "web"}


def cmd_research(args: str, state, config) -> bool:
    from cheetahclaws.research import research, build_time_range, compare
    from cheetahclaws.research.citations import render_notable_section
    from cheetahclaws.research.entities import render_entities_table
    from cheetahclaws.research.sources import SOURCES
    from cheetahclaws.research.synthesizer import (
        format_heat_table, format_publication_trend,
        format_publication_sparkline, render_citations,
        render_compare_brief,
    )
    from cheetahclaws.research import reports as _reports

    a = args.strip()
    if not a:
        info("Usage: /research <topic> [--domain D] [--sources s1,s2]")
        info("                   [--limit N] [--range 30d|1y] [--since YYYY-MM-DD] [--until YYYY-MM-DD]")
        info("                   [--expand N] [--citations] [--citation-threshold N]")
        info("                   [--save-as PATH] [--no-cache] [--no-save] [--no-synth]")
        info("       /research compare \"topic A\" vs \"topic B\" [vs \"topic C\"] [--range 30d] [--limit N]")
        info("       /research list-sources")
        return True

    if a == "list-sources":
        _list_sources()
        return True

    # Compare subcommand: `/research compare "A" vs "B" [vs "C"] [flags]`
    if a.startswith("compare ") or a.startswith("compare:"):
        return _cmd_compare(a[len("compare"):].lstrip(": "),
                            state, config, compare, render_compare_brief,
                            _reports)

    try:
        tokens = shlex.split(a)
    except ValueError as e:
        err(f"Parse error: {e}")
        return True

    # Parse flags
    domains: list[str] | None = None
    sources: list[str] | None = None
    limit = 15
    use_cache = True
    synthesize = True
    time_range_token: str | None = None
    since_str: str | None = None
    until_str: str | None = None
    analyze_citations = False
    citation_threshold = 10000
    expand = 0
    save_as: str | None = None
    auto_save = True
    topic_parts: list[str] = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("--domain", "--domains") and i + 1 < len(tokens):
            parts = [p.strip() for p in tokens[i + 1].split(",") if p.strip()]
            bad = [p for p in parts if p not in _VALID_DOMAINS]
            if bad:
                err(f"Unknown domain(s): {', '.join(bad)}. "
                    f"Valid: {', '.join(sorted(_VALID_DOMAINS))}")
                return True
            domains = parts
            i += 2
        elif tok in ("--source", "--sources") and i + 1 < len(tokens):
            sources = [p.strip() for p in tokens[i + 1].split(",") if p.strip()]
            bad = [p for p in sources if p not in SOURCES]
            if bad:
                err(f"Unknown source(s): {', '.join(bad)}. "
                    f"Run '/research list-sources' to see available sources.")
                return True
            i += 2
        elif tok == "--limit" and i + 1 < len(tokens):
            try:
                limit = max(1, min(int(tokens[i + 1]), 50))
            except ValueError:
                err(f"Invalid --limit value: {tokens[i + 1]}")
                return True
            i += 2
        elif tok in ("--range", "--time-range") and i + 1 < len(tokens):
            time_range_token = tokens[i + 1]
            i += 2
        elif tok == "--since" and i + 1 < len(tokens):
            since_str = tokens[i + 1]
            i += 2
        elif tok == "--until" and i + 1 < len(tokens):
            until_str = tokens[i + 1]
            i += 2
        elif tok in ("--citations", "--analyze-citations"):
            analyze_citations = True
            i += 1
        elif tok == "--expand":
            # Accept `--expand` (default 4) or `--expand N`
            if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                expand = max(2, min(int(tokens[i + 1]), 6))
                i += 2
            else:
                expand = 4
                i += 1
        elif tok == "--citation-threshold" and i + 1 < len(tokens):
            try:
                citation_threshold = max(1, int(tokens[i + 1]))
            except ValueError:
                err(f"Invalid --citation-threshold: {tokens[i + 1]}")
                return True
            i += 2
        elif tok == "--save-as" and i + 1 < len(tokens):
            save_as = tokens[i + 1]
            i += 2
        elif tok == "--no-cache":
            use_cache = False
            i += 1
        elif tok == "--no-save":
            auto_save = False
            i += 1
        elif tok == "--no-synth" or tok == "--no-synthesize":
            synthesize = False
            i += 1
        elif tok.startswith("--"):
            err(f"Unknown flag: {tok}")
            return True
        else:
            topic_parts.append(tok)
            i += 1

    topic = " ".join(topic_parts).strip()
    if not topic:
        err("No topic given.")
        return True

    # Build time range
    try:
        tr = build_time_range(range_token=time_range_token,
                              since=since_str, until=until_str)
    except ValueError as e:
        err(f"Bad time range: {e}")
        return True

    # Progress UI
    started: dict[str, float] = {}

    def progress(source_name: str, status: str) -> None:
        if status == "start":
            started[source_name] = time.time()
            info(clr(f"  → querying {source_name}…", "dim"))
        elif status == "done":
            dt = int((time.time() - started.get(source_name, time.time())) * 1000)
            ok(clr(f"  ✓ {source_name} ({dt} ms)", "dim"))
        elif status == "skipped":
            info(clr(f"  ∘ {source_name} skipped", "dim"))
        elif status == "error":
            warn(clr(f"  ✗ {source_name} failed", "yellow"))

    header = f"Researching: {topic}"
    if tr.label:
        header += clr(f"  (range: {tr.label})", "dim")
    if analyze_citations:
        header += clr(f"  +citation analysis (≥{citation_threshold:,})", "dim")
    if expand:
        header += clr(f"  +expand×{expand}", "dim")
    info(clr(header, "cyan"))

    t0 = time.time()
    try:
        brief = research(
            topic=topic,
            domains=domains,
            sources=sources,
            limit=limit,
            use_cache=use_cache,
            synthesize=synthesize,
            time_range=tr,
            analyze_citations=analyze_citations,
            citation_threshold=citation_threshold,
            expand=expand,
            config=config,
            progress_cb=progress,
        )
    except Exception as e:
        err(f"Research failed: {type(e).__name__}: {e}")
        return True

    elapsed = time.time() - t0
    notable = getattr(brief, "_notable_citers", []) or []

    # Render
    print()
    print(f"# Research Brief: {brief.topic}")
    print()
    header_bits = [
        f"{len(brief.results)} results from "
        f"{sum(1 for s in brief.statuses if s.ok)} sources",
        f"{elapsed:.1f}s",
        f"{brief.cache_hits} cached",
    ]
    if tr.label:
        header_bits.insert(0, f"Range: **{tr.label}**")
    header_bits.insert(0, f"Routed to {', '.join(brief.domains)}")
    print("_" + " · ".join(header_bits) + "_")
    spark = format_publication_sparkline(brief)
    if spark:
        print()
        print(f"`{spark}`")
    print()

    if brief.synthesis:
        print(brief.synthesis)
        print()
    else:
        print("## Cross-platform attention")
        print()
        print(format_heat_table(brief))
        print()

    ents = getattr(brief, "_entities", None)
    if ents is not None and "Top mentioned entities" not in (brief.synthesis or ""):
        et = render_entities_table(ents)
        if et:
            print(et)
            print()

    trend = format_publication_trend(brief, buckets=12)
    if trend and "Publication trend" not in (brief.synthesis or ""):
        print(trend)
        print()

    if notable:
        print(render_notable_section(notable, citation_threshold))
        print()

    if brief.results:
        print(render_citations(brief))
        print()

    failed = [s for s in brief.statuses if not s.ok]
    if failed:
        print("## Missed / skipped sources")
        print()
        for s in failed:
            reason = s.skipped_reason or s.error or "unknown"
            print(f"- **{s.name}** — {reason}")
        print()

    # Save
    if auto_save:
        try:
            # Reassemble full rendered markdown for the saved file
            from io import StringIO
            buf = StringIO()
            buf.write(f"# Research Brief: {brief.topic}\n\n")
            buf.write("_" + " · ".join(header_bits) + "_\n\n")
            if spark:
                buf.write(f"`{spark}`\n\n")
            if brief.synthesis:
                buf.write(brief.synthesis + "\n\n")
            else:
                buf.write("## Cross-platform attention\n\n" + format_heat_table(brief) + "\n\n")
            if ents is not None and "Top mentioned entities" not in (brief.synthesis or ""):
                et = render_entities_table(ents)
                if et:
                    buf.write(et + "\n\n")
            if trend:
                buf.write(trend + "\n\n")
            if notable:
                buf.write(render_notable_section(notable, citation_threshold) + "\n\n")
            if brief.results:
                buf.write(render_citations(brief) + "\n\n")
            if failed:
                buf.write("## Missed / skipped sources\n\n")
                for s in failed:
                    buf.write(f"- **{s.name}** — {s.skipped_reason or s.error}\n")
            path = _reports.save(brief, buf.getvalue(),
                                 notable=notable, also_save_as=save_as)
            info(clr(f"Saved: {path}", "dim"))
            if save_as:
                info(clr(f"  also → {save_as}", "dim"))
        except OSError as e:
            warn(f"Save failed: {e}")
    return True


def _cmd_compare(args: str, state, config, compare_fn, render_fn,
                 _reports_mod) -> bool:
    """Parse `/research compare "A" vs "B" [vs "C"] [--range 30d] [--limit N]`."""
    try:
        tokens = shlex.split(args)
    except ValueError as e:
        err(f"Parse error: {e}")
        return True

    # Split on `vs` separator into 2-3 topics, with optional flags after
    topics: list[str] = []
    flag_tokens: list[str] = []
    current: list[str] = []
    seen_first_flag = False

    for tok in tokens:
        if seen_first_flag:
            flag_tokens.append(tok)
            continue
        if tok.startswith("--"):
            if current:
                topics.append(" ".join(current))
                current = []
            flag_tokens.append(tok)
            seen_first_flag = True
        elif tok.lower() == "vs":
            if current:
                topics.append(" ".join(current))
                current = []
        else:
            current.append(tok)
    if current:
        topics.append(" ".join(current))

    topics = [t.strip() for t in topics if t.strip()]
    if len(topics) < 2:
        err('Usage: /research compare "topic A" vs "topic B" [vs "topic C"]')
        return True
    if len(topics) > 3:
        err("Compare supports at most 3 topics.")
        return True

    # Parse flags subset — only --range/--since/--until/--limit/--sources/--domain
    # /--no-cache/--save-as. --expand / --citations don't apply in compare mode.
    domains: list[str] | None = None
    sources: list[str] | None = None
    limit = 10
    use_cache = True
    time_range_token: str | None = None
    since_str: str | None = None
    until_str: str | None = None
    save_as: str | None = None
    auto_save = True

    i = 0
    while i < len(flag_tokens):
        tok = flag_tokens[i]
        if tok in ("--range", "--time-range") and i + 1 < len(flag_tokens):
            time_range_token = flag_tokens[i + 1]; i += 2
        elif tok == "--since" and i + 1 < len(flag_tokens):
            since_str = flag_tokens[i + 1]; i += 2
        elif tok == "--until" and i + 1 < len(flag_tokens):
            until_str = flag_tokens[i + 1]; i += 2
        elif tok == "--limit" and i + 1 < len(flag_tokens):
            try:
                limit = max(3, min(int(flag_tokens[i + 1]), 30))
            except ValueError:
                pass
            i += 2
        elif tok in ("--domain", "--domains") and i + 1 < len(flag_tokens):
            domains = [p.strip() for p in flag_tokens[i + 1].split(",") if p.strip()]
            i += 2
        elif tok in ("--source", "--sources") and i + 1 < len(flag_tokens):
            sources = [p.strip() for p in flag_tokens[i + 1].split(",") if p.strip()]
            i += 2
        elif tok == "--save-as" and i + 1 < len(flag_tokens):
            save_as = flag_tokens[i + 1]; i += 2
        elif tok == "--no-cache":
            use_cache = False; i += 1
        elif tok == "--no-save":
            auto_save = False; i += 1
        else:
            err(f"Unknown flag in compare: {tok}")
            return True

    from cheetahclaws.research import build_time_range
    try:
        tr = build_time_range(range_token=time_range_token,
                              since=since_str, until=until_str)
    except ValueError as e:
        err(f"Bad time range: {e}")
        return True

    # Progress
    started: dict[str, float] = {}
    def progress(name: str, status: str) -> None:
        if status == "start":
            started[name] = time.time()
            info(clr(f"  → {name}", "dim"))
        elif status == "done":
            dt = int((time.time() - started.get(name, time.time())) * 1000)
            ok(clr(f"  ✓ {name} ({dt} ms)", "dim"))
        elif status == "skipped":
            info(clr(f"  ∘ {name} skipped", "dim"))
        elif status == "error":
            warn(clr(f"  ✗ {name} failed", "yellow"))

    header = "Comparing: " + clr(" vs ".join(f'"{t}"' for t in topics), "cyan")
    if tr.label:
        header += clr(f"  (range: {tr.label})", "dim")
    info(header)

    t0 = time.time()
    try:
        result = compare_fn(
            topic_a=topics[0], topic_b=topics[1],
            topic_c=topics[2] if len(topics) == 3 else None,
            domains=domains, sources=sources,
            limit=limit, use_cache=use_cache,
            time_range=tr, config=config, progress_cb=progress,
        )
    except Exception as e:
        err(f"Compare failed: {type(e).__name__}: {e}")
        return True

    elapsed = time.time() - t0
    rendered = render_fn(result)

    print()
    print(rendered)
    print()
    info(clr(f"  · {elapsed:.1f}s total · "
             f"{sum(len(b.results) for b in result['briefs'])} combined results",
             "dim"))

    # Save
    if auto_save:
        try:
            # Use first topic as the saved report's topic slug; sidecar stores all
            from cheetahclaws.research.types import Brief, SourceStatus
            fake_brief = Brief(
                topic=" vs ".join(topics),
                domains=list({d for b in result["briefs"] for d in b.domains}),
                results=[r for b in result["briefs"] for r in b.results],
                statuses=[s for b in result["briefs"] for s in b.statuses],
                synthesis=result.get("comparison", ""),
                total_duration_ms=result.get("total_duration_ms", 0),
            )
            path = _reports_mod.save(fake_brief, rendered,
                                     notable=[], also_save_as=save_as)
            info(clr(f"Saved: {path}", "dim"))
            if save_as:
                info(clr(f"  also → {save_as}", "dim"))
        except OSError as e:
            warn(f"Save failed: {e}")
    return True


def cmd_reports(args: str, state, config) -> bool:
    from cheetahclaws.research import reports as _reports
    a = (args or "").strip()

    if not a or a in ("list", "ls"):
        reports = _reports.list_reports(limit=50)
        if not reports:
            info("No saved reports yet. Run /research <topic> to create one.")
            return True
        info(clr(f"Saved research reports ({len(reports)}):", "cyan"))
        for r in reports:
            info(f"  {r['id']:3d}  {r['created_at'][:19]}  "
                 f"[{r['results_count']:3d} results · {r['sources_ok']} src · "
                 f"{r['size_kb']:.1f}KB]  {r['topic'][:60]}")
        info(clr("\n  Use: /reports open <id>  to view", "dim"))
        return True

    parts = a.split(None, 1)
    cmd = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if cmd == "open" or cmd == "view":
        if not rest:
            err("Usage: /reports open <id>  or  /reports open <stem>")
            return True
        info_r = _find(rest)
        if not info_r:
            err(f"No report matching '{rest}'.")
            return True
        md = _reports.read_markdown(stem=info_r["stem"])
        if md is None:
            err(f"Could not read markdown file for {info_r['stem']}.")
            return True
        print()
        print(md)
        print()
        return True

    if cmd == "delete" or cmd == "rm":
        if not rest:
            err("Usage: /reports delete <id>")
            return True
        try:
            rid = int(rest)
        except ValueError:
            err(f"Invalid id: {rest}")
            return True
        if _reports.delete(rid):
            info(f"Deleted report #{rid}.")
        else:
            err(f"No report #{rid}.")
        return True

    if cmd == "path":
        if not rest:
            err("Usage: /reports path <id>")
            return True
        info_r = _find(rest)
        if not info_r:
            err(f"No report matching '{rest}'.")
            return True
        print(info_r["md_path"])
        return True

    err(f"Unknown /reports subcommand: {cmd}")
    info("Usage: /reports [list|open <id>|delete <id>|path <id>]")
    return True


def _find(ref: str):
    from cheetahclaws.research import reports as _reports
    try:
        rid = int(ref)
        return _reports.get_by_id(rid)
    except ValueError:
        return _reports.get_by_stem(ref)


def _list_sources() -> None:
    from cheetahclaws.research.sources import SOURCES
    info("Registered sources:")
    free = [s for s in SOURCES.values() if s.tier == "free"]
    optional = [s for s in SOURCES.values() if s.tier == "optional"]

    def _row(spec):
        dom = ", ".join(spec.domains)
        env = f" [needs {', '.join(spec.requires_env)}]" if spec.requires_env else ""
        info(f"  {spec.name:18s} · {dom:30s}{env}")
        info(clr(f"    {spec.description}", "dim"))

    info(clr("\n  Free (always queried):", "cyan"))
    for s in sorted(free, key=lambda x: x.name):
        _row(s)
    if optional:
        info(clr("\n  Optional (need API key / cookie / package):", "cyan"))
        for s in sorted(optional, key=lambda x: x.name):
            _row(s)
