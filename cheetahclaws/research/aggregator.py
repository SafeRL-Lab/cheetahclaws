"""research() — the main entry point.

Fans out to the selected sources in parallel, dedupes + ranks results,
and optionally synthesizes a brief using the agent's current model.
"""
from __future__ import annotations

import concurrent.futures as _cf
import time
from typing import Iterable

from . import cache as _cache
from . import classifier as _cls
from . import ranker as _rank
from . import synthesizer as _syn
from .sources import SOURCES, SourceSkipped, sources_for_domains
from .time_range import TimeRange
from .types import Brief, Domain, Result, SourceStatus

DEFAULT_PER_SOURCE_LIMIT = 15
DEFAULT_SOURCE_TIMEOUT = 12.0


def research(
    topic: str,
    domains: list[Domain] | None = None,
    sources: list[str] | None = None,
    limit: int = DEFAULT_PER_SOURCE_LIMIT,
    use_cache: bool = True,
    synthesize: bool = True,
    max_total_results: int = 60,
    source_timeout: float = DEFAULT_SOURCE_TIMEOUT,
    time_range: TimeRange | None = None,
    analyze_citations: bool = False,
    citation_threshold: int = 10000,
    expand: int = 0,
    config: dict | None = None,
    progress_cb=None,
) -> Brief:
    """Run a multi-source research query.

    Args:
        topic: the search query (natural language).
        domains: restrict to these domain buckets. If None, classifier decides.
        sources: explicit source names; overrides domains when set.
        limit: max results per source.
        use_cache: read/write the 24h SQLite cache.
        synthesize: call the active model to produce a markdown brief.
        max_total_results: cap on results kept after ranking.
        source_timeout: per-source timeout in seconds.
        config: cheetahclaws config (for model + keys in synth / sources).
        progress_cb: optional callable(source_name, status) — "start"/"done"/"skipped"/"error".

    Returns:
        Brief with populated results, per-source statuses, and optional synthesis.
    """
    t_start = time.time()
    topic = (topic or "").strip()
    if not topic:
        return Brief(topic="", domains=[], results=[], statuses=[])

    # Multi-query expansion: if expand>0, ask the model for N subqueries
    # and run each, merging results before dedup/rank.
    expanded_queries: list[str] = []
    if expand and expand > 0:
        try:
            expanded_queries = _expand_subqueries(
                topic, n=min(max(expand, 2), 6), config=config or {}
            )
        except Exception:
            expanded_queries = []
        if expanded_queries and progress_cb:
            progress_cb(
                "expand",
                f"start · {len(expanded_queries)} subqueries",
            )

    # Resolve source set
    if sources:
        specs = [SOURCES[n] for n in sources if n in SOURCES]
        missing = [n for n in sources if n not in SOURCES]
        resolved_domains = _dedupe_domains(_union_domains(specs)) or ["web"]
        unknown = missing
    else:
        resolved_domains = domains or _cls.classify(topic)
        specs = sources_for_domains(resolved_domains)
        unknown = []

    if not specs:
        return Brief(
            topic=topic,
            domains=resolved_domains,
            results=[],
            statuses=[SourceStatus(name=n, ok=False, error="unknown source")
                      for n in unknown],
        )

    cfg = config or {}
    statuses: list[SourceStatus] = []
    all_results: list[Result] = []
    cache_hits = 0

    for n in unknown:
        statuses.append(SourceStatus(name=n, ok=False, error="unknown source"))

    # Expand query list: primary topic + any model-suggested subqueries.
    # Per-subquery limit shrinks so we don't blow out total result count.
    query_list = [topic] + expanded_queries
    per_query_limit = (
        limit if len(query_list) == 1
        else max(3, int(limit * 0.7 / len(query_list)) + 2)
    )

    def _cache_key(source_name: str, q: str) -> str:
        ck = q
        if time_range and time_range.is_bounded:
            ck = (
                f"{q}::since={time_range.to_iso_date('since') or ''}"
                f"::until={time_range.to_iso_date('until') or ''}"
            )
        return ck

    def _run(spec):
        """Execute one source across query_list, merging results.

        With expand disabled (len(query_list) == 1), this behaves exactly
        like a single call. With expand active, the same source is polled
        once per subquery, then all hits are concatenated (dedupe happens
        later at the aggregator level).
        """
        ts = time.time()
        aggregated: list[Result] = []
        any_cache_hit = False
        last_skipped: str | None = None
        last_err: Exception | None = None

        for q in query_list:
            if use_cache:
                cached = _cache.get(spec.name, _cache_key(spec.name, q),
                                    per_query_limit)
                if cached is not None:
                    aggregated.extend(cached)
                    any_cache_hit = True
                    continue

            try:
                try:
                    rs = spec.search(q, per_query_limit, cfg,
                                     time_range=time_range) or []
                except TypeError:
                    # Source not yet upgraded to accept time_range kwarg
                    rs = spec.search(q, per_query_limit, cfg) or []
            except SourceSkipped as e:
                last_skipped = str(e)
                # If ALL subqueries skip (same reason), report as skipped.
                # A single skip mid-expansion is non-fatal.
                continue
            except Exception as e:
                last_err = e
                break   # hard error → abort the rest

            aggregated.extend(rs)
            if use_cache and rs:
                _cache.put(spec.name, _cache_key(spec.name, q),
                           per_query_limit, rs)

        dur_ms = int((time.time() - ts) * 1000)

        if last_err is not None and not aggregated:
            return spec.name, [], SourceStatus(
                name=spec.name, ok=False, duration_ms=dur_ms,
                error=f"{type(last_err).__name__}: {str(last_err)[:160]}",
            ), False
        if last_skipped is not None and not aggregated:
            return spec.name, [], SourceStatus(
                name=spec.name, ok=False, duration_ms=dur_ms,
                skipped_reason=last_skipped,
            ), False

        return spec.name, aggregated, SourceStatus(
            name=spec.name, ok=True, count=len(aggregated), duration_ms=dur_ms,
        ), any_cache_hit

    if progress_cb:
        for spec in specs:
            progress_cb(spec.name, "start")

    # Fan out in parallel. Thread pool size = min(len(specs), 12) — most
    # sources are I/O bound so threads beat processes here.
    #
    # We DON'T use a `with` statement here. The implicit __exit__ calls
    # `shutdown(wait=True)`, which blocks on any in-flight slow source
    # (e.g. arxiv hanging on a stuck socket). When the user Ctrl+Cs, the
    # KeyboardInterrupt fires *during* that join() and Python's atexit
    # hook then ALSO joins those threads — double-blocking and killing
    # the REPL. Manual try/finally with cancel_futures=True returns the
    # partial set of completed sources immediately and lets the hung
    # source die with the daemon threads.
    ex = _cf.ThreadPoolExecutor(max_workers=min(len(specs), 12))
    try:
        futures = {ex.submit(_run, s): s for s in specs}
        try:
            iterator = _cf.as_completed(futures, timeout=source_timeout * 2)
            for fut in iterator:
                spec = futures[fut]
                try:
                    name, rs, st, from_cache = fut.result(timeout=source_timeout)
                except Exception as e:
                    st = SourceStatus(
                        name=spec.name, ok=False,
                        error=f"{type(e).__name__}: {str(e)[:160]}",
                    )
                    rs = []
                    from_cache = False
                all_results.extend(rs)
                statuses.append(st)
                if from_cache:
                    cache_hits += 1
                if progress_cb:
                    if st.skipped_reason:
                        progress_cb(spec.name, "skipped")
                    elif not st.ok:
                        progress_cb(spec.name, "error")
                    else:
                        progress_cb(spec.name, "done")
        except _cf.TimeoutError:
            # One or more sources didn't finish within source_timeout * 2.
            # Mark each unfinished source as timed-out and continue with
            # the partial result set we already have.
            for fut, spec in futures.items():
                if not fut.done():
                    statuses.append(SourceStatus(
                        name=spec.name, ok=False,
                        error="timeout (aggregator deadline exceeded)",
                    ))
                    if progress_cb:
                        progress_cb(spec.name, "error")
        except KeyboardInterrupt:
            # User interrupted during the wait — surface partial results
            # and let the executor be torn down without blocking. The
            # outer caller can decide to re-raise or swallow.
            for fut, spec in futures.items():
                if not fut.done():
                    statuses.append(SourceStatus(
                        name=spec.name, ok=False,
                        error="interrupted by user",
                    ))
            raise
    finally:
        # cancel_futures=True (Python 3.9+) drops queued-but-not-started
        # work. wait=False leaves any thread that's already running to
        # finish on its own time; it's a daemon thread and will die
        # silently with the process.
        ex.shutdown(wait=False, cancel_futures=True)

    # Dedupe + rank + cap
    deduped = _rank.dedupe(all_results)
    ranked = _rank.rank(deduped)[:max_total_results]

    brief = Brief(
        topic=topic,
        domains=resolved_domains,
        results=ranked,
        statuses=statuses,
        total_duration_ms=int((time.time() - t_start) * 1000),
        cache_hits=cache_hits,
    )

    # Extract named entities from all ranked results (offline, no LLM call)
    try:
        from . import entities as _ent
        brief._entities = _ent.extract(ranked)  # type: ignore[attr-defined]
    except Exception:
        brief._entities = None  # type: ignore[attr-defined]

    # Optional secondary citation analysis — extra S2 API calls to find
    # notable authors who cite the top academic results. Surfaced via
    # the `_notable_citers` hidden attribute; the tool/command read it.
    brief._notable_citers = []  # type: ignore[attr-defined]
    if analyze_citations:
        try:
            from . import citations as _cit
            brief._notable_citers = _cit.analyze(  # type: ignore[attr-defined]
                academic_results=[r for r in ranked if r.domain == "academic"],
                threshold=citation_threshold,
                config=cfg,
            )
        except Exception as e:
            brief._notable_citers = []  # type: ignore[attr-defined]
            brief.statuses.append(SourceStatus(
                name="citation_analysis", ok=False,
                error=f"{type(e).__name__}: {str(e)[:160]}",
            ))

    if synthesize and ranked:
        try:
            brief.synthesis = _syn.synthesize(brief, config=cfg)
        except Exception as e:
            brief.synthesis = f"(synthesis failed: {type(e).__name__}: {e})"

    return brief


def _union_domains(specs: Iterable) -> list[Domain]:
    out: list[Domain] = []
    for s in specs:
        for d in s.domains:
            if d not in out:
                out.append(d)
    return out


def _expand_subqueries(topic: str, n: int, config: dict) -> list[str]:
    """Ask the active model to propose N related subqueries.

    Returns [] if no model is available or the model response is unparseable.
    The subqueries are strict siblings of the topic (not paraphrases) so the
    merged result set covers different angles.
    """
    model = config.get("model")
    if not model:
        return []

    try:
        from cheetahclaws.providers import stream, TextChunk, AssistantTurn
    except ImportError:
        return []

    system = (
        "You are a research query expander. Given one topic, you propose "
        "N strictly related but distinct subqueries that would each pull "
        "different evidence (different benchmarks, different subfields, "
        "different stakeholders, different terminologies). Output ONLY a "
        "newline-separated list, no numbering, no preamble, no explanation."
    )
    user = (
        f"Topic: {topic}\n\n"
        f"Propose {n} distinct subqueries (5-10 words each) that would "
        f"broaden coverage of this topic. Different subqueries MUST target "
        f"different angles (theory vs. tooling vs. industry deployment vs. "
        f"controversy, etc.), not paraphrases. Output exactly {n} lines."
    )

    messages = [{"role": "user", "content": user}]
    buf: list[str] = []
    try:
        for ev in stream(
            model=model, system=system, messages=messages,
            tool_schemas=[], config={**config, "no_tools": True},
        ):
            if isinstance(ev, TextChunk):
                buf.append(ev.text)
            elif isinstance(ev, AssistantTurn):
                break
    except Exception:
        return []

    text = "".join(buf).strip()
    if not text:
        return []
    lines = [ln.strip(" -*·\t").strip() for ln in text.splitlines() if ln.strip()]
    # Drop anything that looks like LLM preamble (contains ":" near start)
    lines = [ln for ln in lines if 5 < len(ln) < 150 and not ln.lower().startswith("here")]
    # Drop the original topic if the model accidentally echoed it
    lines = [ln for ln in lines if ln.lower() != topic.lower()]
    return lines[:n]


def compare(
    topic_a: str,
    topic_b: str,
    topic_c: str | None = None,
    domains: list[Domain] | None = None,
    sources: list[str] | None = None,
    limit: int = 10,
    use_cache: bool = True,
    time_range: TimeRange | None = None,
    config: dict | None = None,
    progress_cb=None,
) -> dict:
    """Run 2 or 3 research queries in parallel and build a side-by-side brief.

    Returns a dict:
        {
            "topics":   [topic_a, topic_b, (topic_c)?],
            "briefs":   [Brief, Brief, (Brief)?],   # synthesis disabled
            "comparison": str,                       # LLM-generated comparison prose
            "total_duration_ms": int,
        }

    The caller renders a single unified report from this dict via
    `synthesizer.render_compare_brief()`.
    """
    topics = [topic_a.strip(), topic_b.strip()]
    if topic_c and topic_c.strip():
        topics.append(topic_c.strip())

    t_start = time.time()

    def _one(t: str) -> Brief:
        return research(
            topic=t, domains=domains, sources=sources,
            limit=limit, use_cache=use_cache, synthesize=False,
            time_range=time_range, analyze_citations=False,
            expand=0, config=config or {},
            progress_cb=(lambda n, s, _t=t: progress_cb(f"{_t}::{n}", s))
                if progress_cb else None,
        )

    with _cf.ThreadPoolExecutor(max_workers=len(topics)) as ex:
        briefs = list(ex.map(_one, topics))

    # Compare-mode synthesis
    comparison = ""
    cfg = config or {}
    if cfg.get("model"):
        try:
            comparison = _syn.synthesize_comparison(topics, briefs, config=cfg)
        except Exception as e:
            comparison = f"(comparison synthesis failed: {type(e).__name__}: {e})"

    return {
        "topics": topics,
        "briefs": briefs,
        "comparison": comparison,
        "total_duration_ms": int((time.time() - t_start) * 1000),
    }


def _dedupe_domains(ds: list[Domain]) -> list[Domain]:
    seen = set()
    out = []
    for d in ds:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out
