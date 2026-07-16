"""
/trading slash command for CheetahClaws.

Subcommands:
  /trading analyze <SYMBOL>     — full multi-agent analysis
  /trading backtest <strategy>  — backtest a strategy
  /trading price <SYMBOL>       — quick price check
  /trading indicators <SYMBOL>  — technical indicators
  /trading status               — show trading memory status
  /trading history              — view past decisions
  /trading memory [search|clear] — manage trading memory
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from cheetahclaws.ui.render import info, ok, warn, err, clr


# ── History storage ────────────────────────────────────────────────────────

_HISTORY_DIR = Path.home() / ".cheetahclaws" / "trading" / "history"


def _save_decision(symbol: str, signal: str, details: str) -> None:
    """Save a trading decision to history."""
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "symbol": symbol,
        "signal": signal,
        "timestamp": timestamp,
        "details": details[:2000],
    }
    path = _HISTORY_DIR / f"{timestamp}_{symbol}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))


# ── Analysis orchestrator ──────────────────────────────────────────────────

def _run_analysis(symbol: str, state, config: dict) -> bool:
    """Run full multi-agent analysis pipeline.

    Flow:
      1. Data collection (technical, fundamental, news)
      2. Bull/Bear researcher debate
      3. Research judge recommendation
      4. Risk management panel debate
      5. Portfolio manager final decision

    The LLM invocations happen via the agent loop — this function
    generates the full analysis prompt and injects it as a message.
    """
    from .agents import analyst, researcher, risk_manager, portfolio_manager

    info(f"\n{'='*60}")
    info(f"  Trading Analysis: {clr(symbol, 'bold')}")
    info(f"  Date: {datetime.now().strftime('%Y-%m-%d')}")
    info(f"{'='*60}\n")

    # Phase 1: Data collection
    info(clr("Phase 1/5: Collecting market data...", "cyan"))
    reports = analyst.run_all_analyses(symbol)

    for name, report in reports.items():
        if "Error" in report:
            warn(f"  {name}: {report}")
        else:
            lines = report.count('\n')
            ok(f"  {name}: {lines} lines collected")

    trade_date = datetime.now().strftime("%Y-%m-%d")

    # Phase 2-5: Build the full multi-agent prompt and return it
    # so the REPL sends it to the AI for processing
    info(clr("\nPhase 2-5: Sending to AI for multi-agent analysis...", "cyan"))
    info(clr("  1. Bull vs Bear research debate", "dim"))
    info(clr("  2. Research judge recommendation", "dim"))
    info(clr("  3. Risk management panel (aggressive/conservative/neutral)", "dim"))
    info(clr("  4. Portfolio manager final decision", "dim"))
    info("")

    full_prompt = _build_analysis_prompt(symbol, trade_date, reports)

    # Return as __ssj_query__ so the REPL sends this to the AI
    return ("__ssj_query__", full_prompt)


def _build_analysis_prompt(
    symbol: str, trade_date: str, reports: dict[str, str]
) -> str:
    """Build the multi-agent analysis prompt.

    Designed to be concise enough for small models (gpt-5-nano) while
    still producing structured output across all 5 phases.

    Includes macro context (SPY/QQQ/VIX/TNX), upcoming earnings dates,
    and the current paper-trading book — without these the agent
    recommends single-name BUYs in RISK-OFF regimes, fails to flag
    pre-earnings event risk, and ignores its own existing exposure.
    """
    from . import macro, earnings, paper_trader
    from .alt_data import insider, sentiment, trends

    macro_block = macro.render_macro_context()
    earnings_block = earnings.render_earnings_warning(symbol)
    insider_block = insider.render_insider_summary(symbol)
    sentiment_block = sentiment.render_sentiment_block(symbol)
    trends_block = trends.render_trends_block(symbol)

    # Current portfolio exposure — feed back to the LLM so it doesn't
    # recommend "BUY 5%" on a sector already at 30%.
    book = paper_trader.open_position_summary()
    if book["open_count"]:
        book_lines = [f"## Current Open Paper Trades ({book['open_count']})"]
        book_lines.append(f"- Total exposure: {book['total_exposure_pct']:.1f}%")
        if book["by_sector_pct"]:
            book_lines.append("- Sector breakdown:")
            for sec, pct in sorted(book["by_sector_pct"].items(), key=lambda x: -x[1]):
                book_lines.append(f"  - {sec}: {pct:.1f}%")
        if book["symbols"]:
            book_lines.append(f"- Symbols: {', '.join(book['symbols'])}")
        book_block = "\n".join(book_lines)
    else:
        book_block = ""

    # Check if we have real data or just errors
    has_data = any(
        "Error" not in reports.get(k, "Error")
        for k in ("technical", "fundamental", "news")
    )

    if has_data:
        data_section = f"""## Market Data for {symbol} (Date: {trade_date})

### Technical
{reports.get('technical', 'N/A')}

### Fundamentals
{reports.get('fundamental', 'N/A')}

### News
{reports.get('news', 'N/A')}"""
    else:
        data_section = f"""## Note: Market data unavailable (yfinance not installed)
Use your general knowledge about {symbol} as of {trade_date}. State clearly when using general knowledge vs. data."""

    # Stitch context blocks together — only include non-empty ones.
    context_sections = "\n\n".join(
        b for b in (macro_block, earnings_block,
                    insider_block, sentiment_block, trends_block,
                    book_block)
        if b
    )
    if context_sections:
        data_section = context_sections + "\n\n" + data_section

    return f"""Analyze {symbol} using the 5-phase trading pipeline below. Write each phase ONCE, then move to the next. Do NOT repeat any phase.

{data_section}

---

## Phase 1: BULL CASE
Write 3-4 bullet points arguing FOR buying {symbol} (growth drivers, technical strength, positive catalysts).
End with: **BULL VERDICT: [Buy/Strong Buy]** — one sentence thesis.

## Phase 2: BEAR CASE
Write 3-4 bullet points arguing AGAINST buying {symbol} (risks, overvaluation, negative catalysts).
End with: **BEAR VERDICT: [Sell/Lean Sell]** — one sentence thesis.

## Phase 3: JUDGE DECISION
Which case is stronger? State: **DECISION: BUY / SELL / HOLD** with confidence (High/Medium/Low) and suggested position size (% of portfolio).

## Phase 4: RISK PANEL
- **Aggressive**: one sentence arguing for larger position
- **Conservative**: one sentence arguing for smaller position with tighter stops
- **Neutral**: one sentence with balanced recommendation

## Phase 5: FINAL RATING
**RATING: [BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL]**
**Summary**: 2 sentences on investment thesis.
**Plan**: Entry, Position Size %, Stop Loss %, Take Profit %, Time Horizon.
**Top 3 Risks**: numbered list.
**Conviction**: High / Medium / Low

IMPORTANT: Write each phase exactly once. Do not loop back to Phase 1 after starting Phase 2.
"""


# ── Command handler ────────────────────────────────────────────────────────

def _cmd_trading(args: str, state, config) -> bool:
    """Handle /trading and its subcommands."""
    parts = args.split() if args.strip() else []
    sub = parts[0].lower() if parts else ""
    rest = " ".join(parts[1:])

    if sub == "analyze" or sub == "analyse":
        if not rest:
            err("Usage: /trading analyze <SYMBOL>")
            err("  Example: /trading analyze AAPL")
            return True
        symbol = rest.split()[0].upper()
        return _run_analysis(symbol, state, config)

    elif sub == "backtest":
        return _cmd_backtest(rest, state, config)

    elif sub == "price":
        return _cmd_price(rest)

    elif sub == "indicators":
        return _cmd_indicators(rest)

    elif sub == "status":
        return _cmd_status()

    elif sub == "history":
        return _cmd_history()

    elif sub == "memory":
        return _cmd_memory(rest)

    elif sub == "paper":
        return _cmd_paper(rest)

    elif sub == "calibration" or sub == "calibrate":
        return _cmd_calibration()

    elif sub == "watch":
        return _cmd_watch(rest)

    elif sub == "scan":
        return _cmd_scan(rest, state, config)

    elif sub == "verify":
        return _cmd_verify(rest)

    elif sub == "walkforward" or sub == "wf":
        return _cmd_walkforward(rest, config)

    elif sub == "review":
        return _cmd_review(rest, state, config)

    elif sub == "manage":
        return _cmd_manage(rest, state, config)

    elif sub == "optimize":
        return _cmd_optimize(rest)

    elif sub == "ml":
        return _cmd_ml(rest)

    elif sub == "discover":
        return _cmd_discover(rest, state, config)

    elif sub == "rank":
        return _cmd_rank(rest, state, config)

    elif sub == "anomaly":
        return _cmd_anomaly(rest)

    elif sub == "monitor":
        return _cmd_monitor(rest, state, config)

    elif sub == "factors":
        return _cmd_factors(rest)

    elif sub == "agent":
        return _cmd_agent(rest, state, config)

    else:
        _show_help()
        return True


# ── /trading discover ─────────────────────────────────────────────────────

def _cmd_discover(args: str, state, config) -> bool:
    """Run discovery scanners to surface candidate tickers."""
    from .discover import orchestrator
    from . import paper_trader

    parts = args.split() if args else []
    sub_or_source = parts[0].lower() if parts else "all"

    sources: list[str] | None
    if sub_or_source == "all":
        sources = None
    elif sub_or_source in {"insider", "earnings", "momentum-quality", "sector"}:
        sources = [sub_or_source]
    else:
        err(f"Unknown discovery source: {sub_or_source}")
        info("  Sources: all, insider, earnings, momentum-quality, sector")
        return True

    universe = "sp100"
    add_to_watchlist = 0
    for i, p in enumerate(parts):
        if p == "--universe" and i + 1 < len(parts):
            universe = parts[i + 1]
        if p == "--add-watchlist" and i + 1 < len(parts):
            try:
                add_to_watchlist = int(parts[i + 1])
            except ValueError:
                pass

    info(clr(f"\nRunning discovery (sources={sources or 'all'}, "
             f"universe={universe})...", "cyan"))
    info("  This can take 1-3 minutes for sp100. Progress shown below.")

    last_done = [0]
    def progress(done, total, sym):
        # Print a dot every 10 symbols to avoid spam
        if done - last_done[0] >= 10 or done == total:
            last_done[0] = done
            print(f"  [{done}/{total}] last: {sym}", flush=True)

    result = orchestrator.run(sources=sources, universe=universe,
                              progress_cb=progress)

    print()
    print(orchestrator.render_report(result))

    if add_to_watchlist > 0:
        added = []
        for entry in result["ranked"][:add_to_watchlist]:
            paper_trader.watchlist_add(entry["symbol"],
                                       note=f"discovery: {' · '.join(set(entry['sources']))}")
            added.append(entry["symbol"])
        if added:
            print()
            ok(f"Added {len(added)} to watchlist: {', '.join(added)}")
    return True


# ── /trading rank ─────────────────────────────────────────────────────────

def _cmd_rank(args: str, state, config) -> bool:
    """Composite ranking — what's worth investing in NOW."""
    from . import ranker

    parts = args.split() if args else []
    if parts and not parts[0].startswith("--"):
        symbols = [s.upper() for s in parts[0].replace(",", " ").split()]
        universe = None
    else:
        symbols = None
        universe = "sp100"

    use_discovery = "--no-discovery" not in parts
    use_calibration = "--no-calibration" not in parts

    info(clr(f"\nRanking — universe={universe or 'custom'}, "
             f"discovery={'on' if use_discovery else 'off'}", "cyan"))
    info("  Takes ~1-2 min for sp100 (factors + discovery scanners).")

    last_done = [0]
    def progress(done, total, sym):
        if done - last_done[0] >= 10 or done == total:
            last_done[0] = done
            print(f"  [{done}/{total}] {sym}", flush=True)

    rows = ranker.rank(universe=universe, symbols=symbols,
                       use_discovery=use_discovery,
                       use_calibration=use_calibration,
                       progress_cb=progress)
    print()
    print(ranker.render_rank_report(rows))
    return True


# ── /trading anomaly ──────────────────────────────────────────────────────

def _cmd_anomaly(args: str) -> bool:
    """Anomaly scan — unusual volume, price gaps, vol spikes."""
    from .discover import anomaly
    from . import paper_trader

    parts = args.split() if args else []
    if parts:
        symbols = [s.upper() for s in parts[0].replace(",", " ").split()]
    else:
        wl = paper_trader.watchlist_list()
        if not wl:
            err("Watchlist is empty. Pass symbols: `/trading anomaly NVDA,AMD,SPY`")
            return True
        symbols = [w["symbol"] for w in wl]

    info(f"\nScanning {len(symbols)} symbol(s) for anomalies...")
    last_done = [0]
    def progress(done, total, sym):
        if done - last_done[0] >= 5 or done == total:
            last_done[0] = done
            print(f"  [{done}/{total}] {sym}", flush=True)

    hits = anomaly.scan(symbols, progress_cb=progress)
    print()
    print(anomaly.render_anomaly_report(hits))
    return True


# ── /trading monitor ──────────────────────────────────────────────────────

def _cmd_monitor(args: str, state, config) -> bool:
    """Monitor — periodic scan + optional bridge alerts."""
    from . import monitor

    parts = args.split() if args else []
    sub = parts[0].lower() if parts else "scan"

    if sub in ("scan", "run"):
        info(clr("\nRunning monitor scan...", "cyan"))
        info("  Checking: anomalies + stops/take-profits + earnings + new insider activity")
        alerts = monitor.scan()
        print()
        print(monitor.render_alerts(alerts))

        # If the user passed --notify, dispatch to bridges
        if "--notify" in parts:
            channels = []
            for ch in ("telegram", "slack", "wechat"):
                if ch in parts:
                    channels.append(ch)
            if not channels:
                channels = ["telegram", "slack", "wechat"]
            r = monitor.dispatch_to_bridges(alerts, bridges=channels, config=config)
            info(f"\nDispatched to {r['sent']} channel(s): {r['channels']}")
            if r.get("errors"):
                warn(f"  Errors: {r['errors']}")
        return True

    if sub == "status":
        last = monitor.last_run()
        if not last:
            info("Monitor has never run. Try `/trading monitor scan`.")
            return True
        info(clr("Last monitor run", "bold"))
        for k, v in last.items():
            info(f"  {k}: {v}")
        return True

    err(f"Unknown monitor subcommand: {sub}")
    info("  Subcommands: scan [--notify [telegram] [slack] [wechat]], status")
    return True


# ── /trading factors ─────────────────────────────────────────────────────

def _cmd_factors(args: str) -> bool:
    """Show raw factor scores for a universe."""
    from . import factors as f

    parts = args.split() if args else []
    if parts and not parts[0].startswith("--"):
        symbols = [s.upper() for s in parts[0].replace(",", " ").split()]
    else:
        from .universe import SP100
        symbols = SP100[:50]  # default to first 50 of S&P 100 for speed

    if "--clear-cache" in parts:
        f.clear_cache()
        info("Factor cache cleared.")

    info(f"\nFactor scoring on {len(symbols)} symbol(s)...")

    last_done = [0]
    def progress(done, total, sym):
        if done - last_done[0] >= 5 or done == total:
            last_done[0] = done
            print(f"  [{done}/{total}] {sym}", flush=True)

    rows = f.scan_universe(symbols, progress_cb=progress)
    f.score(rows)
    print()
    print(f.render_factor_table(rows, top=25))
    return True


# ── /trading agent — agentic research orchestrator ──────────────────────


def _cmd_agent(args: str, state, config) -> bool:
    """LLM-on-top-of-deterministic-tools research agent.

    Workflow (single LLM round-trip, fast + cheap):
      1. /trading discover — pure-Python scan SP100 across 4 sources
      2. /trading factors  — pure-Python factor scores on top hits
      3. macro snapshot    — SPY/QQQ/VIX/TNX
      4. LLM synthesis     — read user's question + all structured data
                              above and produce a focused dossier ranking
                              candidates by FIT to the question, with
                              entry consideration + macro overlay.

    The LLM never re-fetches data the deterministic pipeline already
    has — it only reasons over what's already on the table. This keeps
    cost low ($0.01-0.05/run on Qwen2.5-72B-class models) and avoids
    LLMs hallucinating numbers.
    """
    if not args.strip():
        err("Usage: /trading agent <research question>")
        info("")
        info("Examples:")
        info("  /trading agent find AI-infra names with insider buying")
        info("  /trading agent which sector looks strongest for the next 30 days")
        info("  /trading agent show me 3 defensive names with positive momentum")
        info("  /trading agent compare AMZN GOOGL META on quality + momentum")
        return True

    question = args.strip()

    info(clr("\nTrading Agent — multi-step research", "bold"))
    info(f"{'=' * 60}")
    info(f"Question: {question}")
    info("")

    # ── Step 1: Discovery (deterministic) ────────────────────────────
    info(clr("Step 1/3: Scanning SP100 candidates via discover...", "cyan"))
    ranked: list = []
    discover_notes: list = []
    try:
        from .discover.orchestrator import run as discover_run
        last_done = [0]

        def _disc_progress(done, total, sym):
            if done - last_done[0] >= 20 or done == total:
                last_done[0] = done
                print(f"  [{done}/{total}] {sym}", flush=True)

        result = discover_run(
            sources=None,           # all four scanners
            universe="sp100",
            top_n=20,
            progress_cb=_disc_progress,
        )
        ranked = result.get("ranked", []) or []
        discover_notes = result.get("notes", []) or []
        ok(f"  → {result.get('n_unique', 0)} unique tickers, "
           f"{result.get('n_total_hits', 0)} hits")
    except Exception as e:
        err(f"  Discovery failed: {type(e).__name__}: {e}")
        ranked = []

    # ── Step 2: Factor scores on top discoveries ─────────────────────
    info(clr("\nStep 2/3: Computing factor scores on top candidates...",
              "cyan"))
    factor_summary: list = []
    if ranked:
        try:
            from . import factors as f
            symbols = [r["symbol"] for r in ranked[:15]]
            last_f = [0]

            def _f_progress(done, total, sym):
                if done - last_f[0] >= 5 or done == total:
                    last_f[0] = done
                    print(f"  [{done}/{total}] {sym}", flush=True)

            rows = f.scan_universe(symbols, progress_cb=_f_progress)
            scored = f.score(rows)
            for r in scored:
                factor_summary.append({
                    "symbol":     r.symbol,
                    "momentum":   r.momentum_score,
                    "quality":    r.quality_score,
                    "low_vol":    r.low_vol_score,
                    "composite":  r.composite_score,
                })
            ok(f"  → factor scores computed for {len(factor_summary)} "
               f"symbol(s)")
        except Exception as e:
            err(f"  Factor scan failed: {type(e).__name__}: {e}")

    # ── Step 3: Macro context ─────────────────────────────────────────
    info(clr("\nStep 3/3: Loading macro context...", "cyan"))
    macro_block = "(macro context unavailable)"
    try:
        from . import macro
        macro_block = macro.render_macro_context()
        ok("  → macro snapshot loaded")
    except Exception as e:
        err(f"  Macro failed: {type(e).__name__}: {e}")

    # ── Build prompt for LLM synthesis ────────────────────────────────
    prompt = _build_agent_prompt(
        question=question,
        ranked=ranked,
        factors=factor_summary,
        macro_block=macro_block,
        discover_notes=discover_notes,
    )

    info(clr("\nSending compiled brief to AI for synthesis...\n", "cyan"))
    return ("__ssj_query__", prompt)


def _build_agent_prompt(
    *,
    question: str,
    ranked: list,
    factors: list,
    macro_block: str,
    discover_notes: list,
) -> str:
    """Compose the dossier-synthesis prompt. All structured data is
    embedded inline so the LLM doesn't need to call any tools — pure
    reasoning, single round-trip."""
    today = datetime.now().strftime("%Y-%m-%d")

    # Discovery table
    if ranked:
        lines = [
            "## Discovery candidates (top 20, ranked by aggregate score)",
            "Sources: insider (SEC EDGAR Form 4), earnings (yfinance "
            "earnings beats), momentum-quality (yfinance factors), "
            "sector (SPDR Select ETFs). Weights: insider 1.0, earnings "
            "0.9, momentum-quality 0.7, sector 0.5; +0.5 bonus when ≥2 "
            "sources flag the same ticker.",
            "",
            "| # | Symbol | Sources | Score | Top reasons |",
            "|---:|---|---|---:|---|",
        ]
        for i, r in enumerate(ranked[:20], 1):
            srcs = " · ".join(sorted(set(r.get("sources", []))))
            reasons = "; ".join(r.get("reasons", [])[:2])
            if len(reasons) > 220:
                reasons = reasons[:217] + "..."
            score_v = r.get("aggregate_score", 0.0)
            lines.append(
                f"| {i} | **{r.get('symbol','')}** | {srcs} | "
                f"{score_v:.2f} | {reasons} |"
            )
        discover_md = "\n".join(lines)
    else:
        discover_md = ("## Discovery candidates\n_No candidates "
                        "returned — discovery layer failed or filters "
                        "too tight. The user may be asking about specific "
                        "symbols not in SP100; reason from the question "
                        "directly._")

    # Factor table
    if factors:
        lines = [
            "## Factor scores (top 15 of discovery candidates)",
            "Scores normalised 0-1 within the scanned cohort. Composite "
            "= 0.5×momentum + 0.3×quality + 0.2×low_vol.",
            "",
            "| Symbol | Momentum | Quality | Low-Vol | Composite |",
            "|---|---:|---:|---:|---:|",
        ]
        for f_ in factors:
            def _fmt(v):
                return f"{v:.2f}" if isinstance(v, (int, float)) else "—"
            lines.append(
                f"| {f_['symbol']} | {_fmt(f_.get('momentum'))} | "
                f"{_fmt(f_.get('quality'))} | "
                f"{_fmt(f_.get('low_vol'))} | "
                f"{_fmt(f_.get('composite'))} |"
            )
        factor_md = "\n".join(lines)
    else:
        factor_md = "## Factor scores\n_(unavailable)_"

    # Discovery notes (errors, skipped sources, etc.)
    notes_md = ""
    if discover_notes:
        notes_md = ("## Discovery notes\n"
                     + "\n".join(f"- {n}" for n in discover_notes))

    return f"""You are a senior portfolio research analyst writing a
focused investment dossier for the user. The data below was just
gathered from SEC EDGAR + Yahoo Finance via deterministic Python
pipelines — treat it as authoritative current state. Today is {today}.

## User's research question
> {question}

{macro_block}

{discover_md}

{factor_md}

{notes_md}

---

## Your task

Produce a markdown dossier that is **decisive and concise**. Structure:

### 1. Top 5 candidates ranked by FIT to the user's question
Re-rank the discovery list specifically for the user's ask (not the raw
score). For each of 5:
- **Symbol — 1-line thesis** explaining why this matches the user's
  question (cite specific data: insider count, earnings beat %, factor
  scores, sector rank).
- **Bull point** (1 sentence, from the data above).
- **Risk / bear point** (1 sentence — what would invalidate the thesis).
- **Entry consideration** (price level / catalyst / timing trigger).

### 2. Macro overlay (3-5 sentences)
Given the macro context above (regime, VIX, yields), which 1-2 of your 5
become higher-conviction and which become lower-conviction? Why?

### 3. What to do next
- **Top 1-2** that deserve a full multi-agent debate via
  `/trading analyze <SYMBOL>` first. Order them by priority.
- Any candidates you are explicitly **passing on** despite a high score,
  and the reason.

### Constraints
- **Be decisive**: no hedging like "could potentially". Take a position.
- **Cite data points** from the tables above (`12 Form 4 filings`,
  `+8.2% sector 1m`, `momentum 0.74`, etc.) — don't make up numbers.
- **Don't recommend buying**. The user does the buying. You decide what
  deserves their attention next.
- **Form 4 caveat**: filings are *counted*, not direction-parsed. A
  high count could be either heavy buying OR heavy selling — flag any
  candidates where the user should manually verify direction via the
  SEC URLs.
- Keep the whole dossier under ~600 words. Tight and actionable.
"""


# ── /trading review (add/reduce/exit on existing positions) ──────────────

def _cmd_review(args: str, state, config) -> bool:
    """Run multi-agent debate on each open paper position: hold/add/reduce/exit."""
    from . import paper_trader
    from .data import fetchers

    parts = args.split() if args else []
    target = parts[0].upper() if parts else None

    opens = paper_trader.list_trades(status="open")
    if target:
        opens = [t for t in opens if t.symbol == target]
    if not opens:
        info("No matching open paper positions to review.")
        return True

    positions_summary = []
    for t in opens:
        pi = fetchers.fetch_current_price(t.symbol)
        cur_price = pi.get("price") if isinstance(pi, dict) else None
        unreal = None
        if cur_price and t.entry_price and t.entry_price > 0:
            unreal = (cur_price - t.entry_price) / t.entry_price * 100.0
            if t.signal in ("SELL", "UNDERWEIGHT"):
                unreal = -unreal
        positions_summary.append({
            "id":         t.id,
            "symbol":     t.symbol,
            "signal":     t.signal,
            "confidence": t.confidence,
            "entry":      t.entry_price,
            "current":    cur_price,
            "unrealized_pct": unreal,
            "size_pct":   t.position_size_pct,
            "stop_pct":   t.stop_loss_pct,
            "tp_pct":     t.take_profit_pct,
            "thesis":     t.thesis,
        })

    info(clr(f"\nReviewing {len(positions_summary)} open position(s)...", "cyan"))
    prompt = _build_review_prompt(positions_summary)
    return ("__ssj_query__", prompt)


def _build_review_prompt(positions: list[dict]) -> str:
    """Build the multi-agent review prompt for incremental decisions."""
    from . import macro

    macro_block = macro.render_macro_context()

    table = ["| ID | Symbol | Original | Conf | Entry | Current | Unrealized | Stop | TP | Thesis |",
             "|---:|---|---|---|---:|---:|---:|---:|---:|---|"]
    for p in positions:
        unreal = f"{p['unrealized_pct']:+.2f}%" if p["unrealized_pct"] is not None else "—"
        cur = f"${p['current']:.2f}" if p["current"] else "—"
        ent = f"${p['entry']:.2f}" if p["entry"] else "—"
        thesis = (p['thesis'] or "")[:80].replace("\n", " ").replace("|", "/")
        table.append(
            f"| {p['id']} | {p['symbol']} | {p['signal']} | {p['confidence']} | "
            f"{ent} | {cur} | {unreal} | "
            f"{p['stop_pct'] or '—'}% | {p['tp_pct'] or '—'}% | {thesis} |"
        )
    table_md = "\n".join(table)

    macro_section = (macro_block + "\n\n") if macro_block else ""

    return f"""# Position Review — incremental decisions on existing book

You are reviewing {len(positions)} OPEN position(s) and must give an
ACTION per position. This is NOT a fresh BUY/SELL recommendation — it's
"given that we already own this, what now?". Available actions:

- `HOLD` — thesis intact, no change
- `ADD` — thesis strengthened; add to the position (specify size %)
- `TRIM` — thesis weakened OR profit-taking; reduce by X%
- `EXIT` — thesis broken, stop hit, or risk-reward inverted; close fully

{macro_section}## Current book

{table_md}

---

## Multi-agent review (write each phase ONCE — do NOT repeat)

### Phase 1: Bull desk
For each position above, write 1 sentence on whether the original bullish
thesis still holds given current price action and macro.

### Phase 2: Bear desk
For each position above, write 1 sentence on what could break the thesis.

### Phase 3: Risk officer
For each position above, flag if stop should be tightened (price moved up
since entry — protect gains) or if position is hitting a defined risk
limit (max drawdown, time horizon expired).

### Phase 4: Portfolio manager — FINAL ACTIONS

For EACH position output a row in this exact format (one row per ID):

```
ACTION ID=<id> SYMBOL=<sym> DECISION=<HOLD|ADD|TRIM|EXIT> SIZE_DELTA=<%> NEW_STOP=<% or same> REASON=<one sentence>
```

Example:
```
ACTION ID=12 SYMBOL=NVDA DECISION=TRIM SIZE_DELTA=-50% NEW_STOP=2% REASON=+22% on entry, locking in profit and tightening trailing stop.
ACTION ID=15 SYMBOL=AMD DECISION=EXIT SIZE_DELTA=-100% NEW_STOP=N/A REASON=Closed below 50d, original momentum thesis broken.
```

Only emit ACTION lines (no other text in Phase 4) so the user can grep
them and the system can persist them automatically."""


# ── /trading manage (managed paper portfolio) ────────────────────────────

def _cmd_manage(args: str, state, config) -> bool:
    """Handle /trading manage subcommands."""
    from . import managed

    parts = args.split() if args else []
    sub = parts[0].lower() if parts else "list"

    if sub == "list" or sub == "":
        ports = managed.list_portfolios()
        if not ports:
            info("No managed portfolios. Create one with `/trading manage start <name> <usd>`.")
            return True
        info(clr("\nManaged portfolios:", "bold"))
        info(f"  {'Name':<20} {'Created':<12} {'Initial':>10} {'Cash':>10}")
        for p in ports:
            info(f"  {p['name']:<20} {p['created_at'][:10]:<12} "
                 f"${p['initial_cash']:>9.2f} ${p['cash']:>9.2f}")
        return True

    if sub == "start":
        if len(parts) < 3:
            err("Usage: /trading manage start <name> <USD>")
            err("  Example: /trading manage start hundred 100")
            return True
        name = parts[1]
        try:
            usd = float(parts[2])
        except ValueError:
            err(f"Invalid USD amount: {parts[2]}")
            return True
        broker = managed.start_portfolio(name=name, initial_cash=usd)
        ok(f"Portfolio '{name}' created with ${usd:.2f}.")
        info("Run `/trading manage step " + name + "` to make the first allocation.")
        info("Run `/trading manage report " + name + "` later to check PnL.")
        return True

    if sub == "step":
        if len(parts) < 2:
            err("Usage: /trading manage step <name> [--dry]")
            return True
        name = parts[1]
        dry = "--dry" in parts
        info(f"Stepping portfolio '{name}'" + (" (dry run)" if dry else "") + "...")
        result = managed.step(name, dry_run=dry)
        for n in result.notes:
            info(f"  {n}")
        if result.orders:
            info(clr(f"  Placed {len(result.orders)} order(s).", "green"))
            for o in result.orders:
                tick = "✓" if o.get("success") else "✗"
                info(f"    {tick} {o.get('side', '?')} {o.get('quantity', 0):.4f} "
                     f"{o.get('symbol', '?')} @ "
                     f"${(o.get('fill_price') or 0):.2f}")
        delta = result.equity_after - result.equity_before
        sign = "+" if delta >= 0 else ""
        info(f"  Equity: ${result.equity_before:.2f} → ${result.equity_after:.2f} "
             f"({sign}${delta:.2f})")
        return True

    if sub == "status":
        if len(parts) < 2:
            err("Usage: /trading manage status <name>")
            return True
        s = managed.status(parts[1])
        sign = "+" if s["pnl_dollars"] >= 0 else ""
        info(f"\nPortfolio '{s['portfolio']}'")
        info(f"  Initial:  ${s['initial_cash']:.2f}")
        info(f"  Equity:   ${s['equity']:.2f}  ({sign}${s['pnl_dollars']:.2f}, "
             f"{sign}{s['pnl_pct']:.2f}%)")
        info(f"  Cash:     ${s['cash']:.2f}")
        info(f"  Open:     {s['open_positions_count']} positions")
        if s["positions"]:
            info("  Holdings:")
            for p in s["positions"]:
                cur = f"${p['current_price']:.2f}" if p['current_price'] else "—"
                mv = f"${p['market_value']:.2f}" if p['market_value'] is not None else "—"
                info(f"    {p['symbol']:<8} qty={p['quantity']:.4f} "
                     f"avg=${p['avg_cost']:.2f} cur={cur} mv={mv}")
        return True

    if sub == "report":
        if len(parts) < 2:
            err("Usage: /trading manage report <name>")
            return True
        print(managed.report(parts[1]))
        return True

    err(f"Unknown manage subcommand: {sub}")
    info("  Subcommands: list, start <name> <usd>, step <name> [--dry], "
         "status <name>, report <name>")
    return True


# ── /trading optimize (mean-variance) ─────────────────────────────────────

def _cmd_optimize(args: str) -> bool:
    """Run mean-variance optimisation on the watchlist (or specified syms)."""
    from . import paper_trader, portfolio
    from .data import fetchers

    parts = args.split() if args else []
    if parts and not parts[0].startswith("--"):
        symbols = [s.upper() for s in parts[0].replace(",", " ").split()]
    else:
        wl = paper_trader.watchlist_list()
        if not wl:
            err("Watchlist empty. Pass symbols: `/trading optimize AAPL,MSFT,SPY,QQQ`.")
            return True
        symbols = [w["symbol"] for w in wl]

    max_weight = 0.20
    for i, p in enumerate(parts):
        if p == "--max-weight" and i + 1 < len(parts):
            try:
                max_weight = float(parts[i + 1])
            except ValueError:
                pass

    info(f"\nMean-variance optimization on {len(symbols)} symbol(s) "
         f"(single-name cap {max_weight*100:.0f}%)...")

    candidates = []
    for sym in symbols:
        result = fetchers.fetch_market_data(sym, interval="1d")
        if result.get("error") or not result.get("data"):
            warn(f"  {sym}: data error, skipping")
            continue
        rows = result["data"]
        if len(rows) < 60:
            warn(f"  {sym}: only {len(rows)} bars, skipping")
            continue
        candidates.append(portfolio.Candidate(
            symbol=sym, closes=[r["close"] for r in rows]
        ))

    if not candidates:
        err("No tradeable candidates.")
        return True

    result = portfolio.optimize(candidates, max_weight=max_weight)
    print(portfolio.render_optimization_report(result))
    return True


# ── /trading ml (train / predict) ────────────────────────────────────────

def _cmd_ml(args: str) -> bool:
    """ML stacker subcommands: train, status."""
    from . import paper_trader
    from .ml import features as feat_mod, stacker as st

    parts = args.split() if args else []
    sub = parts[0].lower() if parts else "status"

    if sub == "train":
        closed = paper_trader.list_trades(status="closed", limit=10000)
        if not closed:
            err("No closed paper trades — nothing to train on.")
            return True
        rows, cols = feat_mod.build_dataset(closed)
        info(f"Training stacker on {len(rows)} closed trades...")
        result = st.train(rows, cols=cols)
        print(st.render_train_report(result))
        return True

    if sub == "status":
        from pathlib import Path
        path = Path.home() / ".cheetahclaws" / "trading" / "ml" / "stacker.pkl"
        if not path.exists():
            info("No trained stacker model.")
            info("Train with `/trading ml train` once you have ≥30 closed trades.")
            return True
        ok(f"Stacker model present at {path}")
        info(f"  Size: {path.stat().st_size:,} bytes")
        return True

    err(f"Unknown ml subcommand: {sub}")
    info("  Subcommands: train, status")
    return True


# ── Paper trading ──────────────────────────────────────────────────────────

def _cmd_paper(args: str) -> bool:
    """Handle /trading paper subcommands."""
    from . import paper_trader

    parts = args.split() if args else []
    sub = parts[0].lower() if parts else "list"

    if sub == "list":
        status_filter = parts[1] if len(parts) > 1 else None
        if status_filter not in (None, "open", "closed"):
            err(f"Invalid status filter: {status_filter}. Use 'open' or 'closed'.")
            return True
        trades = paper_trader.list_trades(status=status_filter)
        if not trades:
            info("No paper trades recorded yet.")
            info("Run `/trading analyze <SYMBOL>` and the agent's recommendation"
                 " is auto-recorded as a paper trade.")
            return True
        info(clr(f"\nPaper trades ({status_filter or 'all'}):", "bold"))
        info(f"{'ID':>4}  {'Symbol':<10}  {'Signal':<11}  {'Conf':<7}  "
             f"{'Size':>6}  {'Entry':>8}  {'Status':<7}  Realized")
        for t in trades:
            sig_color = "green" if t.signal in ("BUY", "OVERWEIGHT") else \
                        "red" if t.signal in ("SELL", "UNDERWEIGHT") else "yellow"
            realized = f"{t.realized_return_pct:+.2f}%" if t.realized_return_pct is not None else "—"
            entry = f"${t.entry_price:.2f}" if t.entry_price else "—"
            size = f"{t.position_size_pct:.1f}%" if t.position_size_pct else "—"
            info(f"{t.id:>4}  {t.symbol:<10}  {clr(t.signal, sig_color):<11}  "
                 f"{t.confidence:<7}  {size:>6}  {entry:>8}  {t.status:<7}  {realized}")
        return True

    if sub == "open":
        if len(parts) < 4:
            err("Usage: /trading paper open <SYMBOL> <SIGNAL> <CONFIDENCE> [size%] [stop%] [tp%]")
            err("  Example: /trading paper open AAPL BUY High 3.5 7 15")
            return True
        try:
            symbol = parts[1].upper()
            signal = parts[2].upper()
            confidence = parts[3].capitalize()
            size_pct = float(parts[4]) if len(parts) > 4 else None
            stop_pct = float(parts[5]) if len(parts) > 5 else None
            tp_pct = float(parts[6]) if len(parts) > 6 else None
        except ValueError as e:
            err(f"Could not parse: {e}")
            return True

        # Try to fetch current price as entry
        from .data import fetchers
        price_info = fetchers.fetch_current_price(symbol)
        entry = price_info.get("price") if isinstance(price_info, dict) else None

        try:
            tid = paper_trader.open_trade(
                symbol=symbol, signal=signal, confidence=confidence,
                entry_price=entry, position_size_pct=size_pct,
                stop_loss_pct=stop_pct, take_profit_pct=tp_pct,
            )
            ok(f"Paper trade #{tid} opened: {signal} {symbol} @ "
               f"{f'${entry:.2f}' if entry else 'no-price'} "
               f"({confidence} confidence)")
        except ValueError as e:
            err(str(e))
        return True

    if sub == "close":
        if len(parts) < 2:
            err("Usage: /trading paper close <id> [price]")
            return True
        try:
            tid = int(parts[1])
        except ValueError:
            err(f"Invalid trade id: {parts[1]}")
            return True
        if len(parts) >= 3:
            try:
                price = float(parts[2])
            except ValueError:
                err(f"Invalid price: {parts[2]}")
                return True
        else:
            t = paper_trader.get_trade(tid)
            if not t:
                err(f"Trade #{tid} not found.")
                return True
            from .data import fetchers
            pi = fetchers.fetch_current_price(t.symbol)
            price = pi.get("price") if isinstance(pi, dict) else None
            if not price:
                err(f"Could not fetch current price for {t.symbol}. Specify manually.")
                return True
        result = paper_trader.close_trade(tid, close_price=price)
        if result is None:
            err(f"Trade #{tid} not found or already closed.")
            return True
        sign = "+" if (result.realized_return_pct or 0) > 0 else ""
        info(f"Paper trade #{tid} closed: {result.symbol} @ ${price:.2f} → "
             f"{sign}{result.realized_return_pct:.2f}% realized")
        return True

    if sub == "update":
        # Refresh snapshots for all open trades
        from .data import fetchers
        opens = paper_trader.list_trades(status="open")
        if not opens:
            info("No open paper trades to update.")
            return True
        info(f"Updating {len(opens)} open trades...")
        for t in opens:
            pi = fetchers.fetch_current_price(t.symbol)
            price = pi.get("price") if isinstance(pi, dict) else None
            if not price:
                warn(f"  {t.symbol}: no price")
                continue
            unreal = paper_trader.add_snapshot(t.id, price)
            sign = "+" if (unreal or 0) > 0 else ""
            info(f"  #{t.id} {t.symbol}: ${price:.2f}  {sign}{unreal:.2f}%")
        return True

    if sub == "summary":
        s = paper_trader.open_position_summary()
        info(clr("\nOpen Position Summary", "bold"))
        info(f"  Open trades: {s['open_count']}")
        info(f"  Total exposure: {s['total_exposure_pct']:.1f}%")
        if s["by_sector_pct"]:
            info(f"  By sector:")
            for sec, pct in sorted(s["by_sector_pct"].items(), key=lambda x: -x[1]):
                info(f"    {sec:<20} {pct:.1f}%")
        return True

    err(f"Unknown paper subcommand: {sub}")
    info("  Subcommands: list [open|closed], open, close, update, summary")
    return True


# ── Calibration ───────────────────────────────────────────────────────────

def _cmd_calibration() -> bool:
    from . import calibration
    stats = calibration.compute_calibration()
    report = calibration.render_calibration_report(stats)
    print(report)
    return True


# ── Watchlist ─────────────────────────────────────────────────────────────

def _cmd_watch(args: str) -> bool:
    """Manage trading watchlist."""
    from . import paper_trader

    parts = args.split() if args else []
    sub = parts[0].lower() if parts else "list"

    if sub == "list" or sub == "":
        wl = paper_trader.watchlist_list()
        if not wl:
            info("Watchlist empty. Add symbols with `/trading watch add SYM[,SYM,...]`.")
            return True
        info(clr("\nWatchlist:", "bold"))
        for entry in wl:
            note = f" — {entry['note']}" if entry["note"] else ""
            info(f"  {entry['symbol']:<10}  added {entry['added_at'][:10]}{note}")
        return True

    if sub == "add":
        if len(parts) < 2:
            err("Usage: /trading watch add <SYM[,SYM,...]> [note]")
            return True
        syms_arg = parts[1]
        note = " ".join(parts[2:]) if len(parts) > 2 else None
        added = []
        for sym in syms_arg.replace(",", " ").split():
            paper_trader.watchlist_add(sym, note=note)
            added.append(sym.upper())
        ok(f"Added {len(added)} to watchlist: {', '.join(added)}")
        return True

    if sub == "remove" or sub == "rm":
        if len(parts) < 2:
            err("Usage: /trading watch remove <SYM[,SYM,...]>")
            return True
        removed, missing = [], []
        for sym in parts[1].replace(",", " ").split():
            if paper_trader.watchlist_remove(sym):
                removed.append(sym.upper())
            else:
                missing.append(sym.upper())
        if removed:
            ok(f"Removed: {', '.join(removed)}")
        if missing:
            warn(f"Not in watchlist: {', '.join(missing)}")
        return True

    err(f"Unknown watch subcommand: {sub}")
    info("  Subcommands: list, add <SYM[,SYM,...]>, remove <SYM[,SYM,...]>")
    return True


# ── Scan (run analyze on whole watchlist) ─────────────────────────────────

def _cmd_scan(args: str, state, config) -> bool:
    """Scan watchlist (or specified symbols) and run analyze on each."""
    from . import paper_trader
    from .data import fetchers, indicators

    parts = args.split() if args else []
    if parts:
        symbols = [s.upper() for s in parts[0].replace(",", " ").split()]
    else:
        wl = paper_trader.watchlist_list()
        if not wl:
            err("Watchlist is empty. Add symbols with `/trading watch add SYM[,SYM,...]`"
                " or pass them: `/trading scan AAPL,NVDA,SPY`")
            return True
        symbols = [w["symbol"] for w in wl]

    info(clr(f"\nScanning {len(symbols)} symbol(s):", "bold"))
    info(f"{'Symbol':<10}  {'Price':>10}  {'RSI':>6}  {'vs SMA50':>10}  {'vs SMA200':>10}  Signal")
    info("─" * 70)

    for sym in symbols:
        result = fetchers.fetch_market_data(sym, interval="1d")
        if result.get("error") or not result.get("data"):
            warn(f"  {sym:<10}  (data error)")
            continue
        rows = result["data"]
        if len(rows) < 200:
            warn(f"  {sym:<10}  (insufficient history: {len(rows)} bars)")
            continue
        closes = [r["close"] for r in rows]
        latest = closes[-1]

        try:
            ind = indicators.compute_all(rows)
        except Exception as e:
            warn(f"  {sym:<10}  (indicator error: {e})")
            continue

        rsi = ind.get("rsi", [None])[-1]
        sma50 = ind.get("sma_50", [None])[-1]
        sma200 = ind.get("sma_200", [None])[-1]

        rsi_str = f"{rsi:.1f}" if rsi is not None else "—"
        sma50_pct = ((latest / sma50 - 1) * 100) if sma50 else None
        sma200_pct = ((latest / sma200 - 1) * 100) if sma200 else None
        sma50_str = f"{sma50_pct:+.1f}%" if sma50_pct is not None else "—"
        sma200_str = f"{sma200_pct:+.1f}%" if sma200_pct is not None else "—"

        # Quick heuristic signal — NOT a recommendation, just a coarse filter
        # to highlight which names might warrant a full /trading analyze.
        if (rsi is not None and sma50 and sma200
            and latest > sma50 > sma200 and rsi < 70):
            signal = clr("watch (uptrend)", "green")
        elif (rsi is not None and sma50 and sma200
              and latest < sma50 < sma200 and rsi > 30):
            signal = clr("avoid (downtrend)", "red")
        elif rsi is not None and rsi > 75:
            signal = clr("overbought", "yellow")
        elif rsi is not None and rsi < 25:
            signal = clr("oversold", "yellow")
        else:
            signal = clr("neutral", "dim")
        info(f"  {sym:<10}  ${latest:>9.2f}  {rsi_str:>6}  {sma50_str:>10}  {sma200_str:>10}  {signal}")

    info("")
    info(clr("Note: this is a coarse heuristic filter — run `/trading analyze "
             "<SYMBOL>` for a real multi-agent recommendation.", "dim"))
    return True


# ── Verifier (manual) ─────────────────────────────────────────────────────

def _cmd_verify(args: str) -> bool:
    """Run risk verifier on a hypothetical trade."""
    from . import verifier

    parts = args.split() if args else []
    if len(parts) < 4:
        err("Usage: /trading verify <SYMBOL> <SIGNAL> <SIZE%> <STOP%> [TP%] [SECTOR]")
        err("  Example: /trading verify NVDA BUY 4 7 15 'Information Technology'")
        return True
    try:
        symbol = parts[0].upper()
        signal = parts[1].upper()
        size = float(parts[2])
        stop = float(parts[3])
        tp = float(parts[4]) if len(parts) > 4 else None
        sector = " ".join(parts[5:]).strip("'\"") if len(parts) > 5 else None
    except ValueError as e:
        err(f"Could not parse: {e}")
        return True

    v = verifier.verify_proposal(
        symbol=symbol, signal=signal,
        position_size_pct=size, stop_loss_pct=stop, take_profit_pct=tp,
        sector=sector,
    )
    print(v.as_markdown())
    return True


# ── Walk-forward backtest ─────────────────────────────────────────────────

def _cmd_walkforward(args: str, config) -> bool:
    """Run walk-forward (out-of-sample) backtest."""
    parts = args.split() if args else []
    if not parts:
        err("Usage: /trading walkforward <SYMBOL> [strategy] [--splits N]")
        err("  Example: /trading walkforward AAPL dual_ma --splits 5")
        return True

    symbol = parts[0].upper()
    strategy = parts[1] if len(parts) > 1 and not parts[1].startswith("--") else "dual_ma"
    n_splits = 5
    for i, p in enumerate(parts):
        if p == "--splits" and i + 1 < len(parts):
            try:
                n_splits = int(parts[i + 1])
            except ValueError:
                pass

    from .engines.equity import EquityEngine
    from .engines.base import BacktestConfig
    from .data import fetchers
    from .tools import _build_strategy

    info(f"\nWalk-forward backtest: {clr(strategy, 'bold')} on {clr(symbol, 'bold')}, "
         f"{n_splits} splits")

    fetched = fetchers.fetch_market_data(symbol, interval="1d")
    if fetched.get("error") or not fetched.get("data"):
        err(f"Data error: {fetched.get('error', 'no data')}")
        return True
    data_map = {symbol: fetched["data"]}

    try:
        signal_engine = _build_strategy(strategy)
    except Exception as e:
        err(f"Could not build strategy '{strategy}': {e}")
        return True

    engine = EquityEngine(BacktestConfig(initial_capital=100_000.0), market="us")
    result = engine.walk_forward(signal_engine, data_map, n_splits=n_splits)

    splits = result["splits"]
    if not splits:
        warn(result["stability"].get("verdict", "No splits produced."))
        return True

    info(f"\n{'Split':<6}  {'Window':<24}  {'Return %':>9}  {'Sharpe':>7}  {'Max DD%':>8}  Trades")
    info("─" * 75)
    for s in splits:
        m = s["metrics"]
        info(f"  {s['split']:<4}  {s['start_date'][:10]} → {s['end_date'][:10]}  "
             f"{m['total_return']:>+9.2f}  {m['sharpe_ratio']:>+7.2f}  "
             f"{m['max_drawdown']:>8.2f}  {s['trade_count']:>5}")

    st = result["stability"]
    info("")
    info(clr("Stability:", "bold"))
    info(f"  Positive chunks: {st['positive_chunks']}/{st['n_splits']} "
         f"({st['return_consistency']:.0f}%)")
    info(f"  Sharpe mean ± σ: {st['sharpe_mean']:+.3f} ± {st['sharpe_stdev']:.3f}, "
         f"min {st['sharpe_min']:+.3f}")
    verdict_color = "green" if "STABLE" in st["verdict"] else \
                    "yellow" if "MIXED" in st["verdict"] or "INCONCLUSIVE" in st["verdict"] else "red"
    info(f"  Verdict: {clr(st['verdict'], verdict_color)}")
    return True


def _cmd_backtest(args: str, state, config) -> bool:
    """Handle /trading backtest."""
    parts = args.split()
    if not parts:
        info("Available strategies:")
        info("  dual_ma          — Dual SMA (20/50) crossover")
        info("  rsi_mean_reversion — RSI 30/70 mean reversion")
        info("  bollinger_breakout — Bollinger Band breakout")
        info("  macd_crossover   — MACD histogram crossover")
        info("")
        info("Usage: /trading backtest <symbol> [strategy] [--capital N]")
        info("  Example: /trading backtest AAPL dual_ma")
        return True

    symbol = parts[0].upper()
    strategy = parts[1] if len(parts) > 1 else "dual_ma"
    capital = 100000

    # Parse --capital flag
    for i, p in enumerate(parts):
        if p == "--capital" and i + 1 < len(parts):
            try:
                capital = float(parts[i + 1])
            except ValueError:
                pass

    info(f"\nRunning backtest: {clr(strategy, 'bold')} on {clr(symbol, 'bold')}")
    info(f"  Capital: ${capital:,.0f}")
    info("")

    from .tools import _run_backtest
    result = _run_backtest(
        {"symbol": symbol, "strategy": strategy, "initial_capital": capital},
        config,
    )
    info(result)
    return True


def _cmd_price(args: str) -> bool:
    """Handle /trading price."""
    if not args:
        err("Usage: /trading price <SYMBOL>")
        return True

    symbol = args.split()[0].upper()
    from .tools import _get_price
    result = _get_price({"symbol": symbol}, {})
    info(result)
    return True


def _cmd_indicators(args: str) -> bool:
    """Handle /trading indicators."""
    if not args:
        err("Usage: /trading indicators <SYMBOL>")
        return True

    symbol = args.split()[0].upper()
    from .tools import _get_technical_indicators
    result = _get_technical_indicators({"symbol": symbol}, {})
    info(result)
    return True


def _cmd_status() -> bool:
    """Show trading memory status."""
    from .agents.memory import get_all_memories
    memories = get_all_memories()

    info(clr("\nTrading Agent Status", "bold"))
    info(f"{'='*40}")
    info(f"\n{'Component':<25} {'Memories':>8}")
    info(f"{'-'*25} {'-'*8}")
    total = 0
    for comp, mem in memories.items():
        count = len(mem)
        total += count
        info(f"  {comp:<23} {count:>6}")
    info(f"{'-'*25} {'-'*8}")
    info(f"  {'Total':<23} {total:>6}\n")

    # Check history
    if _HISTORY_DIR.exists():
        decisions = list(_HISTORY_DIR.glob("*.json"))
        info(f"Past decisions: {len(decisions)}")
    else:
        info("Past decisions: 0")
    return True


def _cmd_history() -> bool:
    """Show past trading decisions."""
    if not _HISTORY_DIR.exists():
        info("No trading history found.")
        return True

    decisions = sorted(_HISTORY_DIR.glob("*.json"), reverse=True)
    if not decisions:
        info("No trading history found.")
        return True

    info(clr("\nTrading Decision History", "bold"))
    info(f"{'='*60}")
    for path in decisions[:20]:
        try:
            record = json.loads(path.read_text())
            ts = record.get("timestamp", "")
            sym = record.get("symbol", "")
            sig = record.get("signal", "")
            sig_clr = "green" if sig in ("BUY", "OVERWEIGHT") else "red" if sig in ("SELL", "UNDERWEIGHT") else "yellow"
            info(f"  {ts}  {sym:<8}  {clr(sig, sig_clr)}")
        except Exception:
            pass
    if len(decisions) > 20:
        info(f"  ... and {len(decisions) - 20} more")
    return True


def _cmd_memory(args: str) -> bool:
    """Handle /trading memory subcommands."""
    parts = args.split() if args else []
    action = parts[0] if parts else "list"
    rest = " ".join(parts[1:])

    from .tools import _trading_memory
    result = _trading_memory(
        {"action": action, "component": rest.split()[0] if rest else "portfolio_manager",
         "query": " ".join(rest.split()[1:]) if len(rest.split()) > 1 else rest},
        {},
    )
    info(result)
    return True


def _show_help() -> bool:
    """Show /trading help."""
    info(clr("\nTrading Agent", "bold"))
    info(f"{'='*50}")
    info("")
    info(clr("Single-name analysis & decisions:", "cyan"))
    info("  /trading analyze <SYMBOL>            Full multi-agent analysis")
    info("                                         (Bull/Bear + Risk + PM, with macro + earnings)")
    info("  /trading verify <SYM> <SIG> <SIZE%> <STOP%> [TP%] [sector]")
    info("                                         Run risk-rule check on a hypothetical trade")
    info("  /trading price <SYMBOL>              Quick price check")
    info("  /trading indicators <SYMBOL>         Technical indicators report")
    info("")
    info(clr("Paper trading & calibration (track-the-record):", "cyan"))
    info("  /trading paper list [open|closed]    List paper trades")
    info("  /trading paper open <SYM> <SIG> <CONF> [size%] [stop%] [tp%]   Open a paper trade")
    info("  /trading paper close <id> [price]    Close a trade (current price if omitted)")
    info("  /trading paper update                Refresh snapshots for all open trades")
    info("  /trading paper summary               Open exposure breakdown")
    info("  /trading calibration                 Hit-rate report by confidence + signal")
    info("")
    info(clr("Watchlist & scanning:", "cyan"))
    info("  /trading watch list                  Show watchlist")
    info("  /trading watch add <SYM[,SYM,...]>   Add to watchlist")
    info("  /trading watch remove <SYM[,...]>    Remove from watchlist")
    info("  /trading scan [SYM,...]              Scan watchlist (or list) — coarse heuristic filter")
    info("")
    info(clr("Discovery & ranking (find candidates automatically):", "cyan"))
    info("  /trading discover [insider|earnings|momentum-quality|sector|all]")
    info("                    [--universe sp100|sectors] [--add-watchlist N]")
    info("                                         Auto-find candidate tickers from multiple sources")
    info("  /trading rank [SYMS] [--no-discovery]  Composite \"what's worth buying NOW\"")
    info("  /trading factors [SYMS]                Raw momentum/quality/low-vol factor scores")
    info("  /trading anomaly [SYMS]                Unusual volume / price gaps / vol spikes")
    info("  /trading monitor scan [--notify telegram slack wechat]")
    info("                                         Anomaly + stop + earnings + insider monitor")
    info("                                         (--notify dispatches to bridges)")
    info("  /trading monitor status                Show last monitor run stats")
    info("")
    info(clr("Agentic research (LLM on top of deterministic tools):", "cyan"))
    info("  /trading agent <natural-language research question>")
    info("                                       Auto-runs discover + factors + macro,")
    info("                                       then LLM synthesizes a focused dossier")
    info("                                       ranked by FIT to your question.")
    info("                                       Examples:")
    info("                                         /trading agent find AI-infra names with insider buying")
    info("                                         /trading agent 3 defensive names with positive momentum")
    info("")
    info(clr("Position management & autonomous mode:", "cyan"))
    info("  /trading review [SYMBOL]             Multi-agent debate on EXISTING positions")
    info("                                         (HOLD / ADD / TRIM / EXIT decisions)")
    info("  /trading manage start <name> <USD>   Create a managed paper portfolio")
    info("  /trading manage step <name> [--dry]  Run one allocation/rebalance cycle")
    info("  /trading manage status <name>        Cash + positions + PnL")
    info("  /trading manage report <name>        Full markdown PnL report")
    info("  /trading manage list                 All managed portfolios")
    info("")
    info(clr("Optimization & ML:", "cyan"))
    info("  /trading optimize [SYMS] [--max-weight 0.20]")
    info("                                         Mean-variance optimal weights")
    info("  /trading ml train                    Train stacker on closed paper trades")
    info("  /trading ml status                   Show trained model info")
    info("")
    info(clr("Backtesting:", "cyan"))
    info("  /trading backtest <SYM> [strat]      Aggregate backtest")
    info("  /trading walkforward <SYM> [strat] [--splits N]")
    info("                                         Out-of-sample walk-forward (more honest)")
    info("")
    info(clr("Memory & history:", "cyan"))
    info("  /trading status                      Memory + decision counts")
    info("  /trading history                     Past LLM decision text (JSON archive)")
    info("  /trading memory [search|clear]       Inspect BM25 memory")
    info("")
    info(clr("Strategies: dual_ma, rsi_mean_reversion, bollinger_breakout, macd_crossover", "dim"))
    return True


# ── Export ─────────────────────────────────────────────────────────────────

COMMAND_DEFS = {
    "trading": {
        "func": _cmd_trading,
        "help": (
            "AI trading agent — analyze, discover (auto-find candidates from "
            "insider clusters / earnings beats / sector rotation / factors), "
            "rank, anomaly detection, market monitor with bridge alerts, "
            "paper-trade tracker, calibration, position review, managed "
            "portfolios ($X→1-week PnL), MV optimization, ML stacker, "
            "alt-data (insider / sentiment / trends), walk-forward backtest",
            [
                "analyze", "review", "verify", "price", "indicators",
                "discover", "rank", "factors", "anomaly", "monitor",
                "paper", "calibration", "watch", "scan",
                "manage", "optimize", "ml",
                "backtest", "walkforward",
                "status", "history", "memory",
            ],
        ),
        "aliases": ["trade"],
    },
}
