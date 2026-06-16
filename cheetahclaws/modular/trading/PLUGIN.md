---
name: trading
version: 3.1.0
description: AI trading agent — multi-agent analysis, automatic candidate discovery (insider/earnings/sector/factor), composite ranking, anomaly detection, market monitor with bridge alerts, paper-trade tracking, calibration, risk verifier, walk-forward backtest, alt-data, mean-variance optimizer, managed portfolios, ML stacker, broker abstraction
author: cheetahclaws
tags: [trading, finance, backtest, agent, calibration, risk, alt-data, ml, optimization]
commands:
  - modular.trading.cmd
dependencies:
  - yfinance
  - rank-bm25
  - scipy
  - scikit-learn
optional_dependencies:
  - lightgbm   # preferred ML backend; sklearn GBC fallback if absent
  - pytrends   # Google Trends; trends block silently disabled if missing
  - ib_insync  # required only for live IBKR mode
homepage: ""
---

# Trading Module v3.1

A research and discipline tool for multi-asset paper trading driven by an
LLM-powered multi-agent analysis pipeline, with empirical guardrails so
you can measure whether the agent's recommendations actually have edge.

## What's new in v3.1 (this release)

### Headline: automatic candidate discovery
Previously you had to feed the agent symbols (`/trading analyze NVDA`).
Now `/trading discover` scans a universe and surfaces candidates from
multiple orthogonal sources:

```bash
/trading discover all                          # all four sources
/trading discover insider                      # SEC EDGAR Form 4 clusters
/trading discover earnings                     # recent ≥10% beats with continuation
/trading discover momentum-quality             # factor intersection
/trading discover sector                       # leading sector ETFs + top holdings
/trading discover all --add-watchlist 10       # auto-add top 10 to watchlist
```

Each source returns a ranked candidate list; the orchestrator merges
across sources and tickers flagged by ≥2 sources get a confidence
bonus. Output is a markdown table you act on with `/trading analyze`.

### Headline: composite ranking
`/trading rank` says "of these N candidates, which deserve your
attention NOW". Combines factor scores (50%) + discovery scores (30%)
+ historical calibration tilt:

```bash
/trading rank                                   # rank S&P 100
/trading rank NVDA,AMD,SPY,QQQ                  # rank a custom set
```

### Headline: real-time-ish market monitor
`/trading monitor scan` runs one full cycle:
  - Anomaly detection on watchlist + open positions (volume spike / price gap / vol spike)
  - Stop-loss + take-profit hits on open paper trades
  - Earnings within blackout window
  - New SEC Form 4 filings since last scan (delta detection)

Optionally pushes alerts to Telegram / Slack / WeChat via existing bridges:

```bash
/trading monitor scan                                    # console output
/trading monitor scan --notify                           # all configured bridges
/trading monitor scan --notify telegram slack            # specific bridges
/trading monitor status                                  # last run stats
```

### Headline: anomaly detector + factor table
`/trading anomaly NVDA,AMD,SPY` — one-shot scan for unusual market behavior.
`/trading factors` — raw momentum/quality/low-vol scores with 24h disk cache.

## What's new in v3 (prior upgrade)

### Headline: managed paper portfolios — "$100, check in a week"
Give the agent a virtual budget and let it allocate + rebalance:

```bash
/trading manage start hundred 100         # $100 virtual portfolio
/trading manage step hundred              # one rebalance cycle (mean-variance)
/trading manage status hundred            # cash + positions + PnL
# wait a week, run `/trading manage step hundred` daily or weekly,
# then…
/trading manage report hundred            # weekly PnL report with equity curve
```

### Headline: position-review framing
Distinct from cold-start `/trading analyze` — this is "given that we already
own X, what now?":

```bash
/trading review            # multi-agent debate on every open paper position
/trading review NVDA       # only this name
```

Output is structured `ACTION ID=… DECISION=HOLD|ADD|TRIM|EXIT …` rows
that can be parsed and persisted programmatically.

### New: alt-data layer
Three sources LLM analysis can actually add value on (vs. classical quant
factors that are already priced in):

| Source     | What it surfaces                                                |
|------------|-----------------------------------------------------------------|
| Insider    | SEC EDGAR Form 4 filings (officers / 10%-holders), free          |
| Sentiment  | LLM-scored yfinance headlines (-10..+10 per headline, aggregated)|
| Trends     | Google Trends 30/90-day search interest (requires pytrends)      |

All three soft-fail to empty blocks if data is unavailable.

### New: mean-variance optimizer
`/trading optimize` runs scipy SLSQP on the watchlist (or a passed set):
long-only, single-name capped (default 20%), optional sector caps. Outputs
target weights for `/trading manage` to execute.

### New: ML stacker
`/trading ml train` builds a LightGBM (or sklearn GBC fallback) classifier
on closed paper trades that learns "did the agent's recommendation
outperform". Once trained, can override the LLM's confidence when its
historical track record argues otherwise.

### New: broker abstraction
- `PaperBroker` (default) — works out of the box, owns named portfolios in SQLite
- `IBKRBroker` (stub) — schema in place; `pip install ib_insync` + IB Gateway
  configuration to enable live trading

## Realistic expectations (read this)

**This is a research and discipline tool, not a money printer.** Public-data
+ LLM analysis does not have predictive edge over quant funds in liquid US
equities. What this module gives you:

- **Information aggregation** faster than reading filings yourself
- **Risk discipline** that doesn't depend on the LLM remembering its own rules
- **Empirical accountability** — calibration tells you when the agent is
  noise, before you risk real money
- **A clean substrate for research** — alt-data + classical factors + LLM
  features lined up in one place ready for stacking

If the calibration report (after 30+ closed paper trades) shows HIGH
confidence doesn't outperform LOW, the agent has no signal — change
prompt / model / feature set, or accept reality before going live.

## All commands

### Discovery & ranking (find candidates automatically)
| Command | Purpose |
|---|---|
| `/trading discover [insider\|earnings\|momentum-quality\|sector\|all]` | Scan a universe and surface candidate tickers |
| `/trading discover ... --add-watchlist N` | Auto-add top N hits to your watchlist |
| `/trading rank [SYMS] [--no-discovery]` | Composite "what's worth buying NOW" |
| `/trading factors [SYMS] [--clear-cache]` | Raw momentum/quality/low-vol factor scores |
| `/trading anomaly [SYMS]` | One-shot anomaly scan (vol spikes, gaps, vol regime) |
| `/trading monitor scan [--notify telegram slack wechat]` | Anomaly + stops + earnings + insider monitor (alerts to bridges) |
| `/trading monitor status` | Last monitor run stats |

### Single-name analysis & decisions
| Command | Purpose |
|---|---|
| `/trading analyze <SYMBOL>` | Multi-agent analysis (now with macro / earnings / insider / sentiment / trends / book) |
| `/trading review [SYMBOL]` | Multi-agent debate on EXISTING positions: HOLD / ADD / TRIM / EXIT |
| `/trading verify <SYM> <SIG> <SIZE%> <STOP%> [TP%] [sector]` | Risk-rule check on a hypothetical trade |
| `/trading price <SYMBOL>` | Quick price |
| `/trading indicators <SYMBOL>` | Technical indicators report |

### Paper trading & calibration
| Command | Purpose |
|---|---|
| `/trading paper list [open\|closed]` | List paper trades |
| `/trading paper open <SYM> <SIG> <CONF> [size%] [stop%] [tp%]` | Manually log a trade |
| `/trading paper close <id> [price]` | Close (auto-fetch price if omitted) |
| `/trading paper update` | Refresh snapshots for all open trades |
| `/trading paper summary` | Open exposure breakdown |
| `/trading calibration` | Hit-rate report by confidence + signal |
| `/trading watch add\|remove\|list` | Watchlist management |
| `/trading scan [SYM,...]` | Heuristic scan of watchlist |

### Managed portfolios (autonomous mode)
| Command | Purpose |
|---|---|
| `/trading manage start <name> <USD>` | Create a virtual portfolio with starting cash |
| `/trading manage step <name> [--dry]` | One MV-optimised rebalance cycle |
| `/trading manage status <name>` | Cash + positions + unrealized PnL |
| `/trading manage report <name>` | Full markdown PnL report with equity curve |
| `/trading manage list` | All managed portfolios |

### Optimization & ML
| Command | Purpose |
|---|---|
| `/trading optimize [SYMS] [--max-weight 0.20]` | Mean-variance optimal weights |
| `/trading ml train` | Train LightGBM stacker on closed paper trades |
| `/trading ml status` | Show trained model info |

### Backtesting
| Command | Purpose |
|---|---|
| `/trading backtest <SYM> [strat]` | In-sample backtest (kept for compatibility) |
| `/trading walkforward <SYM> [strat] [--splits N]` | Out-of-sample walk-forward (preferred) |

### Memory & history
| Command | Purpose |
|---|---|
| `/trading status` | Memory + decision counts |
| `/trading history` | Past LLM decision text |
| `/trading memory [search\|clear]` | Inspect BM25 memory |

## Storage layout

| Path | Purpose |
|---|---|
| `~/.cheetahclaws/trading/paper_trades.db` | Paper trades + watchlist + snapshots |
| `~/.cheetahclaws/trading/managed_portfolios.db` | Managed portfolios (cash, positions, equity curve, orders) |
| `~/.cheetahclaws/trading/history/*.json` | Per-decision LLM output archive |
| `~/.cheetahclaws/trading/memory/*.json` | BM25 reflections |
| `~/.cheetahclaws/trading/ml/stacker.pkl` | Trained ML stacker model |

## Recommended workflow

1. **Bootstrap your universe** — let the agent find candidates:
   ```
   /trading discover all --add-watchlist 15
   ```
   This populates your watchlist with the top 15 tickers across insider clusters,
   earnings beats, factor combos, and sector leaders.
2. **Refine watchlist** (optional manual additions):
   ```
   /trading watch add NVDA,AMD,TSM            # add names you specifically care about
   ```
3. **Daily**: `/trading rank` shows the top-N to focus on; `/trading scan` for a
   coarse heuristic filter; `/trading analyze <SYM>` on the most interesting names
   (auto-records as paper trade).
3. **Weekly**: `/trading review` to revisit existing positions; close winners
   and stops.
4. **Monthly**: `/trading calibration` to see whether the agent's confidence
   carries any signal. If yes, `/trading ml train` to lock in features.
5. **Run a managed portfolio**: `/trading manage start hundred 100`, then
   `/trading manage step hundred` daily. `/trading manage report hundred` at
   week's end. Don't move to real money until the managed portfolio's PnL
   trajectory + calibration tell a consistent positive story over 3+ months.
6. **Going live (eventually)**: install `ib_insync`, configure IB Gateway,
   wire `IBKRBroker` to one of your managed portfolios. The abstraction
   layer makes this swap clean.

## Honest disclaimer

Paper trades ≠ real trades. Live execution adds slippage, partial fills,
unfavourable fills on small accounts, broker commissions, taxes, and
emotion. Do not run this with real money until calibration + walk-forward
are both consistently green for at least 3 months.
