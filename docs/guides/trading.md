# Trading Agent

<div align=center>
<img src="../media/demos/trading_demo.gif" width="850"/>
</div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Trading Agent: SSJ → multi-agent analysis (Bull/Bear debate + Risk panel + PM decision) → backtest → indicators</center>
</div>

CheetahClaws includes a built-in AI-powered trading research module that combines multi-agent debate, technical/fundamental analysis, alternative-data signals (insider trades, LLM-scored news sentiment, Google Trends), persistent paper-trade tracking with calibration metrics, mean-variance portfolio optimization, walk-forward backtesting, an ML stacker that learns from the agent's own track record, and a broker abstraction layer that's paper-trading-ready out of the box and IBKR-ready when you decide to go live.

> **Read this first**: this module is a research and discipline tool, not a money printer. Public-data + LLM analysis does not have predictive edge over quant funds in liquid US equities. What it gives you is faster information aggregation, programmatic risk discipline, and empirical accountability — `/trading calibration` will tell you whether the agent's confidence labels carry signal *before* you risk real money. Run paper for ≥ 3 months with green calibration + walk-forward before considering an IBKR live account.

## Quick start

```bash
# 1. Install dependencies (lightgbm + scipy + sklearn come along)
pip install "cheetahclaws[trading]"

# 2. Start CheetahClaws
cheetahclaws

# 3a. Single-name analysis (auto-records as a paper trade)
[myproject] » /trading analyze NVDA

# 3b. "$100, check in a week" — the canonical autonomous mode
[myproject] » /trading watch add NVDA,AMD,SPY,QQQ,XLE
[myproject] » /trading manage start hundred 100        # virtual $100 portfolio
[myproject] » /trading manage step hundred             # MV optimiser allocates
# ... days later ...
[myproject] » /trading manage step hundred             # rebalance
[myproject] » /trading manage report hundred           # markdown PnL + equity curve

# 3c. Empirical accountability — is the agent any good?
[myproject] » /trading calibration
```

---

## Features overview

| Feature | Description |
|---|---|
| **Multi-agent analysis** | Bull/Bear debate → Research Judge → Risk Panel → Portfolio Manager |
| **Position review** | `/trading review` — multi-agent debate on EXISTING positions: HOLD / ADD / TRIM / EXIT |
| **Macro context** | SPY/QQQ trend, VIX regime, 10y-yield headwind auto-injected into every prompt |
| **Earnings awareness** | 🚨 blackout flag if earnings within 7 days; warning at 7-30 days |
| **Alt-data: insider** | SEC EDGAR Form 4 filings (officers / 10%-holders), free, no API key |
| **Alt-data: sentiment** | LLM-scored yfinance headlines (-10..+10 per headline, aggregated to regime) |
| **Alt-data: trends** | Google Trends 30/90-day search interest (requires `pytrends`, soft-fails) |
| **Paper trading** | SQLite-backed persistent tracker; long *and* short signal accounting |
| **Calibration metrics** | Hit rate by confidence + signal; t-stat vs zero baseline |
| **Risk verifier** | Hard guardrails: position cap, sector cap, total exposure, stop discipline, earnings blackout |
| **Watchlist + scan** | `/trading watch add NVDA,AMD,…` + `/trading scan` heuristic filter |
| **Managed portfolios** | `"$100, check in a week"` — autonomous MV-driven allocation, equity curve, PnL report |
| **Mean-variance optimizer** | scipy SLSQP, long-only, single-name + sector caps |
| **ML stacker** | LightGBM (sklearn fallback) learning from closed paper trades |
| **Broker abstraction** | `PaperBroker` (works) + `IBKRBroker` (stub for live trading) |
| **Walk-forward backtest** | OOS rolling-chunk evaluation; STABLE / MIXED / FRAGILE verdict |
| **Technical indicators** | 11 indicators: SMA, EMA, MACD, RSI, Bollinger, ATR, VWAP, OBV, ADX, Stochastic, WMA |
| **Fundamental analysis** | P/E, EPS, revenue, margins, ROE, debt/equity, beta, 52-week range |
| **BM25 memory + reflection** | Past decisions retrieved per analysis; post-trade lessons stored |
| **Multi-market** | US, HK, A-share, crypto (20+ coins) |

---

## Slash commands

### Discovery & ranking (find candidates automatically)
| Command | Description |
|---|---|
| `/trading discover [insider\|earnings\|momentum-quality\|sector\|all]` | Scan a universe and surface candidate tickers |
| `/trading discover ... --add-watchlist N` | Auto-add top N hits to watchlist |
| `/trading rank [SYMS] [--no-discovery]` | Composite "what's worth buying NOW" ranking |
| `/trading factors [SYMS] [--clear-cache]` | Raw momentum / quality / low-vol scores |
| `/trading anomaly [SYMS]` | One-shot anomaly scan (vol spikes, price gaps, vol regime) |
| `/trading monitor scan [--notify telegram slack wechat]` | Periodic monitor + alert dispatch |
| `/trading monitor status` | Last monitor run stats |

### Single-name analysis & decisions
| Command | Description |
|---|---|
| `/trading analyze <SYMBOL>` | Full multi-agent analysis (auto-records a paper trade) |
| `/trading review [SYMBOL]` | Multi-agent debate on **existing** positions: HOLD / ADD / TRIM / EXIT |
| `/trading verify <SYM> <SIG> <SIZE%> <STOP%> [TP%] [sector]` | Risk-rule check on a hypothetical trade |
| `/trading price <SYMBOL>` | Quick current price |
| `/trading indicators <SYMBOL>` | Technical indicators report |

### Paper trading & calibration
| Command | Description |
|---|---|
| `/trading paper list [open\|closed]` | List paper trades |
| `/trading paper open <SYM> <SIG> <CONF> [size%] [stop%] [tp%]` | Manually log a trade |
| `/trading paper close <id> [price]` | Close (auto-fetches price if omitted) |
| `/trading paper update` | Refresh snapshots for all open trades |
| `/trading paper summary` | Open exposure breakdown by sector |
| `/trading calibration` | Hit-rate report by confidence + signal + t-stat |
| `/trading watch add\|remove\|list <SYM[,SYM,…]>` | Watchlist management |
| `/trading scan [SYM,…]` | Coarse heuristic filter (RSI / 50d / 200d) on watchlist |

### Managed portfolios (autonomous)
| Command | Description |
|---|---|
| `/trading manage start <name> <USD>` | Create a virtual portfolio with starting cash |
| `/trading manage step <name> [--dry]` | One MV-optimised rebalance cycle |
| `/trading manage status <name>` | Cash + positions + unrealized PnL |
| `/trading manage report <name>` | Full markdown PnL report with equity curve |
| `/trading manage list` | All managed portfolios |

### Optimization & ML
| Command | Description |
|---|---|
| `/trading optimize [SYMS] [--max-weight 0.20]` | Mean-variance optimal weights |
| `/trading ml train` | Train LightGBM stacker on closed paper trades |
| `/trading ml status` | Show trained model info |

### Backtesting
| Command | Description |
|---|---|
| `/trading backtest <SYM> [strategy]` | In-sample backtest (kept for compatibility) |
| `/trading walkforward <SYM> [strategy] [--splits N]` | Out-of-sample walk-forward (preferred) |

### Memory & history
| Command | Description |
|---|---|
| `/trading status` | Memory + decision counts |
| `/trading history` | Past LLM decision text |
| `/trading memory [search\|clear]` | Inspect BM25 memory |

Alias: `/trade` works the same as `/trading`.

---

## AI tools (callable by the model)

The trading module registers 7 tools that the AI can invoke autonomously:

| Tool | Description | Read-only |
|---|---|---|
| `GetMarketData` | Fetch OHLCV data for any symbol (US/HK/A-share/crypto) | Yes |
| `GetPrice` | Current price and basic metrics | Yes |
| `GetTechnicalIndicators` | Compute 11 technical indicators with formatted report | Yes |
| `GetFundamentals` | P/E, EPS, revenue, margins, ROE, market cap, beta | Yes |
| `GetNews` | Recent news articles for a symbol | Yes |
| `RunBacktest` | Execute a backtest with a built-in strategy | Yes |
| `TradingMemory` | List, search, or clear trading agent memories | No |

---

## Multi-agent analysis pipeline

When you run `/trading analyze NVDA`, the system executes a 5-phase pipeline:

```
Phase 1: Data Collection
  ├── Technical Analysis  → SMA, EMA, MACD, RSI, Bollinger, ATR, OBV, ADX, ...
  ├── Fundamental Analysis → P/E, EPS, revenue, margins, ROE, debt
  └── News Analysis       → Recent articles, sentiment

Phase 2: Bull Researcher
  └── Builds bullish case citing specific data (growth catalysts, technical support)
      Verdict: Strong Buy / Buy / Lean Buy

Phase 3: Bear Researcher
  └── Builds bearish case citing specific data (risks, technical weakness)
      Verdict: Strong Sell / Sell / Lean Sell

Phase 4: Risk Management Panel (3-way debate)
  ├── Aggressive Analyst  → argues for larger position, cites upside
  ├── Conservative Analyst → argues for risk protection, cites downside
  └── Neutral Analyst     → balanced view, optimal sizing

Phase 5: Portfolio Manager (final decision)
  └── RATING: BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL
      + Executive summary, action plan, stop loss, take profit, key risks
```

This design is inspired by [TradingAgents](https://github.com/TauricResearch/TradingAgents), which models real-world trading firm dynamics with specialized roles debating investment decisions.

### BM25 memory integration

Each agent component maintains its own memory store:

- `bull_researcher` — past bullish analyses and outcomes
- `bear_researcher` — past bearish analyses and outcomes
- `trader` — past trade execution decisions
- `risk_judge` — past research arbitration decisions
- `portfolio_manager` — past portfolio decisions

When analyzing a new situation, each agent retrieves the most similar past decisions using BM25 similarity matching. This allows the system to learn from successes and mistakes without retraining or fine-tuning.

Memory is stored at `~/.cheetahclaws/trading/memory/` as JSON files.

---

## Automatic discovery — find candidates without naming them

Previously you had to feed the agent symbols (`/trading analyze NVDA`).
Now `/trading discover` scans a universe (default S&P 100) and surfaces
candidate tickers from four orthogonal sources, then merges + ranks across
all of them. This is the answer to "can the agent automatically find
high-yield potential stocks for me?".

### Sources

| Source | Signal | What it surfaces |
|---|---|---|
| `insider` | Form 4 cluster | Tickers with ≥3 SEC EDGAR Form 4 filings in 30 days (officer / 10%-holder activity) |
| `earnings` | Surprise + drift | Stocks that beat consensus EPS by ≥10% AND haven't faded post-print |
| `momentum-quality` | Factor combo | High momentum (6m return + 50d>200d) AND high quality (ROE + low debt + margins) |
| `sector` | Sector rotation | Top holdings of leading sector ETFs (1m + 3m positive) |

### Usage

```bash
# Run all four sources, ranked
/trading discover all

# Single source
/trading discover insider
/trading discover earnings
/trading discover momentum-quality
/trading discover sector

# Custom universe
/trading discover all --universe sp100         # default
/trading discover all --universe sectors       # just sector ETFs (fast)

# Auto-add top 10 to watchlist (for /trading scan / /trading analyze later)
/trading discover all --add-watchlist 10
```

### Output

```text
# Discovery — 17 unique tickers, 23 total hits

| # | Symbol | Sources | Score | Reasons |
|---:|---|---|---:|---|
| 1 | NVDA | insider · earnings | 2.15 | [insider] 5 Form 4 filings in 30d; [earnings] beat by 18% on 04-23, +12% since |
| 2 | AAPL | momentum-quality · sector | 1.42 | [momentum-quality] mom 0.85, qual 0.76; [sector] #1 in Tech (XLK +4.2% 1m) |
| 3 | XOM  | insider | 0.85 | [insider] 4 Form 4 filings in 30d (verify direction at SEC) |
…
```

Tickers flagged by ≥2 sources get a `+0.5` aggregate-score bonus — multi-source confluence is a much stronger signal than any single source alone.

### What's realistic

- These factors are public knowledge; what the system gives you is **search-cost reduction**, not edge. Instead of manually scanning 100 tickers, you get a 15-name shortlist to deep-analyze with `/trading analyze`.
- Insider direction is **not yet parsed from Form 4 XML** (we count filings, not buys vs sales). The output includes URLs so you can verify in 5 seconds; clusters of buys are bullish, clusters of sales are bearish, mixed is internal disagreement.
- Scan time on S&P 100: ~1-2 minutes (yfinance rate-limited). Factor data is cached for 24h at `~/.cheetahclaws/trading/factors_cache.json`.

---

## Composite ranking — "what's worth investing in NOW"

`/trading rank` is the triage step **after** discovery: given a universe (or your discovered candidates), output a single ranked list combining factor scores + discovery scores + historical agent track record.

```bash
/trading rank                                   # rank S&P 100
/trading rank NVDA,AMD,SPY,QQQ                  # rank a custom set
/trading rank --no-discovery                    # pure factor ranking (faster)
/trading rank --no-calibration                  # ignore historical agent record
```

### Composition

| Component | Weight | What it captures |
|---|---:|---|
| Factor score | 50% | Momentum + quality from `factors.py` |
| Discovery score | 30% | Insider / earnings / momentum-quality / sector signals |
| Calibration tilt | ±10pp | Global tilt based on `/trading calibration` mean realised return |

The output is a markdown table with 1 row per candidate. Use it as a **triage list** — don't blindly buy the top entry; spend `/trading analyze` tokens on the top 3-5 names.

---

## Anomaly detector — find unusual market behavior

`/trading anomaly` runs three independent checks per ticker:

| Check | Trigger | Why |
|---|---|---|
| Volume spike | today vol / 90d median ≥ 2.0× | Institutional accumulation or distribution |
| Price gap | abs(today open − prior close) / prior close ≥ 3% | Material news / earnings / corporate action |
| Vol regime | 5d realised vol z-score ≥ 2.0σ vs 90d distribution | Regime change — often precedes large moves |

```bash
/trading anomaly NVDA,AMD,SPY                  # one-shot scan
/trading anomaly                                # uses your watchlist
```

Output groups hits by anomaly type with severity scores. This is a **flag tool**, not a recommendation: high volume can mean accumulation OR distribution; large gaps can be reversals OR continuation. Pair with `/trading analyze` to figure out which.

---

## Real-time-ish monitor with bridge alerts

`/trading monitor scan` runs one full cycle of:

1. Anomaly detection on watchlist + open positions
2. Stop-loss / take-profit checks on managed-portfolio + paper holdings
3. Earnings within 3 days for any open position
4. **New** SEC Form 4 filings since last scan (delta detection — state persisted in `~/.cheetahclaws/trading/monitor_state.db`)

Alerts have severity (`critical` / `warning` / `info`) and can be dispatched to the existing Telegram / Slack / WeChat bridges:

```bash
/trading monitor scan                                   # console output
/trading monitor scan --notify                          # all configured bridges
/trading monitor scan --notify telegram                 # specific bridge
/trading monitor status                                 # last run diagnostic
```

### How "real-time" is it?

**Honest answer**: not real-time. yfinance prices for free tier are 15-20 min delayed; SEC EDGAR is updated within minutes of filing receipt; news takes longer. Running this more often than every 5-10 minutes is wasted effort.

To run periodically, three options:

```bash
# Option 1: manual — run when you want
/trading monitor scan --notify

# Option 2: external cron (recommended for "fire and forget")
echo '*/15 * * * * cd $HOME && cheetahclaws -c "/trading monitor scan --notify"' | crontab -

# Option 3: cheetahclaws's /monitor system to run as a recurring task
/monitor add "trading_monitor" "/trading monitor scan --notify telegram" 15m
```

### Stop / TP detection

Open paper trades (and managed-portfolio positions) are checked against:
- Stop-loss: emit `🚨 STOP HIT` alert when current price drops past `stop_loss_pct` from entry
- Take-profit: emit `⚠️ TAKE-PROFIT HIT` when current price reaches `take_profit_pct`

Alerts include trade ID, entry, current, % change so you can decide quickly.

---

## Paper trading & calibration

Every `/trading analyze` recommendation is auto-recorded as a paper trade in a SQLite store. After enough closed trades, `/trading calibration` answers the question that the original pipeline could not: **"is the agent any good?"**

### Lifecycle

```bash
[myproject] » /trading analyze NVDA              # auto-opens paper trade #12
[myproject] » /trading paper update              # refresh unrealized PnL
[myproject] » /trading paper close 12            # close at current market
[myproject] » /trading calibration               # hit rate by confidence + signal
```

### Calibration report

```text
# Trading Agent Calibration Report
Closed trades analysed: 47

## By Confidence
| Confidence | N | Hit % | Mean % | Median % | Stdev % |
|---|---:|---:|---:|---:|---:|
| High   | 18 | 66.7 | +4.21 | +3.50 | 5.10 |
| Medium | 21 | 52.4 | +1.05 | +0.80 | 4.20 |
| Low    | 8  | 37.5 | -1.50 | -2.00 | 3.80 |

## Diagnosis
✓ High-conviction outperforms Low (signal present) · ✓ High > Medium

## Edge vs zero baseline
BUY mean = +3.10%, t = 2.14 — looks real (one-sided p<0.05)
```

If after 30+ closed trades High doesn't outperform Low (or t-stat < 1.65), the agent's confidence label is noise — change prompt, change model, or accept reality before going live.

### Storage

| Path | Contents |
|---|---|
| `~/.cheetahclaws/trading/paper_trades.db` | Trades + snapshots + watchlist (SQLite) |

---

## Position review (incremental decisions)

`/trading review` is distinct from cold-start `/trading analyze` — it's "given that we already own X, what now?". Multi-agent debate evaluates each open position and emits structured `ACTION` rows:

```bash
[myproject] » /trading review
# Output (Phase 4 of the multi-agent pipeline):
ACTION ID=12 SYMBOL=NVDA DECISION=TRIM SIZE_DELTA=-50% NEW_STOP=2% REASON=+22% on entry, locking gains.
ACTION ID=15 SYMBOL=AMD DECISION=EXIT SIZE_DELTA=-100% NEW_STOP=N/A REASON=Closed below 50d.
ACTION ID=18 SYMBOL=SPY DECISION=HOLD SIZE_DELTA=0% NEW_STOP=same REASON=Thesis intact.
```

The structured output is grepable + parseable for downstream automation.

---

## Managed portfolios — "$100, check in a week"

This is the headline autonomous mode. Give the agent a virtual budget; it allocates and rebalances using the mean-variance optimiser over your watchlist, snapshots an equity curve, and produces a weekly markdown report.

### Lifecycle

```bash
[myproject] » /trading watch add NVDA,AMD,SPY,QQQ,XLE,XLF
[myproject] » /trading manage start hundred 100
  Portfolio 'hundred' created with $100.00.

[myproject] » /trading manage step hundred
  Universe: NVDA, AMD, SPY, QQQ, XLE, XLF
  Target weights: {'NVDA': 0.20, 'SPY': 0.20, 'QQQ': 0.20, 'XLF': 0.10}
  Placed 4 order(s).
  Equity: $100.00 → $99.98  (-$0.02)        # within rounding noise

# ... a few days later (run daily or weekly) ...
[myproject] » /trading manage step hundred                  # rebalance

# End of week: the report
[myproject] » /trading manage report hundred
# 🟢 Managed portfolio: `hundred`
**Initial**: $100.00   →   **Now**: $103.42   (+$3.42, +3.42%)
**Cash**: $0.50   |   **Open positions**: 4
## Holdings
| Symbol | Qty | Avg cost | Last | Market value | Unrealized |
|---|---:|---:|---:|---:|---:|
| NVDA | 0.0212 | $945.10 | $980.20 | $20.78 | +$0.74 |
…
```

### How it works

1. **Universe** = your watchlist (or default ETF basket if empty)
2. **Mean-variance optimisation** over the universe (long-only, single-name capped)
3. **Sells** any holdings not in the new target set
4. **Buys/sells** to bring each held name to its target dollar weight
5. **Snapshots** cash + market value to the equity curve
6. **Skips** trades smaller than 2% of equity (avoids commission grind)

### Multiple portfolios in parallel

You can run several at once — each with its own name, cash balance, and equity curve:

```bash
/trading manage start retire     5000        # different risk profile
/trading manage start crypto-only 200
/trading manage start hundred    100
/trading manage list                          # all of them
```

### Storage

| Path | Contents |
|---|---|
| `~/.cheetahclaws/trading/managed_portfolios.db` | Portfolios, positions, orders, equity curves |

### Honest limits

- yfinance prices are **15-20 min delayed** for the free tier — this is on-demand re-evaluation, not real-time HFT
- `step` doesn't run on a schedule by itself — invoke manually, or wire it via `/monitor` / cron
- Paper has no slippage / commission; real $100 accounts are uneconomic in live trading (fixed costs eat returns)

---

## Alternative-data layer

Three sources LLM analysis can actually add value on (vs. classical quant factors which are already priced in by quant funds):

### Insider trades (SEC EDGAR Form 4, free)

Officer / 10%-holder buys & sells. Cluster of buys = strong signal; sales alone are noise (taxes, diversification).

```text
## Insider Activity (NVDA, last 90 days)
- 4 Form 4 filing(s) by officers / 10%-holders
  - 2026-04-12 (4):   https://www.sec.gov/Archives/edgar/data/.../doc.html
  - 2026-04-15 (4):   …
**How to use**: cluster of buys by multiple officers within a short window =
strong signal. Sales alone are noise.
```

### News sentiment (LLM-scored)

The auxiliary cheap model scores each yfinance headline -10..+10. Aggregate rolls up to BULLISH / MIXED / BEARISH regime.

```text
## News Sentiment (NVDA)
- Headlines analysed: 8
- Aggregate score: **+3.2/10** → **BULLISH** (5 bullish, 1 bearish)
- Headlines:
  - NVDA beats Q4 estimates (Reuters) `[+7]`
  - NVDA faces antitrust probe (WSJ) `[-4]`
  …
```

### Google Trends search interest (optional)

Requires `pip install pytrends`. Soft-fails if not installed.

```text
## Google Trends (NVDA)
- Search interest: SPIKE — public attention surge (latest 95, median 45, p90 80, 7-day +20)
- ⚠ Retail attention spikes precede mean-reversion more often than continuation.
```

All three blocks are auto-injected into the `/trading analyze` prompt — the LLM sees them alongside technicals/fundamentals/news.

---

## Mean-variance portfolio optimizer

`/trading optimize` runs scipy SLSQP on the watchlist (or a passed set of symbols): long-only, single-name capped (default 20%), optional sector caps.

```bash
[myproject] » /trading optimize NVDA,AMD,SPY,QQQ,XLE --max-weight 0.25

# Portfolio Optimization (Mean-Variance, Long-Only)
**Expected annual return**: +18.30%
**Expected annual vol**:    21.50%
**Sharpe**:                 +0.665
**Invested**:               80.0%   (cash 20.0%)

## Target weights
| Symbol | Weight |
|---|---:|
| NVDA | 25.0% |
| QQQ  | 25.0% |
| SPY  | 20.0% |
| AMD  | 10.0% |
```

The managed-portfolio mode uses this internally to set target weights at every `step`.

---

## Walk-forward backtest

`/trading walkforward` replaces the dishonest aggregate backtest with rolling out-of-sample chunks. Reports per-chunk metrics + a stability verdict so you know whether a strategy is regime-stable or just lucky in one window.

```bash
[myproject] » /trading walkforward AAPL dual_ma --splits 5

Walk-forward backtest: dual_ma on AAPL, 5 splits

Split  Window                    Return %  Sharpe  Max DD%  Trades
─────────────────────────────────────────────────────────────────
  1    2024-05-08 → 2024-08-12    +12.40   +1.23     5.20      3
  2    2024-08-13 → 2024-11-17    -3.10    -0.52    11.40      4
  3    2024-11-18 → 2025-02-23    +8.50    +0.95     6.80      3
  4    2025-02-24 → 2025-05-30    +15.20   +1.41     4.10      2
  5    2025-05-31 → 2026-05-07    +6.30    +0.62     8.20      4

Stability:
  Positive chunks: 4/5 (80%)
  Sharpe mean ± σ: +0.738 ± 0.671, min -0.52
  Verdict: STABLE — strategy works across most regimes.
```

Verdict tiers:

| Verdict | Meaning |
|---|---|
| `STABLE` | ≥70% positive chunks AND min Sharpe > 0 |
| `MIXED` | ≥50% positive AND mean Sharpe > 0.5 — consider regime filter |
| `FRAGILE` | <30% positive — aggregate metrics are misleading |
| `INCONCLUSIVE` | Too few chunks / noisy |

---

## ML stacker

`/trading ml train` builds a LightGBM (or sklearn `GradientBoostingClassifier` fallback) classifier that learns from your closed paper trades: "did this trade beat zero".

Features per trade:
- LLM signal one-hot (BUY/HOLD/SELL/…)
- Confidence ordinal (Low/Med/High)
- Position size, stop loss %, take profit %
- Sector one-hot

```bash
[myproject] » /trading ml train

# Stacker Training Report
- Samples: 60
- Trained lightgbm model on 60 samples, 3-fold CV.

## Cross-validated performance
- AUC: **0.687 ± 0.043**
- Accuracy: **0.621**

## Top features
- confidence: 0.342
- sector__Technology: 0.158
- position_size_pct: 0.121
…

Model saved to: `~/.cheetahclaws/trading/ml/stacker.pkl`
```

If AUC < 0.55, the model has no edge — typically because there are too few samples (need 50+) or the agent's track record is genuinely noisy.

When integrated in future versions, the stacker output becomes a post-filter on `/trading analyze`: if the LLM says BUY but the stacker says p(hit) < 0.4, downgrade to HOLD with a model-disagreement note.

---

## Broker abstraction

The trading module separates "decision" from "execution" via a tiny `BrokerBackend` interface:

```python
class BrokerBackend:
    def account_summary() -> AccountSummary
    def positions() -> list[Position]
    def quote(symbol) -> float | None
    def place_market_order(symbol, side, quantity) -> OrderResult
```

Two backends ship:

| Backend | Mode | Status |
|---|---|---|
| `PaperBroker` | SQLite-backed paper trading | Production-ready |
| `IBKRBroker` | Interactive Brokers (real money) | Stub — see below |

### Going live (IBKR)

The IBKR backend is wired but disabled. To enable:

```bash
# 1. Install
pip install ib_insync

# 2. Install IB Gateway and configure for API access (paper port 7497, live 7496)
#    https://www.interactivebrokers.com/en/trading/ibgateway-stable.php

# 3. Enable "Allow connections from localhost" in IB Gateway settings

# 4. In Python:
from modular.trading.broker import IBKRBroker
b = IBKRBroker(host="127.0.0.1", port=7497, client_id=42, paper=True)
b.connect()
```

> Do not switch to live trading until `/trading calibration` shows HIGH > LOW for ≥3 months AND a managed portfolio's PnL is consistently positive AND walk-forward verdict is STABLE on your strategies.

---

## Backtesting

### Built-in strategies

| Strategy | Logic | Type |
|---|---|---|
| `dual_ma` | SMA(20) vs SMA(50) crossover | Trend following |
| `rsi_mean_reversion` | Buy RSI < 30, sell RSI > 70 | Mean reversion |
| `bollinger_breakout` | Price vs Bollinger Bands(20, 2σ) | Volatility breakout |
| `macd_crossover` | MACD histogram direction | Momentum |

### Usage

```bash
# Backtest a single strategy
/trading backtest AAPL dual_ma

# Compare all strategies (via SSJ)
/ssj → 14 → b → AAPL → 5 (all)

# Or ask the AI directly
> Backtest all 4 strategies on TSLA for the last 2 years and compare
```

### Performance metrics

Each backtest reports:

| Metric | Description |
|---|---|
| Total Return | Cumulative profit/loss percentage |
| Annualized Return | Annualized compound return |
| Sharpe Ratio | Risk-adjusted return (excess return / volatility) |
| Sortino Ratio | Downside risk-adjusted return |
| Calmar Ratio | Return / max drawdown |
| Max Drawdown | Largest peak-to-trough decline |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross profit / gross loss |
| Avg Bars Held | Average holding period per trade |

### SignalEngine contract

The backtesting system uses a standard signal contract inspired by [Vibe-Trading](https://github.com/Vibe-Trading/Vibe-Trading):

```python
# Signal values: -1.0 to 1.0
#  1.0 = fully long  (100% of capital)
#  0.5 = half long   (50%)
#  0.0 = flat        (no position)
# -0.5 = half short  (50% short)
# -1.0 = fully short (100% short)
```

### Backtest engines

| Engine | Markets | Rules |
|---|---|---|
| `EquityEngine` | US stocks, HK stocks | T+0, fractional shares (US), lot-size rounding (HK), stamp tax (HK) |
| `CryptoEngine` | Crypto spot/perpetuals | 24/7 trading, maker/taker fees, funding fees, liquidation checks |

---

## Data sources

### Fallback chains

The data layer automatically tries multiple sources in order:

| Market | Fallback chain |
|---|---|
| US equity | yfinance |
| HK equity | yfinance |
| Crypto | coingecko → yfinance |
| A-share | akshare → yfinance |

### Symbol formats

| Market | Format | Examples |
|---|---|---|
| US stocks | Ticker | `AAPL`, `MSFT`, `NVDA` |
| HK stocks | Code.HK | `0700.HK`, `9988.HK` |
| A-shares | Code.SZ/SH | `000001.SZ`, `600519.SH` |
| Crypto | Symbol | `BTC`, `ETH`, `SOL`, `BTC-USDT` |

### Supported crypto

BTC, ETH, BNB, SOL, XRP, ADA, DOGE, DOT, AVAX, MATIC, LINK, UNI, LTC, ATOM, NEAR, ARB, OP, APT, SUI, SEI

---

## Reflection mechanism

After a trade outcome is known (profit or loss), the reflection system:

1. Analyzes what each agent component got right or wrong
2. Extracts a condensed lesson (~100 words)
3. Stores the lesson in the component's BM25 memory
4. Future analyses retrieve these lessons when facing similar situations

This creates a continuous learning loop without model retraining.

---

## SSJ integration

The trading module is accessible via SSJ Developer Mode (`/ssj` → option 14):

```
╭─ 📈 Trading Agent ━━━━━━━━━━━━━━━━━━━━━━━━━
│
│  a. 🔍  Quick Analyze — Full multi-agent analysis
│  b. 📊  Backtest     — Test a strategy on historical data
│  c. 💰  Price Check  — Current price & key metrics
│  d. 📉  Indicators   — Technical indicators report
│  e. 🤖  Trading Bot  — Launch autonomous trading agent
│  f. 📜  History      — Past trading decisions
│  g. 🧠  Memory       — Trading memory status
│  0. ↩   Back to SSJ
╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Trading Bot** (option e) runs a multi-symbol autonomous analysis. Enter a comma-separated watchlist (default: `AAPL,MSFT,GOOGL,NVDA,BTC,ETH`) and the agent analyzes each symbol through the full pipeline, producing a summary table with ratings.

---

## Autonomous trading agent

Launch via `/agent start trading_agent` or SSJ → 15 → Agent → custom template:

```bash
/agent start trading_agent AAPL,MSFT,GOOGL,NVDA,BTC,ETH
```

The agent iterates through the watchlist, running the full analysis pipeline for each symbol. It maintains a `trading_log.md` with decisions and ratings.

---

## Skills

Three trading skills are available as prompt templates:

| Skill | Trigger | Description |
|---|---|---|
| `trading-analyze` | `/trading-analyze <SYMBOL>` | Full multi-agent analysis |
| `trading-strategy` | `/trading-strategy <desc>` | Generate and backtest a strategy |
| `trading-backtest` | `/trading-backtest <SYMBOL>` | Backtest with comparison table |

---

## Architecture

```
modular/trading/
├── cmd.py                    # /trading command + all subcommands + SSJ sub-menu
├── tools.py                  # AI tools (TOOL_DEFS) + strategy factory
├── paper_trader.py           # SQLite paper-trade store + Phase-5 parser
├── calibration.py            # Hit-rate aggregation + edge-vs-zero t-stat
├── verifier.py               # Hard risk rules: position/sector/stop/earnings caps
├── macro.py                  # SPY/QQQ/VIX/TNX context block (cached 30 min)
├── earnings.py               # yfinance earnings-calendar warnings
├── managed.py                # Managed portfolio orchestrator ($X → step → report)
├── portfolio.py              # scipy SLSQP mean-variance optimiser
├── universe.py               # S&P 100 + sector ETFs + top holdings
├── factors.py                # Momentum / quality / low-vol scoring (24h cache)
├── ranker.py                 # Composite "what's worth buying now" ranking
├── monitor.py                # Periodic monitor + bridge alert dispatch
├── discover/
│   ├── orchestrator.py       # Merge multi-source hits with cross-source bonus
│   ├── insider_cluster.py    # SEC Form 4 cluster detector
│   ├── earnings_beat.py      # Recent ≥10% beat + post-print drift
│   ├── momentum_quality.py   # Factor intersection
│   ├── sector_rotation.py    # Sector ETF leaderboard + top holdings
│   ├── anomaly.py            # Volume / gap / vol-regime anomaly detector
│   └── types.py              # Shared Discovery dataclass
├── alt_data/
│   ├── insider.py            # SEC EDGAR Form 4 fetcher (urllib, no deps)
│   ├── sentiment.py          # LLM-scored yfinance headlines
│   └── trends.py             # Google Trends (pytrends, soft-fails)
├── broker/
│   ├── base.py               # BrokerBackend protocol + OrderResult / AccountSummary
│   ├── paper_backend.py      # SQLite-backed PaperBroker (named portfolios)
│   └── ibkr_backend.py       # IBKR stub + connection_check + setup docs
├── ml/
│   ├── features.py           # Feature engineering from closed trades
│   └── stacker.py            # LightGBM (sklearn fallback) train + predict
├── data/
│   ├── fetchers.py           # Data sources + fallback chains
│   └── indicators.py         # 11 technical indicators (pure Python)
├── engines/
│   ├── base.py               # SignalEngine contract + backtest + walk_forward + metrics
│   ├── equity.py             # US/HK equity engine
│   └── crypto.py             # Crypto engine (spot + perpetual)
├── agents/
│   ├── memory.py             # BM25 memory system
│   ├── analyst.py            # Technical / fundamental / news / sentiment
│   ├── researcher.py         # Bull/Bear debate + research judge
│   ├── risk_manager.py       # Aggressive / conservative / neutral panel
│   ├── portfolio_manager.py  # Final decision + signal extraction
│   └── reflection.py         # Post-trade reflection → memory
├── skills/                   # 3 markdown skill templates
└── agent_templates/          # Autonomous trading agent template
```

---

## Configuration / storage

| Path | Contents |
|---|---|
| `~/.cheetahclaws/trading/paper_trades.db` | Paper trades + snapshots + watchlist |
| `~/.cheetahclaws/trading/managed_portfolios.db` | Managed portfolios (cash, positions, orders, equity curve) |
| `~/.cheetahclaws/trading/ml/stacker.pkl` | Trained ML stacker model |
| `~/.cheetahclaws/trading/factors_cache.json` | 24h-TTL factor data cache |
| `~/.cheetahclaws/trading/monitor_state.db` | Monitor seen-filings tracker + run history |
| `~/.cheetahclaws/trading/memory/` | BM25 memory JSON files (per agent component) |
| `~/.cheetahclaws/trading/history/` | Past trading decision records |

No API keys required for basic usage. yfinance, CoinGecko, and SEC EDGAR are all free. For A-share data, optionally install `akshare`.

### Risk-rule tuning

Default hard limits enforced by the verifier (override via `rules=` arg):

| Rule | Default |
|---|---|
| `max_single_position_pct` | 5% |
| `max_sector_pct` | 25% |
| `max_total_exposure_pct` | 80% |
| `max_stop_loss_pct` | 10% |
| `min_take_profit_pct` | 5% |
| `earnings_blackout_days` | 3 |
| `earnings_blackout_size_pct` | 2.5% |

---

## Dependencies

| Package | Required | Purpose |
|---|---|---|
| `yfinance` | Yes | US/HK stock data, fundamentals, news, earnings calendar |
| `scipy` | Yes | Mean-variance optimiser (SLSQP) |
| `scikit-learn` | Yes | ML stacker (fallback if lightgbm absent) |
| `rank-bm25` | Optional | BM25 memory similarity (falls back to term-overlap) |
| `lightgbm` | Optional | Preferred ML stacker backend (faster, better calibration) |
| `pytrends` | Optional | Google Trends alt-data block |
| `ib_insync` | Optional | Interactive Brokers live trading (only when going live) |
| `akshare` | Optional | A-share, futures, forex data |

Install:

```bash
pip install "cheetahclaws[trading]"           # core trading deps
pip install pytrends                           # add Google Trends
pip install ib_insync                          # add IBKR live trading
```

---

## Recommended workflow

1. **Bootstrap your universe** — let the agent find candidates:
   `/trading discover all --add-watchlist 15`
   This populates the watchlist with the top 15 tickers across all four discovery sources.
2. **Optional manual additions**: `/trading watch add NVDA,AMD,TSM` for specific names.
3. **Daily**: `/trading rank` for the top-N triage list; `/trading analyze <SYM>` on the most interesting names. Each `analyze` auto-records a paper trade.
3. **Weekly**: `/trading review` to revisit existing positions; close stops + winners.
4. **Monthly**: `/trading calibration` to see whether confidence carries signal. Once 30+ closed trades exist, `/trading ml train` to lock features in.
5. **Run a managed portfolio**: `/trading manage start hundred 100`, then `/trading manage step hundred` daily/weekly. `/trading manage report hundred` at week's end.
6. **Going live (eventually)**: `pip install ib_insync`, configure IB Gateway, switch broker to `IBKRBroker`. The abstraction layer makes the swap clean.

## Honest disclaimer

Paper trades ≠ real trades. Live execution adds slippage, partial fills, broker commissions, taxes, and emotion. **Do not run with real money** until calibration + walk-forward + managed-portfolio PnL are all consistently green for at least 3 months. Small accounts (< $1k) have unfavourable fixed-cost economics in real life regardless of strategy.
