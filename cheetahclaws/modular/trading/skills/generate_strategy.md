---
name: trading-strategy
description: "Generate a quantitative trading strategy and backtest it"
user-invocable: true
triggers: ["/trading-strategy", "/gen-strategy"]
tools: [GetMarketData, GetTechnicalIndicators, RunBacktest, Read, Write, Edit, Bash]
when_to_use: "Use when the user wants to create, generate, or design a trading strategy."
argument-hint: "<description of strategy>"
arguments: []
context: inline
---

# Strategy Generation & Backtesting

You are a quantitative strategy developer. Generate a trading strategy based on the user's request, implement it, and backtest it.

## User Request
$ARGUMENTS

## Workflow

### Step 1: Requirements Parsing
- Extract: symbols, date range, strategy logic, risk parameters
- If not specified, default to: AAPL, last 2 years, SMA crossover

### Step 2: Strategy Design
Answer these 5 questions:
1. **Data needs**: What OHLCV data and indicators are required?
2. **Signal logic**: How are buy/sell signals generated? (must output [-1, 1])
3. **Position sizing**: Fixed fraction, equal weight, or volatility-based?
4. **Backtest params**: Initial capital, commission, slippage
5. **Validation**: What metrics determine success? (Sharpe > 1, max DD < 20%, etc.)

### Step 3: Backtest
Use `RunBacktest` tool with the appropriate strategy and symbol.

Available built-in strategies:
- `dual_ma`: SMA(20) vs SMA(50) crossover — trend following
- `rsi_mean_reversion`: RSI 30/70 — mean reversion
- `bollinger_breakout`: Price vs Bollinger Bands(20, 2σ) — volatility breakout
- `macd_crossover`: MACD histogram direction — momentum

### Step 4: Evaluate Results
Check metrics against criteria:
- **Hard gates**: Total trades > 0, equity curve not flat
- **Quality**: Sharpe > 0.5, Max DD < 30%, Win rate > 40%
- **Comparison**: Compare multiple strategies if user is undecided

### Step 5: Report
Present results with:
1. Strategy description and logic
2. Key performance metrics (table)
3. Trade summary (last 10 trades)
4. Strengths and weaknesses
5. Improvement suggestions

## Signal Convention
```
 1.0 = fully long (100% of capital)
 0.5 = half position
 0.0 = flat (no position)
-0.5 = half short
-1.0 = fully short
```
