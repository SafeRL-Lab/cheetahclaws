---
name: trading-backtest
description: "Backtest a trading strategy on historical data with performance metrics"
user-invocable: true
triggers: ["/trading-backtest", "/backtest"]
tools: [RunBacktest, GetMarketData, GetTechnicalIndicators]
when_to_use: "Use when the user wants to backtest a specific strategy on historical data."
argument-hint: "<SYMBOL> [strategy] [--capital N]"
arguments: [SYMBOL]
context: inline
---

# Strategy Backtesting

Run a backtest on **$SYMBOL** with historical data.

## Instructions

1. Use `RunBacktest` to execute the backtest with the specified parameters
2. If the user didn't specify a strategy, run all 4 built-in strategies and compare:
   - `dual_ma` — Dual SMA (20/50) crossover
   - `rsi_mean_reversion` — RSI 30/70 mean reversion
   - `bollinger_breakout` — Bollinger Band breakout
   - `macd_crossover` — MACD histogram crossover

3. Present results as a comparison table:

| Strategy | Return | Sharpe | MaxDD | WinRate | Trades |
|----------|--------|--------|-------|---------|--------|
| dual_ma  | ...    | ...    | ...   | ...     | ...    |
| rsi_mr   | ...    | ...    | ...   | ...     | ...    |
| bb_break | ...    | ...    | ...   | ...     | ...    |
| macd     | ...    | ...    | ...   | ...     | ...    |

4. Recommend the best strategy based on risk-adjusted returns (Sharpe ratio)

5. If the user provides a custom strategy description, explain how to implement it
   using the SignalEngine contract:
   - Input: data_map = {symbol: [OHLCV dicts]}
   - Output: signal_map = {symbol: [float signals in [-1, 1]]}

$ARGUMENTS
