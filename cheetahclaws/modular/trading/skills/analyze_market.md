---
name: trading-analyze
description: "Run full multi-agent trading analysis on a stock or crypto symbol"
user-invocable: true
triggers: ["/trading-analyze", "/analyze-stock"]
tools: [GetMarketData, GetPrice, GetTechnicalIndicators, GetFundamentals, GetNews, TradingMemory]
when_to_use: "Use when the user wants a comprehensive trading analysis including bull/bear debate, risk assessment, and portfolio manager decision."
argument-hint: "<SYMBOL>"
arguments: [SYMBOL]
context: inline
---

# Multi-Agent Trading Analysis

You are orchestrating a multi-agent trading analysis system. Execute the full pipeline for **$SYMBOL**.

## Step 1: Data Collection

Use tools to gather data:
1. Call `GetTechnicalIndicators` for $SYMBOL
2. Call `GetFundamentals` for $SYMBOL
3. Call `GetNews` for $SYMBOL
4. Call `TradingMemory` with action="search" and query="$SYMBOL" to check past decisions

## Step 2: Bull Researcher

Build a compelling BULLISH case:
- Growth catalysts and competitive advantages
- Bullish technical signals (support levels, momentum, moving average crossovers)
- Strong financials (PE, EPS growth, margins)
- Positive news catalysts

End with: **BULL VERDICT**: [Strong Buy / Buy / Lean Buy] — [one-sentence thesis]

## Step 3: Bear Researcher

Build a compelling BEARISH case:
- Downside risks and vulnerabilities
- Bearish technical signals (resistance, divergences)
- Fundamental concerns (overvaluation, debt, margin compression)
- Negative catalysts

End with: **BEAR VERDICT**: [Strong Sell / Sell / Lean Sell] — [one-sentence thesis]

## Step 4: Research Judge

Evaluate both cases objectively:
- Which side presented stronger, data-backed arguments?
- **DECISION**: BUY / SELL / HOLD
- Include investment plan: entry, position size, stop loss, take profit

## Step 5: Risk Management Panel

Three perspectives:
- **Aggressive**: Argue for larger position, cite upside
- **Conservative**: Argue for risk protection, cite downside
- **Neutral**: Balanced view with optimal sizing

## Step 6: Portfolio Manager Final Decision

**RATING**: [BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL]

**Executive Summary**: 2-3 sentence thesis
**Action Plan**: Entry, size, time horizon, stop loss, take profit
**Key Risks**: Top 3
**Conviction**: High / Medium / Low
