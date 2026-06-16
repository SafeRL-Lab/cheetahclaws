"""
Analyst agents: technical, fundamental, sentiment, and news analysis.

Each analyst takes market data and produces a structured report that feeds
into the researcher debate phase.

These are prompt-based agents — they generate prompts for the LLM to execute.
The actual LLM invocation happens in the orchestrator (cmd.py / tools.py).
"""
from __future__ import annotations

from ..data import fetchers, indicators


# ── Technical Analyst ──────────────────────────────────────────────────────

def run_technical_analysis(symbol: str) -> str:
    """Run technical analysis and return structured report.

    Fetches market data, computes indicators, and formats a report.
    This is a data-driven step (no LLM needed).
    """
    # Fetch data
    result = fetchers.fetch_market_data(symbol, interval="1d")
    if result.get("error"):
        return f"Technical Analysis Error: {result['error']}"

    data = result["data"]
    if not data or len(data) < 30:
        return f"Insufficient data for {symbol} (got {len(data)} bars, need 30+)"

    # Compute indicators
    all_indicators = indicators.compute_all(data)

    # Format report
    report = indicators.format_indicators_report(data, all_indicators)
    return f"# Technical Analysis Report: {symbol}\n\n{report}"


def get_technical_prompt(symbol: str, technical_report: str) -> str:
    """Generate the LLM prompt for technical analysis interpretation."""
    return f"""You are a senior technical analyst at a quantitative trading firm.

Analyze the following technical data for {symbol} and provide your assessment.

{technical_report}

## Your Analysis Should Include:

1. **Trend Assessment**: Is the stock in an uptrend, downtrend, or consolidation?
   - Reference specific moving averages and their crossovers
   - Note ADX strength and directional indicators

2. **Momentum Analysis**:
   - RSI: overbought/oversold conditions, divergences
   - MACD: signal line crossovers, histogram momentum
   - Stochastic: %K/%D crossovers

3. **Volatility Assessment**:
   - Bollinger Band position and bandwidth
   - ATR relative to price (high/low volatility regime)

4. **Volume Analysis**:
   - OBV trend (confirming or diverging from price)
   - VWAP relationship

5. **Key Support/Resistance Levels**: Based on moving averages and Bollinger Bands

6. **Technical Summary Table**:

| Indicator | Value | Signal |
|-----------|-------|--------|
| Trend     | ...   | Bullish/Bearish/Neutral |
| Momentum  | ...   | Bullish/Bearish/Neutral |
| Volatility| ...   | High/Low/Normal |
| Volume    | ...   | Confirming/Diverging |

7. **Overall Technical Rating**: Strong Buy / Buy / Neutral / Sell / Strong Sell
"""


# ── Fundamental Analyst ────────────────────────────────────────────────────

def run_fundamental_analysis(symbol: str) -> str:
    """Fetch and format fundamental data. Returns structured report."""
    fundamentals = fetchers.fetch_fundamentals(symbol)
    if fundamentals.get("error"):
        return f"Fundamental Analysis Error: {fundamentals['error']}"

    lines = [f"# Fundamental Analysis Report: {symbol}"]
    lines.append(f"\n**Company**: {fundamentals.get('name', 'N/A')}")
    lines.append(f"**Sector**: {fundamentals.get('sector', 'N/A')}")
    lines.append(f"**Industry**: {fundamentals.get('industry', 'N/A')}")

    lines.append("\n## Valuation Metrics")
    pe = fundamentals.get("pe_ratio")
    lines.append(f"  P/E Ratio (TTM): {pe:.2f}" if pe else "  P/E Ratio: N/A")
    fpe = fundamentals.get("forward_pe")
    lines.append(f"  Forward P/E: {fpe:.2f}" if fpe else "  Forward P/E: N/A")

    lines.append("\n## Profitability")
    eps = fundamentals.get("eps")
    lines.append(f"  EPS (TTM): ${eps:.2f}" if eps else "  EPS: N/A")
    rev = fundamentals.get("revenue")
    lines.append(f"  Revenue: ${rev:,.0f}" if rev else "  Revenue: N/A")
    pm = fundamentals.get("profit_margin")
    lines.append(f"  Profit Margin: {pm*100:.1f}%" if pm else "  Profit Margin: N/A")
    roe = fundamentals.get("roe")
    lines.append(f"  ROE: {roe*100:.1f}%" if roe else "  ROE: N/A")

    lines.append("\n## Financial Health")
    mcap = fundamentals.get("market_cap")
    lines.append(f"  Market Cap: ${mcap:,.0f}" if mcap else "  Market Cap: N/A")
    dte = fundamentals.get("debt_to_equity")
    lines.append(f"  Debt/Equity: {dte:.2f}" if dte else "  Debt/Equity: N/A")
    beta = fundamentals.get("beta")
    lines.append(f"  Beta: {beta:.2f}" if beta else "  Beta: N/A")

    lines.append("\n## Price Context")
    h52 = fundamentals.get("52w_high")
    l52 = fundamentals.get("52w_low")
    if h52 and l52:
        lines.append(f"  52-Week High: ${h52:,.2f}")
        lines.append(f"  52-Week Low: ${l52:,.2f}")

    dy = fundamentals.get("dividend_yield")
    lines.append(f"  Dividend Yield: {dy*100:.2f}%" if dy else "  Dividend Yield: N/A")

    return "\n".join(lines)


def get_fundamental_prompt(symbol: str, fundamental_report: str) -> str:
    """Generate LLM prompt for fundamental analysis interpretation."""
    return f"""You are a senior fundamental analyst at an investment bank.

Analyze the following fundamental data for {symbol} and provide your assessment.

{fundamental_report}

## Your Analysis Should Include:

1. **Valuation Assessment**: Is the stock overvalued, fairly valued, or undervalued?
   - Compare P/E to industry averages
   - Forward P/E vs trailing (growth expectations)

2. **Financial Health**: Balance sheet strength
   - Debt levels relative to equity
   - Cash flow quality

3. **Profitability**: Earnings quality
   - Margin trends
   - ROE sustainability

4. **Growth Outlook**: Revenue and earnings growth trajectory

5. **Risk Factors**: Key risks identified from fundamentals

6. **Fundamental Rating**: Strong Buy / Buy / Neutral / Sell / Strong Sell
"""


# ── News Analyst ───────────────────────────────────────────────────────────

def run_news_analysis(symbol: str) -> str:
    """Fetch and format recent news. Returns structured report."""
    news = fetchers.fetch_news(symbol, limit=10)
    if news.get("error"):
        return f"News Analysis Error: {news['error']}"

    items = news.get("news", [])
    if not items:
        return f"# News Analysis: {symbol}\n\nNo recent news found."

    lines = [f"# News Analysis Report: {symbol}\n"]
    lines.append(f"Found {len(items)} recent news articles:\n")
    for i, item in enumerate(items, 1):
        lines.append(f"### {i}. {item.get('title', 'Untitled')}")
        lines.append(f"  Source: {item.get('publisher', 'Unknown')}")
        lines.append(f"  Type: {item.get('type', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


def get_news_prompt(symbol: str, news_report: str) -> str:
    """Generate LLM prompt for news sentiment analysis."""
    return f"""You are a senior news analyst specializing in market-moving events.

Analyze the following recent news for {symbol}:

{news_report}

## Your Analysis Should Include:

1. **Sentiment Summary**: Overall sentiment from recent news (Positive/Negative/Mixed/Neutral)

2. **Key Events**: Most impactful news items and their potential market effect

3. **Macro Context**: How broader market/economic trends affect this stock

4. **Catalyst Assessment**: Any upcoming catalysts (earnings, product launches, regulatory events)

5. **News-Based Risk Factors**: Negative headlines or emerging risks

6. **Sentiment Rating**: Very Bullish / Bullish / Neutral / Bearish / Very Bearish
"""


# ── Sentiment Analyst ──────────────────────────────────────────────────────

def get_sentiment_prompt(symbol: str, news_report: str, technical_report: str) -> str:
    """Generate LLM prompt for social/market sentiment analysis."""
    return f"""You are a market sentiment analyst who gauges investor mood and positioning.

Based on the available data for {symbol}, analyze market sentiment:

## Available Data
{news_report}

{technical_report}

## Your Analysis Should Include:

1. **Retail Sentiment**: What does retail investor behavior suggest?
   - Volume patterns (high volume = high interest)
   - Price momentum (chasing or distributing?)

2. **Institutional Signals**: Any signs of institutional activity?
   - Large volume days
   - OBV divergences

3. **Fear/Greed Assessment**: Where is sentiment on the fear-greed spectrum?

4. **Contrarian Indicators**: Any extremes that suggest reversal?
   - RSI extremes
   - Bollinger Band extremes

5. **Sentiment Rating**: Very Bullish / Bullish / Neutral / Bearish / Very Bearish
"""


# ── Combined Analysis ──────────────────────────────────────────────────────

def run_all_analyses(symbol: str) -> dict[str, str]:
    """Run all analysis types and return reports dict."""
    return {
        "technical": run_technical_analysis(symbol),
        "fundamental": run_fundamental_analysis(symbol),
        "news": run_news_analysis(symbol),
    }
