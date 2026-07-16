"""
Trading tools for the AI agent.

Exports TOOL_DEFS — automatically registered by the modular loader.
These tools allow the LLM to fetch market data, run analysis,
execute backtests, and manage trading memory.
"""
from __future__ import annotations

import json
from cheetahclaws.tool_registry import ToolDef

from .data.fetchers import fetch_market_data, fetch_current_price, fetch_fundamentals, fetch_news
from .data.indicators import compute_all, format_indicators_report
from .engines.base import BacktestConfig, calc_metrics, format_metrics_report


# ── Tool implementations ──────────────────────────────────────────────────

def _get_market_data(params: dict, config: dict) -> str:
    """Fetch OHLCV market data for a symbol."""
    symbol = params["symbol"]
    start_date = params.get("start_date")
    end_date = params.get("end_date")
    interval = params.get("interval", "1d")
    source = params.get("source", "auto")

    result = fetch_market_data(symbol, start_date, end_date, interval, source)
    if result.get("error"):
        return f"Error: {result['error']}"

    data = result["data"]
    info = result.get("info", {})
    output = f"Market Data: {symbol} (source: {result['source']})\n"
    if info:
        output += f"Name: {info.get('name', 'N/A')}\n"
        output += f"Current Price: ${info.get('price', 0):,.4f}\n"
    output += f"Period: {data[0]['date']} to {data[-1]['date']} ({len(data)} bars)\n\n"

    # Show last 20 bars
    output += "Recent OHLCV (last 20 bars):\n"
    output += f"{'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}\n"
    output += "-" * 66 + "\n"
    for bar in data[-20:]:
        output += (f"{bar['date']:<12} {bar['open']:>10.4f} {bar['high']:>10.4f} "
                   f"{bar['low']:>10.4f} {bar['close']:>10.4f} {bar['volume']:>12,}\n")

    return output


def _get_price(params: dict, config: dict) -> str:
    """Get current price for a symbol."""
    symbol = params["symbol"]
    result = fetch_current_price(symbol)
    if result.get("error"):
        return f"Error: {result['error']}"

    output = f"Price: {symbol}\n"
    output += f"  Price: ${result.get('price', 0):,.4f}\n"
    output += f"  Change: {result.get('change_pct', 0):+.2f}%\n"
    if result.get("name"):
        output += f"  Name: {result['name']}\n"
    if result.get("market_cap"):
        output += f"  Market Cap: ${result['market_cap']:,.0f}\n"
    if result.get("volume_24h"):
        output += f"  24h Volume: ${result['volume_24h']:,.0f}\n"
    return output


def _get_technical_indicators(params: dict, config: dict) -> str:
    """Compute and display technical indicators."""
    symbol = params["symbol"]
    result = fetch_market_data(symbol, interval="1d")
    if result.get("error"):
        return f"Error: {result['error']}"

    data = result["data"]
    if len(data) < 30:
        return f"Insufficient data for {symbol} ({len(data)} bars, need 30+)"

    indicators = compute_all(data)
    report = format_indicators_report(data, indicators)
    return f"# Technical Indicators: {symbol}\n\n{report}"


def _get_fundamentals(params: dict, config: dict) -> str:
    """Get fundamental data for a symbol."""
    symbol = params["symbol"]
    result = fetch_fundamentals(symbol)
    if result.get("error"):
        return f"Error: {result['error']}"

    output = f"Fundamentals: {symbol} ({result.get('name', 'N/A')})\n"
    output += f"  Sector: {result.get('sector', 'N/A')}\n"
    output += f"  Industry: {result.get('industry', 'N/A')}\n"
    output += f"  Market Cap: ${result.get('market_cap', 0):,.0f}\n"
    pe = result.get('pe_ratio')
    output += f"  P/E Ratio: {pe:.2f}\n" if pe else "  P/E Ratio: N/A\n"
    eps = result.get('eps')
    output += f"  EPS: ${eps:.2f}\n" if eps else "  EPS: N/A\n"
    output += f"  Revenue: ${result.get('revenue', 0):,.0f}\n"
    pm = result.get('profit_margin')
    output += f"  Profit Margin: {pm*100:.1f}%\n" if pm else "  Profit Margin: N/A\n"
    roe = result.get('roe')
    output += f"  ROE: {roe*100:.1f}%\n" if roe else "  ROE: N/A\n"
    beta = result.get('beta')
    output += f"  Beta: {beta:.2f}\n" if beta else "  Beta: N/A\n"
    return output


def _get_news(params: dict, config: dict) -> str:
    """Get recent news for a symbol."""
    symbol = params["symbol"]
    limit = params.get("limit", 10)
    result = fetch_news(symbol, limit)
    if result.get("error"):
        return f"Error: {result['error']}"

    items = result.get("news", [])
    if not items:
        return f"No recent news for {symbol}"

    output = f"Recent News: {symbol} ({len(items)} articles)\n\n"
    for i, item in enumerate(items, 1):
        output += f"{i}. {item.get('title', 'Untitled')}\n"
        output += f"   Source: {item.get('publisher', 'Unknown')}\n\n"
    return output


class _StrategySignalEngine:
    """SignalEngine-compatible wrapper around the inline strategy definitions.

    Used by walk_forward in BaseEngine, which expects a `.generate(data_map)`
    method matching the SignalEngine protocol.
    """

    def __init__(self, name: str):
        self.name = name

    def generate(self, data_map: dict) -> dict:
        out = {}
        for sym, data in data_map.items():
            out[sym] = _strategy_signals(self.name, [d["close"] for d in data])
        return out


def _strategy_signals(strategy: str, closes: list[float]) -> list[float]:
    """Return per-bar signals for the named strategy. Pure function of closes."""
    from .data import indicators as ind
    n = len(closes)
    signals = [0.0] * n

    if strategy == "dual_ma":
        sma_fast = ind.sma(closes, 20)
        sma_slow = ind.sma(closes, 50)
        for i in range(n):
            if sma_fast[i] is not None and sma_slow[i] is not None:
                signals[i] = 1.0 if sma_fast[i] > sma_slow[i] else -1.0

    elif strategy == "rsi_mean_reversion":
        rsi_vals = ind.rsi(closes, 14)
        for i in range(n):
            if rsi_vals[i] is not None:
                if rsi_vals[i] < 30:
                    signals[i] = 1.0
                elif rsi_vals[i] > 70:
                    signals[i] = -1.0

    elif strategy == "bollinger_breakout":
        bb = ind.bollinger_bands(closes, 20, 2.0)
        for i in range(n):
            if bb["upper"][i] is not None:
                if closes[i] > bb["upper"][i]:
                    signals[i] = 1.0
                elif closes[i] < bb["lower"][i]:
                    signals[i] = -1.0

    elif strategy == "macd_crossover":
        macd_data = ind.macd(closes)
        for i in range(n):
            hist = macd_data["histogram"][i]
            if hist is not None:
                signals[i] = 1.0 if hist > 0 else -1.0

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Available: dual_ma, rsi_mean_reversion, bollinger_breakout, macd_crossover"
        )
    return signals


def _build_strategy(name: str) -> _StrategySignalEngine:
    """Public factory used by walk-forward backtests."""
    # Validate by attempting a tiny generate.
    _strategy_signals(name, [1.0] * 60)
    return _StrategySignalEngine(name)


def _run_backtest(params: dict, config: dict) -> str:
    """Run a backtest with a given strategy."""
    symbol = params["symbol"]
    strategy = params.get("strategy", "dual_ma")
    start_date = params.get("start_date")
    end_date = params.get("end_date")
    initial_capital = params.get("initial_capital", 100000)

    # Fetch data
    result = fetch_market_data(symbol, start_date, end_date, interval="1d")
    if result.get("error"):
        return f"Error fetching data: {result['error']}"
    data = result["data"]
    if len(data) < 60:
        return f"Insufficient data for backtest ({len(data)} bars, need 60+)"

    # Generate signals based on strategy
    from .data import indicators as ind
    closes = [d["close"] for d in data]

    signal_map = {}
    if strategy == "dual_ma":
        # Dual moving average crossover: SMA(20) vs SMA(50)
        sma_fast = ind.sma(closes, 20)
        sma_slow = ind.sma(closes, 50)
        signals = [0.0] * len(closes)
        for i in range(len(closes)):
            if sma_fast[i] is not None and sma_slow[i] is not None:
                signals[i] = 1.0 if sma_fast[i] > sma_slow[i] else -1.0
        signal_map[symbol] = signals

    elif strategy == "rsi_mean_reversion":
        rsi_vals = ind.rsi(closes, 14)
        signals = [0.0] * len(closes)
        for i in range(len(closes)):
            if rsi_vals[i] is not None:
                if rsi_vals[i] < 30:
                    signals[i] = 1.0
                elif rsi_vals[i] > 70:
                    signals[i] = -1.0
                else:
                    signals[i] = 0.0
        signal_map[symbol] = signals

    elif strategy == "bollinger_breakout":
        bb = ind.bollinger_bands(closes, 20, 2.0)
        signals = [0.0] * len(closes)
        for i in range(len(closes)):
            if bb["upper"][i] is not None:
                if closes[i] > bb["upper"][i]:
                    signals[i] = 1.0
                elif closes[i] < bb["lower"][i]:
                    signals[i] = -1.0
        signal_map[symbol] = signals

    elif strategy == "macd_crossover":
        macd_data = ind.macd(closes)
        signals = [0.0] * len(closes)
        for i in range(len(closes)):
            hist = macd_data["histogram"][i]
            if hist is not None:
                signals[i] = 1.0 if hist > 0 else -1.0
        signal_map[symbol] = signals

    else:
        return (f"Unknown strategy: {strategy}. "
                f"Available: dual_ma, rsi_mean_reversion, bollinger_breakout, macd_crossover")

    # Determine engine
    from .data.fetchers import detect_market
    market = detect_market(symbol)

    bt_config = BacktestConfig(initial_capital=initial_capital)
    if market == "crypto":
        from .engines.crypto import CryptoEngine
        engine = CryptoEngine(bt_config)
    else:
        from .engines.equity import EquityEngine
        engine = EquityEngine(bt_config, market="hk" if market == "hk_equity" else "us")

    bt_result = engine.run_backtest({symbol: data}, signal_map)

    output = f"# Backtest Results: {strategy} on {symbol}\n\n"
    output += format_metrics_report(bt_result["metrics"])

    trades = bt_result["trades"]
    if trades:
        output += f"\n## Trade Summary ({len(trades)} trades)\n\n"
        output += f"{'Entry Date':<12} {'Exit Date':<12} {'Side':<6} {'Entry':>10} {'Exit':>10} {'PnL':>10}\n"
        output += "-" * 64 + "\n"
        for t in trades[-10:]:  # Show last 10
            output += (f"{t['entry_date']:<12} {t['exit_date']:<12} {t['side']:<6} "
                       f"${t['entry_price']:>9.2f} ${t['exit_price']:>9.2f} ${t['pnl']:>9.2f}\n")
        if len(trades) > 10:
            output += f"  ... and {len(trades) - 10} more trades\n"

    return output


def _trading_memory(params: dict, config: dict) -> str:
    """Inspect or manage trading memory."""
    action = params.get("action", "list")
    component = params.get("component", "portfolio_manager")

    from .agents.memory import get_memory, get_all_memories

    if action == "list":
        memories = get_all_memories()
        output = "Trading Memory Status:\n\n"
        for comp, mem in memories.items():
            output += f"  {comp}: {len(mem)} memories\n"
        return output

    elif action == "search":
        query = params.get("query", "")
        if not query:
            return "Error: query required for search"
        mem = get_memory(component)
        results = mem.get_memories(query, n_matches=5)
        if not results:
            return f"No matching memories in {component}"
        output = f"Search results for '{query}' in {component}:\n\n"
        for r in results:
            output += f"  Similarity: {r['similarity']}\n"
            output += f"  Situation: {r['situation'][:200]}\n"
            output += f"  Recommendation: {r['recommendation'][:200]}\n"
            output += f"  Outcome: {r['outcome'][:100]}\n\n"
        return output

    elif action == "clear":
        mem = get_memory(component)
        count = len(mem)
        mem.clear()
        return f"Cleared {count} memories from {component}"

    return f"Unknown action: {action}. Available: list, search, clear"


# ── TOOL_DEFS export ──────────────────────────────────────────────────────

TOOL_DEFS = [
    ToolDef(
        name="GetMarketData",
        schema={
            "name": "GetMarketData",
            "description": (
                "Fetch OHLCV (Open/High/Low/Close/Volume) market data for a stock or crypto symbol. "
                "Supports US stocks (AAPL, MSFT), HK stocks (0700.HK), A-shares (000001.SZ), "
                "and crypto (BTC, ETH). Uses automatic data source fallback."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol (e.g., AAPL, BTC, 0700.HK)"},
                    "start_date": {"type": "string", "description": "Start date YYYY-MM-DD (default: 1 year ago)"},
                    "end_date": {"type": "string", "description": "End date YYYY-MM-DD (default: today)"},
                    "interval": {"type": "string", "description": "Bar interval: 1d, 1h, 5m (default: 1d)"},
                    "source": {"type": "string", "description": "Data source: auto, yfinance, coingecko (default: auto)"},
                },
                "required": ["symbol"],
            },
        },
        func=_get_market_data,
        read_only=True,
        concurrent_safe=True,
    ),
    ToolDef(
        name="GetPrice",
        schema={
            "name": "GetPrice",
            "description": "Get current price and basic info for a stock or crypto symbol.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol (e.g., AAPL, BTC)"},
                },
                "required": ["symbol"],
            },
        },
        func=_get_price,
        read_only=True,
        concurrent_safe=True,
    ),
    ToolDef(
        name="GetTechnicalIndicators",
        schema={
            "name": "GetTechnicalIndicators",
            "description": (
                "Compute technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, ATR, "
                "OBV, VWAP, ADX, Stochastic) for a symbol and return a formatted report."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"},
                },
                "required": ["symbol"],
            },
        },
        func=_get_technical_indicators,
        read_only=True,
        concurrent_safe=True,
    ),
    ToolDef(
        name="GetFundamentals",
        schema={
            "name": "GetFundamentals",
            "description": (
                "Get fundamental financial data for a stock: P/E ratio, EPS, revenue, "
                "profit margin, ROE, debt/equity, market cap, beta, etc."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                },
                "required": ["symbol"],
            },
        },
        func=_get_fundamentals,
        read_only=True,
        concurrent_safe=True,
    ),
    ToolDef(
        name="GetNews",
        schema={
            "name": "GetNews",
            "description": "Get recent news articles for a stock symbol.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "limit": {"type": "integer", "description": "Max articles (default: 10)"},
                },
                "required": ["symbol"],
            },
        },
        func=_get_news,
        read_only=True,
        concurrent_safe=True,
    ),
    ToolDef(
        name="RunBacktest",
        schema={
            "name": "RunBacktest",
            "description": (
                "Run a backtest with a built-in strategy on historical data. "
                "Available strategies: dual_ma (SMA 20/50 crossover), "
                "rsi_mean_reversion (RSI 30/70), bollinger_breakout, macd_crossover."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"},
                    "strategy": {
                        "type": "string",
                        "description": "Strategy name: dual_ma, rsi_mean_reversion, bollinger_breakout, macd_crossover",
                    },
                    "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
                    "initial_capital": {"type": "number", "description": "Starting capital (default: 100000)"},
                },
                "required": ["symbol"],
            },
        },
        func=_run_backtest,
        read_only=True,
        concurrent_safe=True,
    ),
    ToolDef(
        name="TradingMemory",
        schema={
            "name": "TradingMemory",
            "description": (
                "Manage trading agent memory. Actions: list (show all component memories), "
                "search (find similar past situations), clear (remove memories)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Action: list, search, clear"},
                    "component": {
                        "type": "string",
                        "description": "Agent component: bull_researcher, bear_researcher, trader, risk_judge, portfolio_manager",
                    },
                    "query": {"type": "string", "description": "Search query (for action=search)"},
                },
                "required": ["action"],
            },
        },
        func=_trading_memory,
        read_only=False,
        concurrent_safe=False,
    ),
]
