"""
Base backtesting engine with SignalEngine contract.

Inspired by Vibe-Trading's BaseEngine pattern. Provides:
  - SignalEngine protocol (input OHLCV → output signals [-1, 1])
  - Bar-by-bar execution with position management
  - Performance metrics calculation
  - Artifact generation (equity curve, trades, metrics)
"""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable


# ── SignalEngine Contract ──────────────────────────────────────────────────

@runtime_checkable
class SignalEngine(Protocol):
    """Standard strategy interface.

    Implement this to create a trading strategy:
        - Input:  data_map = {symbol: list[OHLCV_dict]}
        - Output: signal_map = {symbol: list[float]}
        - Signal values: -1.0 (full short) to 1.0 (full long), 0.0 = flat
    """

    def generate(self, data_map: dict[str, list[dict]]) -> dict[str, list[float]]:
        """Generate trading signals from OHLCV data.

        Args:
            data_map: {symbol: [{"date","open","high","low","close","volume"}, ...]}

        Returns:
            {symbol: [signal_float, ...]}
            Signal convention:
                1.0  = fully long (100% of allocated capital)
                0.5  = half long position
                0.0  = flat (no position)
               -0.5  = half short position
               -1.0  = fully short (100% short)
        """
        ...


# ── Trade Record ───────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol: str
    side: str           # "long" | "short"
    entry_date: str
    entry_price: float
    exit_date: str = ""
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0
    commission: float = 0.0
    bars_held: int = 0


@dataclass
class Position:
    symbol: str
    side: str           # "long" | "short"
    entry_date: str
    entry_price: float
    quantity: float
    bars_held: int = 0


# ── Backtest Configuration ─────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission: float = 0.001       # 0.1% per trade
    slippage: float = 0.0005        # 0.05% slippage
    max_position_pct: float = 1.0   # max % of capital per position


# ── Base Engine ────────────────────────────────────────────────────────────

class BaseEngine(ABC):
    """Abstract backtesting engine. Subclasses implement market-specific rules."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.trades: list[Trade] = []
        self.positions: dict[str, Position] = {}
        self.equity_curve: list[dict] = []
        self.cash = self.config.initial_capital

    @abstractmethod
    def can_execute(self, symbol: str, side: str, bar_idx: int) -> bool:
        """Check if trade can be executed (market hours, T+1 rules, etc.)."""
        ...

    @abstractmethod
    def calc_commission(self, price: float, quantity: float, side: str) -> float:
        """Calculate commission for a trade."""
        ...

    @abstractmethod
    def apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to execution price."""
        ...

    def on_bar(self, symbol: str, bar: dict, bar_idx: int) -> None:
        """Per-bar hook for market-specific logic (funding fees, etc.)."""
        pass

    def run_backtest(
        self,
        data_map: dict[str, list[dict]],
        signal_map: dict[str, list[float]],
    ) -> dict:
        """Execute backtest with given data and signals.

        Args:
            data_map: {symbol: [OHLCV_dict, ...]}
            signal_map: {symbol: [signal_float, ...]}

        Returns:
            {"metrics": dict, "trades": list, "equity": list}
        """
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        self.cash = self.config.initial_capital

        # Align data — find common date range
        symbols = list(data_map.keys())
        if not symbols:
            return self._empty_result()

        # Use first symbol's length as reference, signals should be aligned
        max_bars = min(len(data_map[s]) for s in symbols)

        for bar_idx in range(max_bars):
            # Per-bar hooks
            for sym in symbols:
                if bar_idx < len(data_map[sym]):
                    self.on_bar(sym, data_map[sym][bar_idx], bar_idx)

            # Update position holding time
            for pos in self.positions.values():
                pos.bars_held += 1

            # Process signals: rebalance to target weights
            for sym in symbols:
                if bar_idx >= len(signal_map.get(sym, [])):
                    continue
                if bar_idx >= len(data_map[sym]):
                    continue

                target_signal = signal_map[sym][bar_idx]
                target_signal = max(-1.0, min(1.0, target_signal))  # clamp
                bar = data_map[sym][bar_idx]
                price = bar["close"]

                # Skip if next bar doesn't exist (can't execute at next open)
                if bar_idx + 1 >= len(data_map[sym]):
                    continue

                next_bar = data_map[sym][bar_idx + 1]
                exec_price = next_bar["open"]

                current_signal = self._current_signal(sym, exec_price)

                # Only rebalance if signal change is significant
                if abs(target_signal - current_signal) < 0.05:
                    continue

                self._rebalance(sym, target_signal, exec_price, bar_idx + 1, next_bar["date"])

            # Record equity
            equity = self._total_equity(data_map, bar_idx)
            date = data_map[symbols[0]][bar_idx]["date"] if bar_idx < len(data_map[symbols[0]]) else ""
            self.equity_curve.append({"date": date, "equity": round(equity, 2)})

        # Close all remaining positions at last available price
        for sym in list(self.positions.keys()):
            if data_map[sym]:
                last_bar = data_map[sym][-1]
                self._close_position(sym, last_bar["close"], len(data_map[sym]) - 1, last_bar["date"])

        metrics = calc_metrics(
            self.equity_curve, self.trades, self.config.initial_capital
        )
        return {
            "metrics": metrics,
            "trades": [self._trade_to_dict(t) for t in self.trades],
            "equity": self.equity_curve,
        }

    def _current_signal(self, symbol: str, current_price: float) -> float:
        """Get current effective signal for a symbol."""
        if symbol not in self.positions:
            return 0.0
        pos = self.positions[symbol]
        pos_value = pos.quantity * current_price
        total_equity = self.cash + sum(
            p.quantity * current_price for p in self.positions.values()
        )
        if total_equity == 0:
            return 0.0
        ratio = pos_value / total_equity
        return ratio if pos.side == "long" else -ratio

    def _rebalance(
        self, symbol: str, target: float, price: float, bar_idx: int, date: str
    ) -> None:
        """Rebalance position to target signal."""
        total_equity = self.cash + sum(
            p.quantity * price for p in self.positions.values()
        )
        target_value = abs(target) * total_equity * self.config.max_position_pct
        target_side = "long" if target > 0 else "short" if target < 0 else "flat"

        # Close existing position if switching sides or going flat
        if symbol in self.positions:
            pos = self.positions[symbol]
            if target_side == "flat" or pos.side != target_side:
                self._close_position(symbol, price, bar_idx, date)

        if target_side == "flat":
            return

        # Check if we can execute
        if not self.can_execute(symbol, target_side, bar_idx):
            return

        # Open or adjust position
        exec_price = self.apply_slippage(price, target_side)

        if symbol in self.positions:
            # Adjust existing position
            pos = self.positions[symbol]
            current_value = pos.quantity * exec_price
            delta_value = target_value - current_value
            if abs(delta_value) < exec_price:  # minimum 1 share
                return
            delta_qty = abs(delta_value) / exec_price
            commission = self.calc_commission(exec_price, delta_qty, target_side)
            if delta_value > 0:
                # Increase position
                cost = delta_qty * exec_price + commission
                if cost > self.cash:
                    delta_qty = (self.cash - commission) / exec_price
                    cost = delta_qty * exec_price + commission
                if delta_qty <= 0:
                    return
                self.cash -= cost
                pos.quantity += delta_qty
            else:
                # Decrease position
                self.cash += delta_qty * exec_price - commission
                pos.quantity -= delta_qty
        else:
            # New position
            quantity = target_value / exec_price
            commission = self.calc_commission(exec_price, quantity, target_side)
            cost = quantity * exec_price + commission
            if cost > self.cash:
                quantity = (self.cash - commission) / exec_price
                cost = quantity * exec_price + commission
            if quantity <= 0:
                return
            self.cash -= cost
            self.positions[symbol] = Position(
                symbol=symbol,
                side=target_side,
                entry_date=date,
                entry_price=exec_price,
                quantity=quantity,
            )

    def _close_position(
        self, symbol: str, price: float, bar_idx: int, date: str
    ) -> None:
        """Close a position and record the trade."""
        if symbol not in self.positions:
            return
        pos = self.positions.pop(symbol)
        exec_price = self.apply_slippage(price, "sell" if pos.side == "long" else "cover")
        commission = self.calc_commission(exec_price, pos.quantity, pos.side)

        if pos.side == "long":
            pnl = (exec_price - pos.entry_price) * pos.quantity - commission
            self.cash += pos.quantity * exec_price - commission
        else:
            pnl = (pos.entry_price - exec_price) * pos.quantity - commission
            self.cash += pos.quantity * (2 * pos.entry_price - exec_price) - commission

        self.trades.append(Trade(
            symbol=symbol,
            side=pos.side,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=date,
            exit_price=exec_price,
            quantity=pos.quantity,
            pnl=round(pnl, 2),
            commission=round(commission, 2),
            bars_held=pos.bars_held,
        ))

    def _total_equity(self, data_map: dict[str, list[dict]], bar_idx: int) -> float:
        """Calculate total equity (cash + positions marked to market)."""
        equity = self.cash
        for sym, pos in self.positions.items():
            if bar_idx < len(data_map.get(sym, [])):
                price = data_map[sym][bar_idx]["close"]
                if pos.side == "long":
                    equity += pos.quantity * price
                else:
                    equity += pos.quantity * (2 * pos.entry_price - price)
        return equity

    def _empty_result(self) -> dict:
        return {
            "metrics": calc_metrics([], [], self.config.initial_capital),
            "trades": [],
            "equity": [],
        }

    @staticmethod
    def _trade_to_dict(t: Trade) -> dict:
        return {
            "symbol": t.symbol, "side": t.side,
            "entry_date": t.entry_date, "entry_price": t.entry_price,
            "exit_date": t.exit_date, "exit_price": t.exit_price,
            "quantity": round(t.quantity, 4), "pnl": t.pnl,
            "commission": t.commission, "bars_held": t.bars_held,
        }

    # ── Walk-forward backtest ──────────────────────────────────────────────

    def walk_forward(
        self,
        signal_engine: "SignalEngine",
        data_map: dict[str, list[dict]],
        n_splits: int = 5,
        min_split_bars: int = 60,
    ) -> dict:
        """Run rolling out-of-sample backtests across n_splits chunks.

        Why: ``run_backtest`` runs on the entire history at once — so a
        strategy that worked great in 2017-2019 but blew up in 2022 still
        looks fine on aggregate. Walk-forward chunks the history and
        reports per-chunk performance, surfacing regime-dependent rot
        that aggregate metrics hide.

        For parameter-free strategies (the four that ship with /trading
        backtest) every chunk is a true out-of-sample window. For
        parameterised strategies the caller is responsible for fitting
        only on data preceding each test window.

        Returns:
            {
                "splits": [{"start_date", "end_date", "metrics", "trades"}…],
                "stability": {  # cross-chunk stats
                    "sharpe_min", "sharpe_mean", "sharpe_stdev",
                    "return_consistency",  # fraction of chunks with positive return
                    "verdict": str,
                },
                "aggregate_metrics": dict,  # for comparison with run_backtest
            }
        """
        symbols = list(data_map.keys())
        if not symbols:
            return {"splits": [], "stability": {}, "aggregate_metrics": _empty_metrics(self.config.initial_capital)}

        max_bars = min(len(data_map[s]) for s in symbols)
        if max_bars < n_splits * min_split_bars:
            n_splits = max(1, max_bars // min_split_bars)
        if n_splits < 2:
            return {
                "splits": [],
                "stability": {"verdict": f"Not enough history for walk-forward ({max_bars} bars, need {2 * min_split_bars}+)."},
                "aggregate_metrics": _empty_metrics(self.config.initial_capital),
            }

        chunk = max_bars // n_splits
        split_results = []

        # Generate signals once on full history (signal engines are pure
        # functions of the data window seen so far for the strategies that
        # ship with /trading backtest — they only look back, never ahead).
        signal_map_full = signal_engine.generate(data_map)

        per_chunk_returns: list[float] = []
        per_chunk_sharpes: list[float] = []

        for i in range(n_splits):
            start = i * chunk
            end = (i + 1) * chunk if i < n_splits - 1 else max_bars
            sliced_data = {s: data_map[s][start:end] for s in symbols}
            sliced_signals = {
                s: signal_map_full.get(s, [])[start:end] for s in symbols
            }

            # Reset engine state for each chunk so the chunks are
            # genuinely independent. Subclasses must be re-instantiable
            # cheaply; if state matters use a deepcopy here instead.
            self.trades = []
            self.positions = {}
            self.equity_curve = []
            self.cash = self.config.initial_capital

            chunk_result = self.run_backtest(sliced_data, sliced_signals)

            metrics = chunk_result["metrics"]
            split_results.append({
                "split": i + 1,
                "start_idx": start,
                "end_idx": end,
                "start_date": sliced_data[symbols[0]][0]["date"] if sliced_data[symbols[0]] else "",
                "end_date":   sliced_data[symbols[0]][-1]["date"] if sliced_data[symbols[0]] else "",
                "metrics": metrics,
                "trade_count": len(chunk_result["trades"]),
            })
            per_chunk_returns.append(metrics["total_return"])
            per_chunk_sharpes.append(metrics["sharpe_ratio"])

        # Stability stats
        positive_chunks = sum(1 for r in per_chunk_returns if r > 0)
        consistency = positive_chunks / len(per_chunk_returns)
        sharpe_mean = sum(per_chunk_sharpes) / len(per_chunk_sharpes)
        sharpe_min = min(per_chunk_sharpes)
        if len(per_chunk_sharpes) >= 2:
            mean = sharpe_mean
            sharpe_stdev = math.sqrt(
                sum((s - mean) ** 2 for s in per_chunk_sharpes) / len(per_chunk_sharpes)
            )
        else:
            sharpe_stdev = 0.0

        if consistency >= 0.7 and sharpe_min > 0:
            verdict = "STABLE — strategy works across most regimes."
        elif consistency >= 0.5 and sharpe_mean > 0.5:
            verdict = "MIXED — works on average but blows up in some regimes; consider regime filter."
        elif consistency < 0.3:
            verdict = "FRAGILE — fails in most chunks; aggregate Sharpe is misleading."
        else:
            verdict = "INCONCLUSIVE — too noisy to call."

        return {
            "splits": split_results,
            "stability": {
                "n_splits": len(split_results),
                "return_consistency": round(consistency * 100, 1),
                "sharpe_mean": round(sharpe_mean, 3),
                "sharpe_min": round(sharpe_min, 3),
                "sharpe_stdev": round(sharpe_stdev, 3),
                "positive_chunks": positive_chunks,
                "verdict": verdict,
            },
        }


# ── Performance Metrics ───────────────────────────────────────────────────

def calc_metrics(
    equity_curve: list[dict],
    trades: list[Trade],
    initial_capital: float,
    bars_per_year: int = 252,
) -> dict:
    """Calculate comprehensive backtest metrics."""
    if not equity_curve:
        return _empty_metrics(initial_capital)

    equities = [e["equity"] for e in equity_curve]
    final_equity = equities[-1]

    # Returns
    total_return = (final_equity - initial_capital) / initial_capital
    n_bars = len(equities)
    years = n_bars / bars_per_year if bars_per_year else 1
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Daily returns
    daily_returns = []
    for i in range(1, len(equities)):
        if equities[i - 1] > 0:
            daily_returns.append(equities[i] / equities[i - 1] - 1)

    # Volatility
    if daily_returns:
        mean_ret = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_ret) ** 2 for r in daily_returns) / len(daily_returns)
        daily_vol = math.sqrt(variance)
        ann_vol = daily_vol * math.sqrt(bars_per_year)
    else:
        ann_vol = 0

    # Sharpe
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Sortino (downside deviation)
    neg_returns = [r for r in daily_returns if r < 0]
    if neg_returns:
        downside_var = sum(r ** 2 for r in neg_returns) / len(daily_returns)
        downside_dev = math.sqrt(downside_var) * math.sqrt(bars_per_year)
        sortino = ann_return / downside_dev if downside_dev > 0 else 0
    else:
        sortino = 0

    # Max drawdown
    peak = equities[0]
    max_dd = 0
    max_dd_duration = 0
    dd_start = 0
    for i, eq in enumerate(equities):
        if eq > peak:
            peak = eq
            dd_start = i
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd
            max_dd_duration = i - dd_start

    # Calmar
    calmar = ann_return / max_dd if max_dd > 0 else 0

    # Trade stats
    n_trades = len(trades)
    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]
    win_rate = len(winning) / n_trades if n_trades > 0 else 0

    avg_win = sum(t.pnl for t in winning) / len(winning) if winning else 0
    avg_loss = abs(sum(t.pnl for t in losing) / len(losing)) if losing else 0
    profit_factor = (sum(t.pnl for t in winning) / abs(sum(t.pnl for t in losing))
                     if losing and sum(t.pnl for t in losing) != 0 else 0)

    total_commission = sum(t.commission for t in trades)
    avg_bars_held = sum(t.bars_held for t in trades) / n_trades if n_trades else 0

    # Max consecutive losses
    max_consec_loss = 0
    consec = 0
    for t in trades:
        if t.pnl <= 0:
            consec += 1
            max_consec_loss = max(max_consec_loss, consec)
        else:
            consec = 0

    return {
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "total_return": round(total_return * 100, 2),
        "annualized_return": round(ann_return * 100, 2),
        "annualized_volatility": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "max_dd_duration_bars": max_dd_duration,
        "total_trades": n_trades,
        "win_rate": round(win_rate * 100, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "total_commission": round(total_commission, 2),
        "avg_bars_held": round(avg_bars_held, 1),
        "max_consecutive_losses": max_consec_loss,
    }


def _empty_metrics(initial_capital: float) -> dict:
    return {
        "initial_capital": initial_capital,
        "final_equity": initial_capital,
        "total_return": 0, "annualized_return": 0,
        "annualized_volatility": 0, "sharpe_ratio": 0,
        "sortino_ratio": 0, "calmar_ratio": 0,
        "max_drawdown": 0, "max_dd_duration_bars": 0,
        "total_trades": 0, "win_rate": 0,
        "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
        "total_commission": 0, "avg_bars_held": 0,
        "max_consecutive_losses": 0,
    }


def format_metrics_report(metrics: dict) -> str:
    """Format metrics into a readable report string."""
    return f"""## Backtest Results

### Returns
  Initial Capital:    ${metrics['initial_capital']:>12,.2f}
  Final Equity:       ${metrics['final_equity']:>12,.2f}
  Total Return:       {metrics['total_return']:>11.2f}%
  Annualized Return:  {metrics['annualized_return']:>11.2f}%

### Risk
  Annual Volatility:  {metrics['annualized_volatility']:>11.2f}%
  Max Drawdown:       {metrics['max_drawdown']:>11.2f}%
  Max DD Duration:    {metrics['max_dd_duration_bars']:>11d} bars

### Risk-Adjusted
  Sharpe Ratio:       {metrics['sharpe_ratio']:>11.3f}
  Sortino Ratio:      {metrics['sortino_ratio']:>11.3f}
  Calmar Ratio:       {metrics['calmar_ratio']:>11.3f}

### Trades
  Total Trades:       {metrics['total_trades']:>11d}
  Win Rate:           {metrics['win_rate']:>11.2f}%
  Avg Win:            ${metrics['avg_win']:>12,.2f}
  Avg Loss:           ${metrics['avg_loss']:>12,.2f}
  Profit Factor:      {metrics['profit_factor']:>11.3f}
  Avg Bars Held:      {metrics['avg_bars_held']:>11.1f}
  Max Consec. Losses: {metrics['max_consecutive_losses']:>11d}
  Total Commission:   ${metrics['total_commission']:>12,.2f}
"""
