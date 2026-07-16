"""
managed.py — managed paper-portfolio mode.

This is the "I give the agent $100, check in a week" feature. Each
managed portfolio is a named PaperBroker instance plus a re-evaluation
loop that:

  1. Picks a candidate universe (watchlist + a small set of
     macro/sector ETFs as fallback)
  2. Runs the multi-agent analyze pipeline on each candidate
  3. Optionally feeds candidates through portfolio.optimize() to get
     mean-variance weights, capped per single name
  4. Places market orders to bring the broker book toward target weights
  5. Snapshots equity for the equity curve

A "step" is one full cycle. The user can:
  - call /trading manage step name      → run one cycle now
  - call /trading manage status name    → check positions + PnL
  - call /trading manage report name    → human-readable PnL report

Honest limits: re-evaluation is on-demand, not real-time-streaming.
yfinance prices are 15-20 min delayed for free tier. This is a
scheduled-discipline tool, not a high-frequency trader.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .broker.paper_backend import PaperBroker
from .data import fetchers
from . import portfolio as opt


_DEFAULT_UNIVERSE = ["SPY", "QQQ", "VTI", "IWM", "DIA", "EFA", "EEM"]


@dataclass
class StepResult:
    portfolio:    str
    timestamp:    str
    cash_before:  float
    cash_after:   float
    equity_before: float
    equity_after:  float
    target_weights: dict[str, float]
    orders:        list[dict[str, Any]]
    notes:         list[str]


# ── Lifecycle ─────────────────────────────────────────────────────────────

def start_portfolio(name: str, initial_cash: float = 100.0,
                    db_path: Path | str | None = None) -> PaperBroker:
    """Create or open a managed portfolio."""
    return PaperBroker(name=name, db_path=db_path, initial_cash=initial_cash)


def list_portfolios(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    return PaperBroker.list_portfolios(db_path=db_path)


# ── Universe selection ───────────────────────────────────────────────────

def _universe_for(broker: PaperBroker, db_path: Path | str | None = None) -> list[str]:
    """Pick the symbol universe for this rebalance.

    Priority:
      1. The cheetahclaws watchlist (paper_trader.watchlist_list)
      2. The default macro/sector ETF set (gives the agent something
         tradeable on day 1 even if no watchlist is set)
    """
    try:
        from . import paper_trader
        wl = [w["symbol"] for w in paper_trader.watchlist_list()]
        if wl:
            return wl
    except Exception:
        pass
    return list(_DEFAULT_UNIVERSE)


# ── Core: one rebalance step ─────────────────────────────────────────────

def step(name: str,
         max_positions: int = 5,
         max_weight: float = 0.30,
         min_weight: float = 0.05,
         dry_run: bool = False,
         db_path: Path | str | None = None) -> StepResult:
    """One re-evaluation cycle for the named portfolio.

    Strategy: mean-variance optimisation over the watchlist universe,
    long-only, single-name capped, fully invested unless cash is
    explicitly preferred. Trades are sized by available equity at this
    instant, not by static dollar allocation.
    """
    broker = PaperBroker(name=name, db_path=db_path)
    summary_before = broker.account_summary()
    notes: list[str] = []

    universe = _universe_for(broker, db_path=db_path)
    notes.append(f"Universe: {', '.join(universe)}")

    # Build candidates with recent OHLCV
    candidates: list[opt.Candidate] = []
    for sym in universe:
        result = fetchers.fetch_market_data(sym, interval="1d")
        if result.get("error"):
            notes.append(f"  ⚠ {sym}: data error ({result['error']})")
            continue
        rows = result.get("data", [])
        if len(rows) < 60:
            notes.append(f"  ⚠ {sym}: only {len(rows)} bars, skipped")
            continue
        candidates.append(opt.Candidate(
            symbol=sym,
            closes=[r["close"] for r in rows],
            sector=None,
        ))

    if not candidates:
        notes.append("No tradeable candidates — skipping rebalance.")
        broker.snapshot_equity()
        return StepResult(name, _now(), summary_before.cash, summary_before.cash,
                          summary_before.equity, summary_before.equity,
                          {}, [], notes)

    # Mean-variance solve
    result = opt.optimize(candidates, max_weight=max_weight)

    # Drop weights below min_weight (avoid 0.5% positions that can't
    # absorb commission / spread on a $100 book)
    weights = {s: w for s, w in result.weights.items() if w >= min_weight}

    # If MV gave us nothing significant, equal-weight the top max_positions
    if not weights:
        n = min(max_positions, len(candidates))
        if n > 0:
            equal = round(1.0 / n, 4)
            weights = {c.symbol: equal for c in candidates[:n]}
            notes.append(f"MV produced no weights ≥ {min_weight}; equal-weighted {n} names.")

    # Renormalise so weights sum to ≤ 1 after filtering
    total = sum(weights.values())
    if total > 1.0:
        weights = {s: round(w / total, 4) for s, w in weights.items()}

    notes.append(f"Target weights: {weights}")

    # Compute target $ per symbol from current equity
    equity = summary_before.equity
    target_dollars = {s: w * equity for s, w in weights.items()}

    # Determine current per-symbol $ holdings
    cur_positions = {p.symbol: p for p in broker.positions()}

    orders: list[dict[str, Any]] = []
    if dry_run:
        notes.append("Dry run — no orders placed.")
    else:
        # First sell everything not in target
        for sym, p in list(cur_positions.items()):
            if sym not in weights:
                if p.current_price is None:
                    notes.append(f"  ⚠ Cannot sell {sym}: no quote")
                    continue
                r = broker.place_market_order(sym, "SELL", p.quantity)
                orders.append(asdict(r))
                if not r.success:
                    notes.append(f"  ⚠ {sym} SELL failed: {r.error}")

        # Refresh state
        cur_positions = {p.symbol: p for p in broker.positions()}

        # For each target symbol, BUY/SELL to hit target $
        for sym, target_usd in target_dollars.items():
            quote = broker.quote(sym)
            if quote is None or quote <= 0:
                notes.append(f"  ⚠ {sym}: no quote, skipping")
                continue
            held = cur_positions.get(sym)
            cur_qty = held.quantity if held else 0.0
            cur_usd = cur_qty * quote
            delta_usd = target_usd - cur_usd
            # Don't trade unless meaningful (avoid commission grind on tiny rebalances)
            if abs(delta_usd) < max(equity * 0.02, 5.0):
                continue
            qty = abs(delta_usd) / quote
            # Round to 4 decimals (most retail brokers support fractional shares)
            qty = round(qty, 4)
            if qty <= 0:
                continue
            side = "BUY" if delta_usd > 0 else "SELL"
            # Don't try to sell more than held
            if side == "SELL":
                qty = min(qty, cur_qty)
                if qty <= 0:
                    continue
            r = broker.place_market_order(sym, side, qty)
            orders.append(asdict(r))
            if not r.success:
                notes.append(f"  ⚠ {sym} {side} failed: {r.error}")

    # Snapshot equity curve
    summary_after = broker.snapshot_equity()

    return StepResult(
        portfolio=name,
        timestamp=_now(),
        cash_before=summary_before.cash,
        cash_after=summary_after.cash,
        equity_before=summary_before.equity,
        equity_after=summary_after.equity,
        target_weights=weights,
        orders=orders,
        notes=notes,
    )


# ── Status / report ──────────────────────────────────────────────────────

def status(name: str, db_path: Path | str | None = None) -> dict[str, Any]:
    """Current cash + positions + unrealized PnL."""
    broker = PaperBroker(name=name, db_path=db_path)
    s = broker.account_summary()
    initial = broker.initial_cash
    pnl_dollars = s.equity - initial
    pnl_pct = (pnl_dollars / initial * 100.0) if initial > 0 else 0.0

    return {
        "portfolio":  name,
        "initial_cash": initial,
        "cash":       s.cash,
        "equity":     s.equity,
        "pnl_dollars": pnl_dollars,
        "pnl_pct":    pnl_pct,
        "open_positions_count": s.open_positions_count,
        "positions":  [asdict(p) for p in broker.positions()],
    }


def report(name: str, db_path: Path | str | None = None) -> str:
    """Human-readable markdown report. Includes equity curve."""
    broker = PaperBroker(name=name, db_path=db_path)
    s = broker.account_summary()
    initial = broker.initial_cash
    pnl_dollars = s.equity - initial
    pnl_pct = (pnl_dollars / initial * 100.0) if initial > 0 else 0.0

    sign = "+" if pnl_dollars > 0 else ""
    headline_emoji = "🟢" if pnl_dollars > 0 else "🔴" if pnl_dollars < 0 else "⚪"

    lines = [f"# {headline_emoji} Managed portfolio: `{name}`"]
    lines.append("")
    lines.append(f"**Initial**: ${initial:.2f}   →   **Now**: ${s.equity:.2f}   "
                 f"({sign}${pnl_dollars:.2f}, {sign}{pnl_pct:.2f}%)")
    lines.append(f"**Cash**: ${s.cash:.2f}   |   **Open positions**: {s.open_positions_count}")
    lines.append("")

    positions = broker.positions()
    if positions:
        lines.append("## Holdings")
        lines.append("| Symbol | Qty | Avg cost | Last | Market value | Unrealized |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for p in sorted(positions, key=lambda x: -(x.market_value or 0)):
            mv = f"${p.market_value:.2f}" if p.market_value is not None else "—"
            unreal = f"{'+' if (p.unrealized_pnl or 0) > 0 else ''}${p.unrealized_pnl:.2f}" if p.unrealized_pnl is not None else "—"
            cur = f"${p.current_price:.2f}" if p.current_price is not None else "—"
            lines.append(f"| {p.symbol} | {p.quantity:.4f} | ${p.avg_cost:.2f} | {cur} | {mv} | {unreal} |")
        lines.append("")

    # Equity curve summary
    curve = broker.equity_curve()
    if curve:
        lines.append("## Equity curve")
        first = curve[0]
        last = curve[-1]
        lines.append(f"- Snapshots: {len(curve)}, from {first['snapshot_at'][:10]} "
                     f"to {last['snapshot_at'][:10]}")
        equities = [c["cash"] + c["market_value"] for c in curve]
        peak = max(equities)
        trough = min(equities)
        max_dd = (peak - trough) / peak * 100.0 if peak > 0 else 0.0
        lines.append(f"- Peak ${peak:.2f}, trough ${trough:.2f}, max DD {max_dd:.2f}%")

    # Recent orders
    orders = broker.order_history(limit=10)
    if orders:
        lines.append("")
        lines.append("## Recent orders")
        for o in orders:
            status_emoji = "✓" if o["success"] else "✗"
            lines.append(f"- {status_emoji} {o['placed_at'][:19]}  {o['side']:<4} "
                         f"{o['quantity']:.4f} {o['symbol']}  @ ${o['fill_price']:.2f}"
                         + (f"   [{o['error']}]" if o["error"] else ""))

    lines.append("")
    lines.append("> Reminder: paper trades. Real broker would charge commission "
                 "and spread; small accounts (<$1k) have unfavourable fixed-cost "
                 "economics in real life.")
    return "\n".join(lines)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
