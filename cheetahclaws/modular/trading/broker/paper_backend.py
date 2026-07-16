"""
broker/paper_backend.py — simulated broker on top of a managed cash + position book.

Each PaperBroker instance owns one named portfolio (its own SQLite table
namespace) so the user can run multiple managed portfolios in parallel
without collisions ("retire", "highrisk", "test", etc.).
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .base import BrokerBackend, OrderResult, Position, AccountSummary


_DEFAULT_DB = Path.home() / ".cheetahclaws" / "trading" / "managed_portfolios.db"
_LOCK = threading.Lock()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolios (
    name TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    initial_cash REAL NOT NULL,
    cash REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD'
);

CREATE TABLE IF NOT EXISTS portfolio_positions (
    portfolio TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_cost REAL NOT NULL,
    PRIMARY KEY (portfolio, symbol),
    FOREIGN KEY (portfolio) REFERENCES portfolios(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS portfolio_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio TEXT NOT NULL,
    placed_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    fill_price REAL,
    success INTEGER NOT NULL,
    error TEXT,
    FOREIGN KEY (portfolio) REFERENCES portfolios(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS portfolio_equity_curve (
    portfolio TEXT NOT NULL,
    snapshot_at TEXT NOT NULL,
    cash REAL NOT NULL,
    market_value REAL NOT NULL,
    PRIMARY KEY (portfolio, snapshot_at),
    FOREIGN KEY (portfolio) REFERENCES portfolios(name) ON DELETE CASCADE
);
"""


@contextmanager
def _conn(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path) if db_path is not None else _DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        c = sqlite3.connect(str(path))
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        try:
            yield c
            c.commit()
        finally:
            c.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db(db_path: Path | str | None = None) -> None:
    with _conn(db_path) as c:
        c.executescript(_SCHEMA)


# ── Public PaperBroker ────────────────────────────────────────────────────

class PaperBroker(BrokerBackend):
    """SQLite-backed paper-trading broker for one named portfolio."""

    def __init__(self, name: str = "default", db_path: Path | str | None = None,
                 initial_cash: float = 100.0):
        self.name = name
        self._db = db_path or _DEFAULT_DB
        self._initial_cash = initial_cash
        init_db(self._db)
        self._ensure_portfolio()

    @property
    def mode(self) -> str:
        return "paper"

    def _ensure_portfolio(self) -> None:
        with _conn(self._db) as c:
            row = c.execute(
                "SELECT name FROM portfolios WHERE name = ?", (self.name,),
            ).fetchone()
            if row is None:
                c.execute(
                    "INSERT INTO portfolios (name, created_at, initial_cash, cash) "
                    "VALUES (?, ?, ?, ?)",
                    (self.name, _now(), self._initial_cash, self._initial_cash),
                )

    # ── Account ─────────────────────────────────────────────────────

    def account_summary(self) -> AccountSummary:
        with _conn(self._db) as c:
            row = c.execute(
                "SELECT cash, currency, initial_cash FROM portfolios WHERE name = ?",
                (self.name,),
            ).fetchone()
            if row is None:
                return AccountSummary(0.0, 0.0, 0.0, 0)
            cash = row["cash"]
            positions = c.execute(
                "SELECT symbol, quantity, avg_cost FROM portfolio_positions WHERE portfolio = ?",
                (self.name,),
            ).fetchall()

        market_value = 0.0
        for p in positions:
            q = self.quote(p["symbol"])
            if q is not None:
                market_value += q * p["quantity"]
            else:
                # Fallback to last cost if quote fails
                market_value += p["avg_cost"] * p["quantity"]

        return AccountSummary(
            cash=cash,
            equity=cash + market_value,
            buying_power=cash,
            open_positions_count=len(positions),
            currency=row["currency"] or "USD",
        )

    def positions(self) -> list[Position]:
        with _conn(self._db) as c:
            rows = c.execute(
                "SELECT symbol, quantity, avg_cost FROM portfolio_positions WHERE portfolio = ?",
                (self.name,),
            ).fetchall()

        out = []
        for r in rows:
            price = self.quote(r["symbol"])
            mv = price * r["quantity"] if price else None
            cost_basis = r["avg_cost"] * r["quantity"]
            unrealized = (mv - cost_basis) if mv is not None else None
            out.append(Position(
                symbol=r["symbol"],
                quantity=r["quantity"],
                avg_cost=r["avg_cost"],
                current_price=price,
                market_value=mv,
                unrealized_pnl=unrealized,
            ))
        return out

    # ── Quote (delegates to fetchers.fetch_current_price) ──────────

    def quote(self, symbol: str) -> float | None:
        try:
            from ..data import fetchers
        except ImportError:
            return None
        try:
            info = fetchers.fetch_current_price(symbol)
            return info.get("price") if isinstance(info, dict) else None
        except Exception:
            return None

    # ── Orders ─────────────────────────────────────────────────────

    def place_market_order(self, symbol: str, side: str, quantity: float) -> OrderResult:
        side = side.upper().strip()
        if side not in ("BUY", "SELL"):
            return OrderResult(False, None, symbol, side, quantity, None, "Side must be BUY or SELL")
        if quantity <= 0:
            return OrderResult(False, None, symbol, side, quantity, None, "Quantity must be > 0")

        price = self.quote(symbol)
        if price is None or price <= 0:
            return OrderResult(False, None, symbol, side, quantity, None,
                               f"Could not quote {symbol}")

        with _conn(self._db) as c:
            row = c.execute(
                "SELECT cash FROM portfolios WHERE name = ?", (self.name,),
            ).fetchone()
            cash = row["cash"]

            pos_row = c.execute(
                "SELECT quantity, avg_cost FROM portfolio_positions "
                "WHERE portfolio = ? AND symbol = ?",
                (self.name, symbol),
            ).fetchone()

            if side == "BUY":
                cost = price * quantity
                if cost > cash + 1e-6:
                    err = f"Insufficient cash: ${cash:.2f} available, ${cost:.2f} required"
                    c.execute(
                        "INSERT INTO portfolio_orders (portfolio, placed_at, symbol, side, "
                        "quantity, fill_price, success, error) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                        (self.name, _now(), symbol, side, quantity, price, err),
                    )
                    return OrderResult(False, None, symbol, side, quantity, None, err)

                new_cash = cash - cost
                if pos_row:
                    new_qty = pos_row["quantity"] + quantity
                    new_avg = (pos_row["quantity"] * pos_row["avg_cost"] + cost) / new_qty
                    c.execute(
                        "UPDATE portfolio_positions SET quantity = ?, avg_cost = ? "
                        "WHERE portfolio = ? AND symbol = ?",
                        (new_qty, new_avg, self.name, symbol),
                    )
                else:
                    c.execute(
                        "INSERT INTO portfolio_positions (portfolio, symbol, quantity, avg_cost) "
                        "VALUES (?, ?, ?, ?)",
                        (self.name, symbol, quantity, price),
                    )
                c.execute("UPDATE portfolios SET cash = ? WHERE name = ?",
                          (new_cash, self.name))
            else:  # SELL
                held_qty = pos_row["quantity"] if pos_row else 0.0
                if quantity > held_qty + 1e-6:
                    err = f"Selling {quantity} but only hold {held_qty}"
                    c.execute(
                        "INSERT INTO portfolio_orders (portfolio, placed_at, symbol, side, "
                        "quantity, fill_price, success, error) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                        (self.name, _now(), symbol, side, quantity, price, err),
                    )
                    return OrderResult(False, None, symbol, side, quantity, None, err)

                proceeds = price * quantity
                new_cash = cash + proceeds
                new_qty = held_qty - quantity
                if new_qty < 1e-6:
                    c.execute(
                        "DELETE FROM portfolio_positions WHERE portfolio = ? AND symbol = ?",
                        (self.name, symbol),
                    )
                else:
                    c.execute(
                        "UPDATE portfolio_positions SET quantity = ? WHERE portfolio = ? AND symbol = ?",
                        (new_qty, self.name, symbol),
                    )
                c.execute("UPDATE portfolios SET cash = ? WHERE name = ?",
                          (new_cash, self.name))

            cur = c.execute(
                "INSERT INTO portfolio_orders (portfolio, placed_at, symbol, side, "
                "quantity, fill_price, success, error) VALUES (?, ?, ?, ?, ?, ?, 1, NULL)",
                (self.name, _now(), symbol, side, quantity, price),
            )
            order_id = cur.lastrowid

        return OrderResult(True, str(order_id), symbol, side, quantity, price)

    # ── Equity curve ───────────────────────────────────────────────

    def snapshot_equity(self) -> AccountSummary:
        """Record current cash + market_value to portfolio_equity_curve."""
        summary = self.account_summary()
        market_value = summary.equity - summary.cash
        with _conn(self._db) as c:
            c.execute(
                "INSERT OR REPLACE INTO portfolio_equity_curve "
                "(portfolio, snapshot_at, cash, market_value) VALUES (?, ?, ?, ?)",
                (self.name, _now(), summary.cash, market_value),
            )
        return summary

    def equity_curve(self) -> list[dict]:
        with _conn(self._db) as c:
            return [dict(r) for r in c.execute(
                "SELECT snapshot_at, cash, market_value FROM portfolio_equity_curve "
                "WHERE portfolio = ? ORDER BY snapshot_at", (self.name,),
            ).fetchall()]

    def order_history(self, limit: int = 50) -> list[dict]:
        with _conn(self._db) as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM portfolio_orders WHERE portfolio = ? "
                "ORDER BY placed_at DESC LIMIT ?",
                (self.name, limit),
            ).fetchall()]

    @property
    def initial_cash(self) -> float:
        with _conn(self._db) as c:
            row = c.execute(
                "SELECT initial_cash FROM portfolios WHERE name = ?", (self.name,),
            ).fetchone()
            return row["initial_cash"] if row else 0.0

    @classmethod
    def list_portfolios(cls, db_path: Path | str | None = None) -> list[dict]:
        init_db(db_path)
        with _conn(db_path) as c:
            return [dict(r) for r in c.execute(
                "SELECT name, created_at, initial_cash, cash, currency FROM portfolios "
                "ORDER BY created_at",
            ).fetchall()]
