"""
broker/base.py — abstract broker contract.

Same shape works for paper trading (SQLite) and a live broker (IBKR /
Tiger / Futu). Keep the surface tiny: get-account, place-market-order,
get-positions, get-quote.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OrderResult:
    success: bool
    order_id: str | None
    symbol: str
    side: str            # "BUY" | "SELL"
    quantity: float
    fill_price: float | None
    error: str | None = None


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float | None
    market_value: float | None
    unrealized_pnl: float | None


@dataclass
class AccountSummary:
    cash:                float
    equity:              float       # cash + market value of positions
    buying_power:        float
    open_positions_count: int
    currency:            str = "USD"


class BrokerBackend(ABC):
    """Common interface for paper / live broker integrations."""

    @property
    @abstractmethod
    def mode(self) -> str: ...

    @abstractmethod
    def account_summary(self) -> AccountSummary: ...

    @abstractmethod
    def positions(self) -> list[Position]: ...

    @abstractmethod
    def quote(self, symbol: str) -> float | None: ...

    @abstractmethod
    def place_market_order(self, symbol: str, side: str, quantity: float) -> OrderResult:
        """Side: 'BUY' or 'SELL'. Quantity is positive shares."""
        ...
