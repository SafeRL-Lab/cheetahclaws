"""
Crypto backtest engine for perpetual contracts and spot.

Market rules:
  - 24/7 trading, no restrictions
  - Maker/taker fee split
  - Funding fees (every 8 hours for perpetuals)
  - Liquidation checks
  - Fractional sizes (6 decimal places)
"""
from __future__ import annotations

from .base import BaseEngine, BacktestConfig


class CryptoEngine(BaseEngine):
    """Backtest engine for cryptocurrency markets."""

    def __init__(
        self,
        config: BacktestConfig | None = None,
        taker_fee: float = 0.0005,      # 0.05%
        maker_fee: float = 0.0002,      # 0.02%
        funding_rate: float = 0.0001,   # 0.01% per 8h (typical perpetual)
        is_perpetual: bool = False,
    ):
        super().__init__(config)
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.funding_rate = funding_rate
        self.is_perpetual = is_perpetual
        self._funding_applied: set[int] = set()

    def can_execute(self, symbol: str, side: str, bar_idx: int) -> bool:
        """Crypto trades 24/7 — always executable."""
        return True

    def calc_commission(self, price: float, quantity: float, side: str) -> float:
        """Calculate commission with maker/taker split.

        Opens typically hit taker rate, closes hit maker rate.
        """
        notional = price * quantity
        if side in ("long", "short", "buy"):
            return notional * self.taker_fee
        return notional * self.maker_fee

    def apply_slippage(self, price: float, side: str) -> float:
        """Apply directional slippage."""
        slip = self.config.slippage
        if side in ("long", "buy"):
            return price * (1 + slip)
        return price * (1 - slip)

    def on_bar(self, symbol: str, bar: dict, bar_idx: int) -> None:
        """Apply funding fees for perpetual contracts (every 8h equivalent).

        For daily bars, apply funding 3x per day (00:00, 08:00, 16:00).
        Simplified: apply once per bar for daily data.
        """
        if not self.is_perpetual:
            return
        if symbol not in self.positions:
            return
        if bar_idx in self._funding_applied:
            return

        pos = self.positions[symbol]
        price = bar["close"]
        notional = pos.quantity * price
        funding = notional * self.funding_rate

        # Long pays funding, short receives (typical positive funding rate)
        if pos.side == "long":
            self.cash -= funding
        else:
            self.cash += funding

        self._funding_applied.add(bar_idx)
