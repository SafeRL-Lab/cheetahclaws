"""
Equity backtest engine for US and HK stocks.

Market rules:
  - US: T+0, long/short, fractional shares, zero/low commission
  - HK: T+0, long/short, 100-share lot size, stamp tax + levies
"""
from __future__ import annotations

from .base import BaseEngine, BacktestConfig


class EquityEngine(BaseEngine):
    """Backtest engine for US and HK equities."""

    def __init__(
        self,
        config: BacktestConfig | None = None,
        market: str = "us",
    ):
        super().__init__(config)
        self.market = market.lower()

    def can_execute(self, symbol: str, side: str, bar_idx: int) -> bool:
        """US and HK are both T+0 — always executable."""
        return True

    def calc_commission(self, price: float, quantity: float, side: str) -> float:
        """Calculate commission based on market.

        US: flat rate from config (default 0.1%)
        HK: stamp tax (0.1%) + broker (0.015%) + SFC levy + CCASS fee
        """
        notional = price * quantity
        if self.market == "hk":
            # HK fee components (bilateral)
            stamp = notional * 0.001       # stamp duty: 0.1%
            broker = notional * 0.00015    # broker commission: 0.015%
            sfc = notional * 0.0000278     # SFC levy
            frc = notional * 0.0000015     # FRC levy
            ccass = notional * 0.00002     # CCASS settlement
            return stamp + broker + sfc + frc + ccass
        # US: use config commission rate
        return notional * self.config.commission

    def apply_slippage(self, price: float, side: str) -> float:
        """Apply directional slippage."""
        slip = self.config.slippage
        if side in ("long", "buy"):
            return price * (1 + slip)
        return price * (1 - slip)

    def round_quantity(self, quantity: float) -> float:
        """Round to market-appropriate lot size.

        HK: round down to nearest 100 shares
        US: round to 2 decimals (fractional shares)
        """
        if self.market == "hk":
            return int(quantity / 100) * 100
        return round(quantity, 2)
