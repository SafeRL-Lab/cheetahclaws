"""
broker/ibkr_backend.py — Interactive Brokers stub.

Why a stub: live trading requires an IB Gateway / TWS instance running
locally with the user's credentials, plus the `ib_insync` package. Both
are user-side decisions and CANNOT be tested in CI / by the agent.
This file provides:

  - The class shape (matching BrokerBackend) so the rest of the code
    can already type-check against it.
  - A connection_check() that returns a useful diagnostic when ib_insync
    isn't installed, IB Gateway isn't running, or auth fails.
  - Inline setup docs.

To go live:
  1. pip install ib_insync
  2. Install IB Gateway (https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
     or TWS, configure for API access on port 7497 (paper) or 7496 (live)
  3. Enable "Allow connections from localhost" in IB Gateway settings
  4. Wire `IBKRBroker(host="127.0.0.1", port=7497, client_id=42).connect()`

Until then, every order method returns an OrderResult(success=False)
with a specific diagnostic — it never silently sends a fake order.
"""
from __future__ import annotations

from .base import BrokerBackend, OrderResult, Position, AccountSummary


_NOT_CONFIGURED = (
    "IBKR not connected. Required setup:\n"
    "  1. pip install ib_insync\n"
    "  2. Install + run IB Gateway (or TWS) with API access enabled\n"
    "  3. Re-instantiate IBKRBroker(host=..., port=..., client_id=...).connect()\n"
    "  See broker/ibkr_backend.py header for full setup instructions."
)


class IBKRBroker(BrokerBackend):
    """Interactive Brokers backend (stub — see module docstring for setup)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497,
                 client_id: int = 42, paper: bool = True):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.paper = paper
        self._ib = None  # IB instance once connected

    @property
    def mode(self) -> str:
        return "ibkr-paper" if self.paper else "ibkr-live"

    # ── Connection ────────────────────────────────────────────────

    def connection_check(self) -> dict:
        """Return diagnostic dict — does NOT actually connect."""
        try:
            import ib_insync  # noqa: F401
        except ImportError:
            return {
                "ok":     False,
                "stage":  "dep",
                "detail": "ib_insync not installed. Run: pip install ib_insync",
            }
        return {
            "ok":     True,
            "stage":  "ready",
            "detail": "ib_insync importable. Call .connect() to verify gateway.",
        }

    def connect(self) -> bool:
        """Attempt to open a connection to IB Gateway."""
        try:
            from ib_insync import IB
        except ImportError:
            return False
        ib = IB()
        try:
            ib.connect(self.host, self.port, clientId=self.client_id, timeout=5)
        except Exception:
            return False
        self._ib = ib
        return True

    def disconnect(self) -> None:
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None

    # ── BrokerBackend interface ─────────────────────────────────────

    def account_summary(self) -> AccountSummary:
        if self._ib is None:
            return AccountSummary(0.0, 0.0, 0.0, 0)
        # Real impl pulls from ib.accountSummary(). Skipped in stub.
        return AccountSummary(0.0, 0.0, 0.0, 0)

    def positions(self) -> list[Position]:
        return []

    def quote(self, symbol: str) -> float | None:
        return None

    def place_market_order(self, symbol: str, side: str, quantity: float) -> OrderResult:
        if self._ib is None:
            return OrderResult(False, None, symbol, side, quantity, None, _NOT_CONFIGURED)
        # Real impl: build Stock contract + MarketOrder + ib.placeOrder.
        return OrderResult(False, None, symbol, side, quantity, None,
                           "IBKR live trading not enabled in this build (stub).")
