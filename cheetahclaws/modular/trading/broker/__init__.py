"""
Broker abstraction layer.

Lets the rest of the system not care whether trades are paper or live.
The PaperBroker is a thin wrapper over our internal paper_trader SQLite
store; the IBKRBroker is a stub that documents the live-wiring path.

Use `get_broker(mode)` to obtain a backend:
  - mode="paper" → PaperBroker (default, always works)
  - mode="ibkr"  → IBKRBroker (requires `ib_insync` + IB Gateway running)
"""
from __future__ import annotations

from .base import BrokerBackend, OrderResult, AccountSummary
from .paper_backend import PaperBroker
from .ibkr_backend import IBKRBroker


def get_broker(mode: str = "paper", **kwargs) -> BrokerBackend:
    """Return the broker backend for the requested mode."""
    mode = mode.lower().strip()
    if mode == "paper":
        return PaperBroker(**kwargs)
    if mode == "ibkr":
        return IBKRBroker(**kwargs)
    raise ValueError(f"Unknown broker mode: {mode!r}. Use 'paper' or 'ibkr'.")


__all__ = ["BrokerBackend", "OrderResult", "AccountSummary",
           "PaperBroker", "IBKRBroker", "get_broker"]
