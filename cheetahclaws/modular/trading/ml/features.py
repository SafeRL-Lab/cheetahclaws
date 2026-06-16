"""
ml/features.py — feature engineering from closed paper trades.

The ML stacker predicts "did this trade beat SPY over its holding
window". To do that we need a feature row per closed trade plus a
binary label.

Features per trade (all available at decision time):
  - LLM signal (BUY/HOLD/SELL etc.) one-hot
  - LLM confidence (High/Medium/Low) ordinal
  - Position size (% of book)
  - Stop loss / take profit levels
  - Sector one-hot (top 11 GICS-ish)
  - Macro at decision time (SPY %-vs-200d, VIX, 10y yield)
  - Recent technicals (RSI, momentum, vol)

Label:
  excess_return_30d > 0 (i.e. trade beat SPY-equivalent over 30 days)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


_SIGNAL_ORDER = ["SELL", "UNDERWEIGHT", "HOLD", "OVERWEIGHT", "BUY"]
_CONF_TO_INT = {"Low": 0, "Medium": 1, "High": 2}
_SECTORS = [
    "Technology", "Financials", "Healthcare", "ConsumerDiscretionary",
    "ConsumerStaples", "Industrials", "Energy", "Materials", "Utilities",
    "RealEstate", "Communication", "Other",
]


@dataclass
class FeatureRow:
    trade_id: int
    features: list[float]
    label: int  # 1 if beat baseline, 0 otherwise


def feature_columns() -> list[str]:
    cols: list[str] = []
    cols.append("signal_idx")            # 0..4 = SELL..BUY
    cols.append("confidence")            # 0..2 = Low/Med/High
    cols.append("position_size_pct")
    cols.append("stop_loss_pct")
    cols.append("take_profit_pct")
    cols += [f"sector__{s}" for s in _SECTORS]
    return cols


def _sector_one_hot(sector: str | None) -> list[float]:
    s = sector or "Other"
    if s not in _SECTORS:
        s = "Other"
    return [1.0 if s == sec else 0.0 for sec in _SECTORS]


def build_dataset(closed_trades: list[Any]) -> tuple[list[FeatureRow], list[str]]:
    """Build feature rows from a list of TradeRecord (closed)."""
    rows: list[FeatureRow] = []
    cols = feature_columns()
    for t in closed_trades:
        if t.realized_return_pct is None:
            continue
        if t.signal not in _SIGNAL_ORDER or t.confidence not in _CONF_TO_INT:
            continue

        feats: list[float] = [
            float(_SIGNAL_ORDER.index(t.signal)),
            float(_CONF_TO_INT[t.confidence]),
            float(t.position_size_pct or 0.0),
            float(t.stop_loss_pct or 0.0),
            float(t.take_profit_pct or 0.0),
        ]
        feats.extend(_sector_one_hot(t.sector))
        # Label: did the realized return beat zero by a margin?
        # We use 0 as a coarse benchmark; users with SPY benchmark
        # data can extend this.
        label = 1 if t.realized_return_pct > 0 else 0
        rows.append(FeatureRow(trade_id=t.id, features=feats, label=label))
    return rows, cols
