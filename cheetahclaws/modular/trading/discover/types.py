"""Shared types for the discovery layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Discovery:
    """One discovery hit. Multiple sources can flag the same ticker; the
    orchestrator merges them by symbol and aggregates the scores."""
    symbol:    str
    source:    str           # "insider" | "earnings" | "sector" | "momentum-quality" | "anomaly"
    score:     float          # 0..1; meaning depends on source
    reason:    str            # one-liner the user reads
    details:   dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "source": self.source,
            "score": self.score, "reason": self.reason,
            "details": self.details,
        }
