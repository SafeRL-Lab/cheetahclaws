"""
verifier.py — hard risk rules applied AFTER the LLM portfolio manager decides.

The Portfolio Manager prompt asks the LLM to suggest position size, stop
loss, and take profit. LLMs are wishy-washy on discipline ("recommend
8-12% position with -10% stop") and forget about the rest of your book.
This module enforces non-negotiable rules:

  - Single position cap (default 5% of portfolio)
  - Sector concentration cap (default 25%)
  - Stop discipline (no stops wider than 10%)
  - Earnings blackout (no full-size buys within N days of earnings)
  - Total exposure cap across open paper trades

It reads the current open-position book from paper_trader and either
APPROVES the trade as-is, ADJUSTS it (e.g., shrinks position), or
REJECTS with explanation. Returns a Verdict that callers pretty-print.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paper_trader, earnings


# ── Tunable rules ─────────────────────────────────────────────────────────

DEFAULT_RULES: dict[str, float] = {
    "max_single_position_pct":   5.0,
    "max_sector_pct":            25.0,
    "max_total_exposure_pct":    80.0,
    "max_stop_loss_pct":         10.0,
    "min_take_profit_pct":       5.0,
    "earnings_blackout_days":    3.0,
    "earnings_blackout_size_pct": 2.5,  # cap during blackout
}


@dataclass
class Verdict:
    status: str                    # "approve" | "adjust" | "reject"
    reasons: list[str] = field(default_factory=list)
    adjustments: dict[str, Any] = field(default_factory=dict)
    rule_book: dict[str, float] = field(default_factory=dict)

    def as_markdown(self) -> str:
        emoji = {"approve": "✅", "adjust": "⚙️", "reject": "🛑"}[self.status]
        title = {"approve": "Approved", "adjust": "Approved with adjustments",
                 "reject": "REJECTED"}[self.status]
        lines = [f"## {emoji} Risk Verifier: {title}"]
        for r in self.reasons:
            lines.append(f"- {r}")
        if self.adjustments:
            lines.append("")
            lines.append("**Adjusted plan:**")
            for k, v in self.adjustments.items():
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)


def verify_proposal(
    symbol: str,
    signal: str,
    position_size_pct: float | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    sector: str | None = None,
    rules: dict[str, float] | None = None,
    db_path: Path | None = None,
    skip_earnings_check: bool = False,
) -> Verdict:
    """Validate a proposed trade against risk rules and current book."""
    R = {**DEFAULT_RULES, **(rules or {})}
    v = Verdict(status="approve", rule_book=R)

    # Non-directional signals don't allocate capital — pass through.
    if signal in ("HOLD", "SELL", "UNDERWEIGHT"):
        v.reasons.append(f"{signal} doesn't allocate new capital — no risk check needed.")
        return v

    # Position size cap
    if position_size_pct is None:
        v.status = "adjust"
        v.reasons.append("No position size specified — defaulting to a defensive 2%.")
        v.adjustments["position_size_pct"] = 2.0
        position_size_pct = 2.0
    elif position_size_pct > R["max_single_position_pct"]:
        v.status = "adjust"
        v.reasons.append(
            f"Position size {position_size_pct}% exceeds single-name cap "
            f"({R['max_single_position_pct']}%) — capped."
        )
        v.adjustments["position_size_pct"] = R["max_single_position_pct"]
        position_size_pct = R["max_single_position_pct"]

    # Sector cap (using the live paper-trader book)
    if sector:
        book = paper_trader.open_position_summary(db_path=db_path)
        existing = book["by_sector_pct"].get(sector, 0.0)
        if existing + position_size_pct > R["max_sector_pct"]:
            allowed = max(R["max_sector_pct"] - existing, 0.0)
            if allowed < 0.5:
                v.status = "reject"
                v.reasons.append(
                    f"Sector '{sector}' already at {existing:.1f}% of book; "
                    f"cap is {R['max_sector_pct']}%. No room for this trade."
                )
                return v
            v.status = "adjust"
            v.reasons.append(
                f"Sector '{sector}' exposure would exceed {R['max_sector_pct']}% "
                f"(have {existing:.1f}%, adding {position_size_pct}%) — shrinking to {allowed:.1f}%."
            )
            v.adjustments["position_size_pct"] = round(allowed, 1)
            position_size_pct = allowed

    # Total exposure cap
    book = paper_trader.open_position_summary(db_path=db_path)
    if book["total_exposure_pct"] + position_size_pct > R["max_total_exposure_pct"]:
        v.status = "reject"
        v.reasons.append(
            f"Total exposure would hit {book['total_exposure_pct'] + position_size_pct:.1f}% "
            f"(cap {R['max_total_exposure_pct']}%). Close winners before adding."
        )
        return v

    # Stop discipline
    if stop_loss_pct is None:
        v.status = "adjust"
        v.reasons.append("No stop loss specified — defaulting to -7%.")
        v.adjustments["stop_loss_pct"] = 7.0
    elif stop_loss_pct > R["max_stop_loss_pct"]:
        v.status = "adjust"
        v.reasons.append(
            f"Stop {stop_loss_pct}% wider than {R['max_stop_loss_pct']}% — tightening."
        )
        v.adjustments["stop_loss_pct"] = R["max_stop_loss_pct"]
    elif stop_loss_pct < 1.0:
        v.status = "adjust"
        v.reasons.append(
            f"Stop {stop_loss_pct}% is too tight — likely to be stopped on noise. "
            f"Setting to 4%."
        )
        v.adjustments["stop_loss_pct"] = 4.0

    # Take profit sanity
    if take_profit_pct is not None and take_profit_pct < R["min_take_profit_pct"]:
        v.reasons.append(
            f"Take-profit {take_profit_pct}% is < {R['min_take_profit_pct']}% — "
            f"asymmetric reward/risk; consider widening or removing."
        )

    # Earnings blackout
    if not skip_earnings_check:
        e = earnings.upcoming_earnings(symbol)
        days = e.get("days_until")
        if days is not None and days <= R["earnings_blackout_days"]:
            cap = R["earnings_blackout_size_pct"]
            current = v.adjustments.get("position_size_pct", position_size_pct)
            if current > cap:
                v.status = "adjust"
                v.reasons.append(
                    f"Earnings in {days} days ({e.get('date')}) — blackout reduces "
                    f"max size to {cap}%."
                )
                v.adjustments["position_size_pct"] = cap

    if v.status == "approve" and not v.reasons:
        v.reasons.append("All checks passed.")
    return v
