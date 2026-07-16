"""Tests for the upgraded /trading pipeline.

Covers:
  - paper_trader: open/close/snapshot/list, signal validation, return math
                  (long vs short), watchlist CRUD, Phase-5 parser
  - calibration: hit-rate aggregation, edge-vs-zero t-stat, by-confidence buckets
  - verifier: position-cap clamp, sector-cap rejection, stop discipline,
              earnings blackout enforcement
  - walk_forward: per-chunk metrics, stability verdict
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cheetahclaws.modular.trading import paper_trader, calibration, verifier
from cheetahclaws.modular.trading.engines.base import BaseEngine, BacktestConfig
from cheetahclaws.modular.trading.engines.equity import EquityEngine


# ── paper_trader ──────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path) -> Path:
    """Fresh paper_trades.db per test."""
    p = tmp_path / "paper.db"
    paper_trader.init_db(p)
    return p


def test_open_close_long_trade_realized_return(db):
    tid = paper_trader.open_trade(
        symbol="AAPL", signal="BUY", confidence="High",
        entry_price=100.0, position_size_pct=3.0,
        stop_loss_pct=7.0, take_profit_pct=15.0, db_path=db,
    )
    closed = paper_trader.close_trade(tid, close_price=110.0, db_path=db)
    assert closed is not None
    assert closed.status == "closed"
    assert closed.realized_return_pct == pytest.approx(10.0)


def test_short_signal_inverts_realized_return(db):
    """SELL/UNDERWEIGHT should profit when price drops."""
    tid = paper_trader.open_trade(
        symbol="AAPL", signal="SELL", confidence="Medium",
        entry_price=100.0, db_path=db,
    )
    closed = paper_trader.close_trade(tid, close_price=90.0, db_path=db)
    assert closed.realized_return_pct == pytest.approx(10.0)  # short profited 10%

    tid2 = paper_trader.open_trade(
        symbol="AAPL", signal="UNDERWEIGHT", confidence="Medium",
        entry_price=100.0, db_path=db,
    )
    closed2 = paper_trader.close_trade(tid2, close_price=110.0, db_path=db)
    assert closed2.realized_return_pct == pytest.approx(-10.0)  # underweight lost 10%


def test_invalid_signal_or_confidence_rejected(db):
    with pytest.raises(ValueError):
        paper_trader.open_trade(symbol="X", signal="MOON", confidence="High", db_path=db)
    with pytest.raises(ValueError):
        paper_trader.open_trade(symbol="X", signal="BUY", confidence="Cosmic", db_path=db)


def test_snapshot_records_unrealized(db):
    tid = paper_trader.open_trade(
        symbol="X", signal="BUY", confidence="High", entry_price=50.0, db_path=db,
    )
    unreal = paper_trader.add_snapshot(tid, price=60.0, db_path=db)
    assert unreal == pytest.approx(20.0)


def test_open_position_summary_tracks_sector_exposure(db):
    paper_trader.open_trade(symbol="AAPL", signal="BUY", confidence="High",
                            position_size_pct=3.0, sector="Tech", db_path=db)
    paper_trader.open_trade(symbol="MSFT", signal="BUY", confidence="High",
                            position_size_pct=4.0, sector="Tech", db_path=db)
    paper_trader.open_trade(symbol="JPM", signal="BUY", confidence="Medium",
                            position_size_pct=2.0, sector="Financials", db_path=db)
    s = paper_trader.open_position_summary(db_path=db)
    assert s["open_count"] == 3
    assert s["total_exposure_pct"] == pytest.approx(9.0)
    assert s["by_sector_pct"]["Tech"] == pytest.approx(7.0)
    assert s["by_sector_pct"]["Financials"] == pytest.approx(2.0)


def test_watchlist_add_remove_list(db):
    paper_trader.watchlist_add("aapl", db_path=db)
    paper_trader.watchlist_add("NVDA", note="GPU thesis", db_path=db)
    wl = paper_trader.watchlist_list(db_path=db)
    syms = sorted(e["symbol"] for e in wl)
    assert syms == ["AAPL", "NVDA"]  # uppercased on insert
    assert paper_trader.watchlist_remove("AAPL", db_path=db) is True
    assert paper_trader.watchlist_remove("DOES_NOT_EXIST", db_path=db) is False
    assert len(paper_trader.watchlist_list(db_path=db)) == 1


def test_phase5_parser_extracts_structured_fields():
    text = """
## Phase 5: FINAL RATING
**RATING: BUY**
**Summary**: Strong technicals + earnings momentum.
**Plan**: Entry at market. Position Size 3.5%. Stop Loss -7%. Take Profit +15%. Time Horizon: 6 months
**Top 3 Risks**:
1. Macro reversal
2. Sector rotation
3. Valuation compression
**Conviction**: High
"""
    parsed = paper_trader._parse_phase5(text)
    assert parsed["signal"] == "BUY"
    assert parsed["confidence"] == "High"
    assert parsed["position_size_pct"] == pytest.approx(3.5)
    assert parsed["stop_loss_pct"] == pytest.approx(7.0)
    assert parsed["take_profit_pct"] == pytest.approx(15.0)
    assert "6 months" in parsed["time_horizon"]


def test_phase5_parser_returns_none_without_rating():
    assert paper_trader._parse_phase5("just some text, no rating block") is None


# ── calibration ───────────────────────────────────────────────────────────

def test_calibration_distinguishes_high_from_low(db):
    """When High-conviction trades win and Low ones lose, the diagnosis fires."""
    for _ in range(5):
        tid = paper_trader.open_trade("X", "BUY", "High", entry_price=100.0, db_path=db)
        paper_trader.close_trade(tid, close_price=110.0, db_path=db)
    for _ in range(5):
        tid = paper_trader.open_trade("Y", "BUY", "Low", entry_price=100.0, db_path=db)
        paper_trader.close_trade(tid, close_price=95.0, db_path=db)

    stats = calibration.compute_calibration(db_path=db)
    assert stats["total_closed"] == 10
    assert stats["by_confidence"]["High"]["mean_return_pct"] == pytest.approx(10.0)
    assert stats["by_confidence"]["Low"]["mean_return_pct"] == pytest.approx(-5.0)
    assert "High-conviction outperforms Low" in stats["calibration_check"]


def test_calibration_handles_empty_db(db):
    stats = calibration.compute_calibration(db_path=db)
    assert stats["total_closed"] == 0
    assert "Insufficient" in stats["calibration_check"]
    rendered = calibration.render_calibration_report(stats)
    assert "No closed paper trades" in rendered


def test_calibration_report_renders_markdown(db):
    for i in range(3):
        tid = paper_trader.open_trade("Z", "BUY", "Medium", entry_price=100.0, db_path=db)
        paper_trader.close_trade(tid, close_price=105.0, db_path=db)
    stats = calibration.compute_calibration(db_path=db)
    report = calibration.render_calibration_report(stats)
    assert "# Trading Agent Calibration Report" in report
    assert "## By Confidence" in report
    assert "## By Signal" in report
    assert "BUY" in report


# ── verifier ──────────────────────────────────────────────────────────────

def test_verifier_caps_oversize_position(db):
    v = verifier.verify_proposal(
        symbol="AAPL", signal="BUY",
        position_size_pct=12.0, stop_loss_pct=5.0,
        sector="Tech", db_path=db, skip_earnings_check=True,
    )
    assert v.status == "adjust"
    assert v.adjustments["position_size_pct"] == 5.0


def test_verifier_rejects_when_sector_full(db):
    paper_trader.open_trade("X", "BUY", "High", position_size_pct=23.0,
                            sector="Tech", db_path=db)
    v = verifier.verify_proposal(
        symbol="NVDA", signal="BUY", position_size_pct=4.0,
        stop_loss_pct=7.0, sector="Tech",
        db_path=db, skip_earnings_check=True,
    )
    # 23% existing + 4% new = 27% > 25% cap; allowed = 2% (still > 0.5 floor)
    assert v.status == "adjust"
    assert v.adjustments["position_size_pct"] == pytest.approx(2.0, abs=0.1)


def test_verifier_hard_rejects_when_no_sector_room(db):
    paper_trader.open_trade("X", "BUY", "High", position_size_pct=24.9,
                            sector="Tech", db_path=db)
    v = verifier.verify_proposal(
        symbol="NVDA", signal="BUY", position_size_pct=4.0,
        stop_loss_pct=7.0, sector="Tech",
        db_path=db, skip_earnings_check=True,
    )
    assert v.status == "reject"
    assert any("Sector 'Tech'" in r for r in v.reasons)


def test_verifier_tightens_wide_stops(db):
    v = verifier.verify_proposal(
        symbol="X", signal="BUY",
        position_size_pct=3.0, stop_loss_pct=20.0,
        db_path=db, skip_earnings_check=True,
    )
    assert v.status == "adjust"
    assert v.adjustments["stop_loss_pct"] == 10.0


def test_verifier_flags_too_tight_stops(db):
    v = verifier.verify_proposal(
        symbol="X", signal="BUY",
        position_size_pct=3.0, stop_loss_pct=0.5,
        db_path=db, skip_earnings_check=True,
    )
    assert v.status == "adjust"
    assert v.adjustments["stop_loss_pct"] == 4.0


def test_verifier_passes_through_hold_signals(db):
    v = verifier.verify_proposal(
        symbol="X", signal="HOLD",
        position_size_pct=None, stop_loss_pct=None,
        db_path=db, skip_earnings_check=True,
    )
    assert v.status == "approve"


def test_verifier_caps_size_during_earnings_blackout(db, monkeypatch):
    # Stub upcoming_earnings to claim earnings in 2 days
    monkeypatch.setattr(
        verifier.earnings, "upcoming_earnings",
        lambda sym: {"date": "2026-05-12", "days_until": 2, "session": "AMC"},
    )
    v = verifier.verify_proposal(
        symbol="AAPL", signal="BUY",
        position_size_pct=4.0, stop_loss_pct=7.0,
        db_path=db,
    )
    assert v.status == "adjust"
    assert v.adjustments["position_size_pct"] == 2.5  # earnings_blackout_size_pct
    assert any("Earnings in 2 days" in r for r in v.reasons)


def test_verifier_rejects_when_total_exposure_capped(db):
    paper_trader.open_trade("A", "BUY", "High", position_size_pct=78.0,
                            sector="Mixed", db_path=db)
    v = verifier.verify_proposal(
        symbol="B", signal="BUY", position_size_pct=4.0,
        stop_loss_pct=7.0,
        db_path=db, skip_earnings_check=True,
    )
    assert v.status == "reject"


# ── walk-forward ─────────────────────────────────────────────────────────

def _synthetic_uptrend(n_bars: int = 500, drift: float = 0.0008) -> list[dict]:
    """Generate a synthetic uptrending series (no random noise => deterministic)."""
    import math
    rows = []
    price = 100.0
    for i in range(n_bars):
        # Sinusoid + drift, so dual-MA crosses both directions, walking forward.
        price *= (1.0 + drift + 0.005 * math.sin(i / 20))
        rows.append({
            "date": f"2024-{1 + i // 30:02d}-{1 + i % 30:02d}",
            "open": round(price * 0.999, 2),
            "high": round(price * 1.002, 2),
            "low":  round(price * 0.998, 2),
            "close": round(price, 2),
            "volume": 1_000_000,
        })
    return rows


def test_walk_forward_produces_per_chunk_metrics():
    from cheetahclaws.modular.trading.tools import _build_strategy
    rows = _synthetic_uptrend(500)
    engine = EquityEngine(BacktestConfig(initial_capital=100_000.0), market="us")
    strat = _build_strategy("dual_ma")
    result = engine.walk_forward(strat, {"AAPL": rows}, n_splits=5)
    assert len(result["splits"]) == 5
    for s in result["splits"]:
        assert "metrics" in s
        assert "start_date" in s and "end_date" in s
    assert "verdict" in result["stability"]


def test_walk_forward_too_short_returns_diagnostic():
    from cheetahclaws.modular.trading.tools import _build_strategy
    rows = _synthetic_uptrend(100)  # too short for 5×60-bar splits
    engine = EquityEngine(BacktestConfig(initial_capital=100_000.0))
    strat = _build_strategy("dual_ma")
    result = engine.walk_forward(strat, {"X": rows}, n_splits=5, min_split_bars=60)
    # Either runs with fewer splits or returns "Not enough" — both are valid
    assert "verdict" in result["stability"]


def test_strategy_factory_validates_unknown_name():
    from cheetahclaws.modular.trading.tools import _build_strategy
    with pytest.raises(ValueError, match="Unknown strategy"):
        _build_strategy("moonshot_2x")


# ── Integration: analyze prompt absorbs macro/earnings/book context ───────

def test_analysis_prompt_includes_book_block_when_positions_exist(db, monkeypatch):
    """Open paper trades should appear in the prompt so the LLM sees its book."""
    paper_trader.open_trade("X", "BUY", "High", position_size_pct=4.0,
                            sector="Tech", db_path=db)

    # Point _parse_phase5's underlying paper_trader to our temp db
    monkeypatch.setattr(paper_trader, "_DB_PATH", db)
    # Stub macro + earnings to keep test offline
    from cheetahclaws.modular.trading import macro as macro_mod, earnings as earnings_mod
    monkeypatch.setattr(macro_mod, "render_macro_context", lambda: "")
    monkeypatch.setattr(earnings_mod, "render_earnings_warning", lambda s: "")

    from cheetahclaws.modular.trading.cmd import _build_analysis_prompt
    prompt = _build_analysis_prompt("NVDA", "2026-05-07", {
        "technical": "RSI 55, above 50d", "fundamental": "PE 30",
        "news": "Strong demand reported",
    })
    assert "Current Open Paper Trades" in prompt
    assert "Tech" in prompt
