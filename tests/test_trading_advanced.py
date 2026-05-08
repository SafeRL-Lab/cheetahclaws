"""Tests for the advanced trading layer (alt-data, broker, optimizer, managed, ML).

The headline test is `test_managed_portfolio_lifecycle_with_mocked_quotes` —
it simulates the user's "$100, check in a week" scenario end-to-end.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


def _have(*mods: str) -> bool:
    """True iff every module in *mods* is importable on the current install.

    Several tests below exercise the optimizer / ML stacker / managed
    portfolio paths, all of which depend on numpy + scipy + sklearn from
    the ``[trading]`` extra.  CI's lean ``[web,autosuggest]`` install
    deliberately does not pull these in, so the tests skip cleanly
    instead of failing — a missing optional dep is not a regression.
    """
    return all(importlib.util.find_spec(m) is not None for m in mods)


_skip_if_no_numpy = pytest.mark.skipif(
    not _have("numpy"),
    reason="numpy not installed (needs [trading] extra)",
)
_skip_if_no_scipy = pytest.mark.skipif(
    not _have("numpy", "scipy"),
    reason="numpy/scipy not installed (needs [trading] extra)",
)
_skip_if_no_sklearn = pytest.mark.skipif(
    not _have("numpy", "sklearn"),
    reason="numpy/sklearn not installed (needs [trading] extra)",
)


# ── Alt-data: SEC EDGAR insider ───────────────────────────────────────────

def test_insider_summary_soft_fails_when_sec_unreachable(monkeypatch):
    from modular.trading.alt_data import insider
    monkeypatch.setattr(insider, "_http_get", lambda url: None)
    assert insider.fetch_recent_insider_filings("AAPL") == []
    assert insider.render_insider_summary("AAPL") == ""


def test_insider_filters_to_form4_within_window(monkeypatch):
    from modular.trading.alt_data import insider
    import json

    def fake_get(url: str):
        if "company_tickers.json" in url:
            return json.dumps({"0": {"ticker": "TEST", "cik_str": 1234}}).encode()
        if "submissions/CIK" in url:
            from datetime import datetime, timedelta
            recent_dt = (datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d")
            old_dt    = (datetime.utcnow() - timedelta(days=400)).strftime("%Y-%m-%d")
            return json.dumps({
                "filings": {"recent": {
                    "form":            ["4", "10-K", "4/A", "4"],
                    "accessionNumber": ["acc-1", "acc-2", "acc-3", "acc-4"],
                    "filingDate":      [recent_dt, recent_dt, recent_dt, old_dt],
                    "primaryDocument": ["a.html", "b.html", "c.html", "d.html"],
                }}
            }).encode()
        return None

    monkeypatch.setattr(insider, "_http_get", fake_get)
    filings = insider.fetch_recent_insider_filings("TEST", days=90)
    forms = [f["form"] for f in filings]
    assert forms.count("4") + forms.count("4/A") == len(filings)
    # Old Form 4 (>90d) should be filtered out
    assert all("acc-4" != f["accession"] for f in filings)


# ── Alt-data: trends soft-fail without pytrends ──────────────────────────

def test_trends_soft_fail_when_pytrends_missing(monkeypatch):
    from modular.trading.alt_data import trends
    # Ensure pytrends import path raises
    import sys
    monkeypatch.setitem(sys.modules, "pytrends.request", None)
    out = trends.fetch_interest("NVDA")
    assert isinstance(out, dict)


def test_trends_render_empty_string_when_unavailable(monkeypatch):
    from modular.trading.alt_data import trends
    monkeypatch.setattr(trends, "fetch_interest", lambda s, lookback_days=90: {})
    assert trends.render_trends_block("NVDA") == ""


# ── Alt-data: sentiment ──────────────────────────────────────────────────

def test_sentiment_returns_empty_when_no_headlines(monkeypatch):
    from modular.trading.alt_data import sentiment
    monkeypatch.setattr(sentiment, "fetch_recent_headlines", lambda *a, **k: [])
    assert sentiment.render_sentiment_block("NVDA") == ""


def test_sentiment_renders_with_aux_scores(monkeypatch):
    from modular.trading.alt_data import sentiment
    monkeypatch.setattr(sentiment, "fetch_recent_headlines",
                        lambda *a, **k: [
                            {"title": "NVDA beats on Q4", "publisher": "Reuters", "link": "", "ts": 0},
                            {"title": "NVDA faces antitrust probe", "publisher": "WSJ", "link": "", "ts": 0},
                        ])
    monkeypatch.setattr(
        sentiment, "_score_with_aux_model",
        lambda sym, hl: {"scores": [7, -4], "mean": 1.5, "n_pos": 1, "n_neg": 1,
                         "reasoning": "mixed"},
    )
    block = sentiment.render_sentiment_block("NVDA")
    assert "## News Sentiment" in block
    assert "+1.5/10" in block
    assert "MIXED" in block
    assert "antitrust" in block


# ── Portfolio optimizer ──────────────────────────────────────────────────

def _make_uptrend(n=120, start=100.0, drift=0.001, noise_seed: int = 0) -> list[float]:
    """Realistic-ish series: drift + small idiosyncratic noise so the cov
    matrix is actually invertible. Without noise the optimizer has no
    reason to diversify and SLSQP can violate sector caps."""
    import math
    out = []
    p = start
    for i in range(n):
        # Deterministic pseudo-noise from a simple LCG seeded per call so
        # tests stay reproducible without depending on numpy/random global state.
        noise = math.sin((i + 1) * (noise_seed + 1) * 0.317) * 0.012
        p *= (1.0 + drift + noise)
        out.append(p)
    return out


@_skip_if_no_scipy
def test_optimizer_returns_long_only_caps_weight():
    from modular.trading.portfolio import Candidate, optimize
    cands = [
        Candidate("A", _make_uptrend(120, 100, 0.0010, noise_seed=1)),
        Candidate("B", _make_uptrend(120, 50,  0.0005, noise_seed=2)),
        Candidate("C", _make_uptrend(120, 200, 0.0008, noise_seed=3)),
    ]
    r = optimize(cands, max_weight=0.40)
    assert all(0 <= w <= 0.40 + 1e-6 for w in r.weights.values())
    assert sum(r.weights.values()) <= 1.0 + 1e-6


@_skip_if_no_scipy
def test_optimizer_handles_too_short_history():
    from modular.trading.portfolio import Candidate, optimize
    cands = [Candidate("A", [100.0] * 10)]
    r = optimize(cands)
    assert "insufficient history" in r.diagnostics.get("reason", "")


@_skip_if_no_scipy
def test_optimizer_respects_sector_caps():
    from modular.trading.portfolio import Candidate, optimize
    cands = [
        Candidate("A", _make_uptrend(120, noise_seed=1), sector="Tech"),
        Candidate("B", _make_uptrend(120, drift=0.0008, noise_seed=2), sector="Tech"),
        Candidate("C", _make_uptrend(120, drift=0.0006, noise_seed=3), sector="Finance"),
    ]
    r = optimize(cands, max_weight=0.40, sector_caps={"Tech": 0.30})
    tech_total = r.weights.get("A", 0.0) + r.weights.get("B", 0.0)
    assert tech_total <= 0.30 + 0.01  # small SLSQP slack


@_skip_if_no_scipy
def test_optimization_report_renders():
    from modular.trading.portfolio import Candidate, optimize, render_optimization_report
    cands = [Candidate("A", _make_uptrend(120))]
    r = optimize(cands)
    md = render_optimization_report(r)
    assert "# Portfolio Optimization" in md


# ── Broker abstraction (paper backend) ────────────────────────────────────

@pytest.fixture
def fresh_broker(tmp_path, monkeypatch):
    """PaperBroker with isolated DB and a deterministic price quote."""
    from modular.trading.broker.paper_backend import PaperBroker

    db = tmp_path / "managed.db"
    broker = PaperBroker(name="test_account", db_path=db, initial_cash=100.0)

    # Pin quote to a deterministic value
    monkeypatch.setattr(broker, "quote", lambda sym: {"AAPL": 50.0, "MSFT": 25.0}.get(sym))
    return broker


def test_broker_initial_account_summary(fresh_broker):
    s = fresh_broker.account_summary()
    assert s.cash == 100.0
    assert s.equity == 100.0
    assert s.open_positions_count == 0


def test_broker_buy_then_sell_round_trip(fresh_broker):
    r1 = fresh_broker.place_market_order("AAPL", "BUY", 1.0)
    assert r1.success
    s_after_buy = fresh_broker.account_summary()
    assert s_after_buy.cash == pytest.approx(50.0)
    assert s_after_buy.open_positions_count == 1

    r2 = fresh_broker.place_market_order("AAPL", "SELL", 1.0)
    assert r2.success
    s_after_sell = fresh_broker.account_summary()
    assert s_after_sell.cash == pytest.approx(100.0)
    assert s_after_sell.open_positions_count == 0


def test_broker_rejects_buy_above_cash(fresh_broker):
    r = fresh_broker.place_market_order("AAPL", "BUY", 5.0)  # 5×50 = 250 > 100
    assert not r.success
    assert "Insufficient cash" in (r.error or "")


def test_broker_rejects_sell_more_than_held(fresh_broker):
    fresh_broker.place_market_order("AAPL", "BUY", 1.0)
    r = fresh_broker.place_market_order("AAPL", "SELL", 2.0)
    assert not r.success
    assert "only hold" in (r.error or "")


def test_broker_average_cost_updates_on_add(fresh_broker, monkeypatch):
    # 0.5 share @ $50 = $25; 0.5 share @ $60 = $30; total $55 → avg = $55
    fresh_broker.place_market_order("AAPL", "BUY", 0.5)
    monkeypatch.setattr(fresh_broker, "quote", lambda s: 60.0)
    fresh_broker.place_market_order("AAPL", "BUY", 0.5)
    pos = fresh_broker.positions()[0]
    # Weighted avg cost: (0.5*50 + 0.5*60) / 1.0 = 55
    assert pos.avg_cost == pytest.approx(55.0)


def test_ibkr_stub_reports_setup_required():
    from modular.trading.broker import IBKRBroker
    b = IBKRBroker()
    diag = b.connection_check()
    assert "ok" in diag
    r = b.place_market_order("AAPL", "BUY", 1.0)
    assert not r.success


# ── Managed portfolio mode ───────────────────────────────────────────────

@_skip_if_no_scipy
def test_managed_portfolio_lifecycle_with_mocked_quotes(tmp_path, monkeypatch):
    """End-to-end: $100 → step → check status → simulate price move → step again.

    Uses a deterministic quote function so the test doesn't hit the network.
    Skipped when [trading] extras aren't installed — managed.step calls
    into portfolio.optimize which requires numpy + scipy.
    """
    from modular.trading import managed
    from modular.trading.broker.paper_backend import PaperBroker

    db = tmp_path / "managed.db"

    # Patch fetch_market_data and fetch_current_price for deterministic universe
    def synthetic_history(sym, start_date=None, end_date=None, interval="1d", source="auto"):
        # 250 bars of mild uptrend; same shape per sym so MV picks based on vol
        return {
            "symbol": sym, "source": "synthetic",
            "data": [{"date": f"2025-01-{(i % 28) + 1:02d}", "open": 100.0 + i * 0.5,
                      "high": 100.5 + i * 0.5, "low": 99.5 + i * 0.5,
                      "close": 100.0 + i * 0.5, "volume": 1_000_000}
                     for i in range(250)],
            "info": {}, "error": None,
        }
    monkeypatch.setattr("modular.trading.data.fetchers.fetch_market_data", synthetic_history)

    quote_value = {"v": 100.0}
    monkeypatch.setattr(
        "modular.trading.data.fetchers.fetch_current_price",
        lambda sym: {"price": quote_value["v"]},
    )

    # Override the universe so it doesn't depend on watchlist state
    monkeypatch.setattr(managed, "_universe_for", lambda b, db_path=None: ["AAPL", "MSFT", "GOOG"])

    # Step 1 — initial allocation from $100
    broker = managed.start_portfolio("hundred", initial_cash=100.0, db_path=db)
    assert broker.account_summary().equity == 100.0

    result1 = managed.step("hundred", db_path=db)
    assert isinstance(result1.target_weights, dict)
    assert len(result1.orders) > 0
    after1 = managed.status("hundred", db_path=db)
    # Should have spent some cash on positions
    assert after1["cash"] < 100.0
    assert after1["open_positions_count"] >= 1
    # Equity is approximately preserved (no spread / commission in PaperBroker)
    assert abs(after1["equity"] - 100.0) < 5.0

    # Simulate price up 10% — equity should reflect gain
    quote_value["v"] = 110.0
    after2 = managed.status("hundred", db_path=db)
    assert after2["pnl_dollars"] > 0
    assert after2["pnl_pct"] > 0

    # Run report — should not crash and contain key sections
    report = managed.report("hundred", db_path=db)
    assert "Managed portfolio" in report
    assert "Holdings" in report

    # Listing portfolios returns ours
    ports = managed.list_portfolios(db_path=db)
    assert any(p["name"] == "hundred" for p in ports)


@_skip_if_no_scipy
def test_managed_portfolio_dry_run_places_no_orders(tmp_path, monkeypatch):
    from modular.trading import managed

    db = tmp_path / "managed.db"
    monkeypatch.setattr(
        "modular.trading.data.fetchers.fetch_market_data",
        lambda sym, **kw: {
            "data": [{"date": "2025-01-01", "open": 100, "high": 100, "low": 100,
                      "close": 100 + i * 0.5, "volume": 1000}
                     for i in range(120)],
            "error": None,
        },
    )
    monkeypatch.setattr(
        "modular.trading.data.fetchers.fetch_current_price",
        lambda sym: {"price": 100.0},
    )
    monkeypatch.setattr(managed, "_universe_for", lambda b, db_path=None: ["AAPL"])

    managed.start_portfolio("dryport", initial_cash=100.0, db_path=db)
    result = managed.step("dryport", dry_run=True, db_path=db)
    # dry run = zero orders even though weights computed
    assert result.orders == []
    assert any("Dry run" in n for n in result.notes)


# ── ML stacker ────────────────────────────────────────────────────────────

def _synth_closed_trade(realized_pct: float, signal: str = "BUY",
                       confidence: str = "Medium",
                       sector: str = "Technology", trade_id: int = 0):
    """Make a TradeRecord-shaped object for feature extraction."""
    from modular.trading.paper_trader import TradeRecord
    return TradeRecord(
        id=trade_id, created_at="2025-01-01T00:00:00", symbol="X",
        market="us_equity", signal=signal, confidence=confidence,
        entry_price=100.0, position_size_pct=3.0,
        stop_loss_pct=7.0, take_profit_pct=15.0,
        time_horizon="3m", thesis="t", sector=sector,
        source_run_id=None, status="closed",
        closed_at="2025-02-01T00:00:00", close_price=100.0 + realized_pct,
        realized_return_pct=realized_pct, close_reason="manual",
    )


def test_ml_features_build_dataset_correct_shape():
    from modular.trading.ml.features import build_dataset, feature_columns
    closed = [_synth_closed_trade(5, trade_id=i) for i in range(10)]
    rows, cols = build_dataset(closed)
    assert len(rows) == 10
    assert len(cols) == len(rows[0].features)
    assert all(r.label == 1 for r in rows)


def test_ml_train_returns_diagnostic_when_too_few_samples():
    from modular.trading.ml.stacker import train
    from modular.trading.ml.features import build_dataset
    closed = [_synth_closed_trade(5, trade_id=i) for i in range(3)]
    rows, cols = build_dataset(closed)
    r = train(rows, cols=cols, min_samples=30)
    assert r.cv_auc_mean == 0.0
    assert any("≥ 30" in n for n in r.notes)


@_skip_if_no_sklearn
def test_ml_train_with_mixed_labels(tmp_path):
    # Real model fit — needs the heavy ML stack from the [trading] extra.
    # CI's lean `[web,autosuggest]` install skips this cleanly.
    from modular.trading.ml.stacker import train, predict_proba
    from modular.trading.ml.features import build_dataset

    closed = []
    # 30 winners, 30 losers — give the classifier something to learn
    for i in range(30):
        closed.append(_synth_closed_trade(5, signal="BUY", confidence="High",
                                          sector="Technology", trade_id=i))
    for i in range(30):
        closed.append(_synth_closed_trade(-3, signal="BUY", confidence="Low",
                                          sector="Energy", trade_id=100 + i))
    rows, cols = build_dataset(closed)

    model_path = tmp_path / "stacker.pkl"
    r = train(rows, cols=cols, n_folds=3, model_path=model_path, min_samples=30)
    assert r.n_samples == 60
    assert model_path.exists()
    assert r.cv_auc_mean > 0.55  # confidence/sector are perfectly informative here

    # Predict on a "looks like a winner" feature row
    pred = predict_proba(rows[0].features, model_path=model_path)
    assert "proba_hit" in pred
    assert 0.0 <= pred["proba_hit"] <= 1.0


def test_ml_predict_returns_empty_when_no_model(tmp_path):
    from modular.trading.ml.stacker import predict_proba
    out = predict_proba([0.0] * 17, model_path=tmp_path / "nonexistent.pkl")
    assert out == {}


# ── Review prompt builder ────────────────────────────────────────────────

def test_review_prompt_includes_actions_format(monkeypatch):
    """Review prompt must contain the ACTION line format the parser expects."""
    from modular.trading import macro
    monkeypatch.setattr(macro, "render_macro_context", lambda: "")
    from modular.trading.cmd import _build_review_prompt
    prompt = _build_review_prompt([
        {"id": 1, "symbol": "AAPL", "signal": "BUY", "confidence": "High",
         "entry": 100.0, "current": 110.0, "unrealized_pct": 10.0,
         "size_pct": 3.0, "stop_pct": 7.0, "tp_pct": 15.0, "thesis": "test"},
    ])
    assert "ACTION ID=" in prompt
    assert "DECISION=" in prompt
    assert "HOLD|ADD|TRIM|EXIT" in prompt
