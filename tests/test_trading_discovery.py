"""Tests for the discovery + ranking + monitor + anomaly layer.

These tests run offline (mocked yfinance / SEC) so they're fast and
don't depend on network or API rate limits.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


# Same skip pattern as test_trading_advanced.py — the stub_yfinance
# fixture instantiates a FakeHist that imports pandas, so any test
# that triggers `Ticker(sym).history()` needs pandas installed.
_skip_if_no_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None,
    reason="pandas not installed (needs [trading] extra)",
)


# ── Universe helpers ─────────────────────────────────────────────────────

def test_resolve_universe_default_returns_sp100():
    from modular.trading.universe import resolve_universe, SP100
    out = resolve_universe(None)
    assert out == SP100
    assert "AAPL" in out
    assert len(out) >= 100


def test_resolve_universe_custom_overrides():
    from modular.trading.universe import resolve_universe
    out = resolve_universe(None, custom=["aapl", "  msft", "GOOG  ", ""])
    assert out == ["AAPL", "MSFT", "GOOG"]


def test_resolve_universe_preset_name():
    from modular.trading.universe import resolve_universe
    sectors = resolve_universe("sectors")
    assert "XLK" in sectors
    assert "XLF" in sectors


def test_sector_top_holdings_keyed_by_etf():
    from modular.trading.universe import SECTOR_TOP_HOLDINGS
    assert "XLK" in SECTOR_TOP_HOLDINGS
    assert "AAPL" in SECTOR_TOP_HOLDINGS["XLK"]
    assert all(len(v) >= 5 for v in SECTOR_TOP_HOLDINGS.values())


# ── Factor scoring (stub yfinance) ───────────────────────────────────────

@pytest.fixture
def stub_yfinance(monkeypatch):
    """Stub yfinance.Ticker to return synthetic data."""
    class FakeHist:
        def __init__(self, n=250, drift=0.001):
            import pandas as pd
            self._closes = [100 * (1 + drift) ** i for i in range(n)]
            self._df = pd.DataFrame({
                "Close": self._closes,
            })
            # Pandas index doesn't really matter for our computations
        def __len__(self):
            return len(self._df)
        def __getitem__(self, k):
            return self._df[k]
        @property
        def empty(self):
            return False
        def dropna(self):
            return self

    class FakeTicker:
        def __init__(self, sym, drift=0.001, info_overrides=None):
            self.symbol = sym
            self._drift = drift
            self._info_overrides = info_overrides or {}
        def history(self, period="1y", interval="1d", auto_adjust=False):
            return FakeHist(n=250, drift=self._drift)
        @property
        def info(self):
            base = {
                "trailingPE":      18.0,
                "returnOnEquity":  0.18,
                "debtToEquity":    50.0,
                "operatingMargins": 0.20,
                "marketCap":       500_000_000_000,
                "sector":          "Technology",
            }
            base.update(self._info_overrides)
            return base

    class FakeYF:
        def __init__(self):
            self.calls = 0
        def Ticker(self, sym):
            self.calls += 1
            # Pick drift based on symbol so different tickers score differently
            d = {"AAPL": 0.0015, "MSFT": 0.0010, "GOOG": 0.0008}.get(sym, 0.0005)
            return FakeTicker(sym, drift=d)

    fake = FakeYF()
    import sys
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    return fake


@_skip_if_no_pandas
def test_factor_scan_and_score(stub_yfinance, tmp_path, monkeypatch):
    from modular.trading import factors
    monkeypatch.setattr(factors, "_CACHE_PATH", tmp_path / "factors.json")

    rows = factors.scan_universe(["AAPL", "MSFT", "GOOG"], use_cache=False)
    assert len(rows) == 3
    assert all(r.error is None or r.error == "" for r in rows)

    factors.score(rows)
    composites = [r.composite_score for r in rows if r.composite_score is not None]
    assert len(composites) == 3

    # AAPL has highest drift, should rank highest on momentum
    by_sym = {r.symbol: r for r in rows}
    assert by_sym["AAPL"].momentum_score >= by_sym["GOOG"].momentum_score


@_skip_if_no_pandas
def test_factor_render_table(stub_yfinance, tmp_path, monkeypatch):
    from modular.trading import factors
    monkeypatch.setattr(factors, "_CACHE_PATH", tmp_path / "factors.json")
    rows = factors.scan_universe(["AAPL", "MSFT"], use_cache=False)
    factors.score(rows)
    md = factors.render_factor_table(rows)
    assert "Factor Scores" in md
    assert "AAPL" in md or "MSFT" in md


# ── Discovery: insider cluster ───────────────────────────────────────────

def test_insider_cluster_flags_clusters(monkeypatch):
    from modular.trading.discover import insider_cluster
    from modular.trading.alt_data import insider as ins_mod

    def fake_filings(sym, days=30, max_filings=20):
        # Return ≥3 filings only for "TSLA"
        if sym == "TSLA":
            return [
                {"accession": f"acc-{i}", "filed_date": "2026-04-15",
                 "form": "4", "primary_doc_url": f"https://sec.gov/{sym}/{i}"}
                for i in range(5)
            ]
        if sym == "AAPL":
            return [{"accession": "x", "filed_date": "2026-04-15",
                     "form": "4", "primary_doc_url": "https://sec.gov/AAPL/x"}]
        return []

    monkeypatch.setattr(ins_mod, "fetch_recent_insider_filings", fake_filings)

    hits = insider_cluster.scan(symbols=["TSLA", "AAPL", "GOOG"],
                                min_cluster_size=3, max_workers=1)
    assert len(hits) == 1
    assert hits[0].symbol == "TSLA"
    assert "5 Form 4" in hits[0].reason or "5" in hits[0].reason
    assert hits[0].source == "insider"


# ── Discovery: momentum-quality ──────────────────────────────────────────

@_skip_if_no_pandas
def test_momentum_quality_filters_below_threshold(stub_yfinance, tmp_path, monkeypatch):
    from modular.trading import factors
    from modular.trading.discover import momentum_quality
    monkeypatch.setattr(factors, "_CACHE_PATH", tmp_path / "factors.json")

    hits = momentum_quality.scan(symbols=["AAPL", "MSFT", "GOOG"],
                                 min_momentum=0.0, min_quality=0.0)
    assert len(hits) >= 1
    assert all(h.source == "momentum-quality" for h in hits)


# ── Discovery: sector rotation ───────────────────────────────────────────

def test_sector_rotation_picks_top_sectors(monkeypatch):
    from modular.trading.discover import sector_rotation

    # Stub fetch_market_data to return synthetic strong-up ETF data only for XLK
    def fake_data(sym, **kw):
        n = 80
        if sym == "XLK":
            closes = [100 * (1 + 0.002) ** i for i in range(n)]
        elif sym == "XLF":
            closes = [100 * (1 + 0.0015) ** i for i in range(n)]
        else:
            # All other ETFs flat or down
            closes = [100 * (1 - 0.0005) ** i for i in range(n)]
        return {
            "data": [{"date": f"2025-{1+i//30:02d}-{1+i%30:02d}", "close": c,
                      "open": c, "high": c, "low": c, "volume": 1_000_000}
                     for i, c in enumerate(closes)],
            "error": None,
        }
    monkeypatch.setattr(
        "modular.trading.data.fetchers.fetch_market_data", fake_data,
    )

    hits = sector_rotation.scan(top_sectors=2, top_per_sector=3)
    # Should surface XLK + XLF top holdings (which are AAPL/MSFT/NVDA + JPM/V/MA)
    assert len(hits) > 0
    syms = [h.symbol for h in hits]
    assert any(s in syms for s in ["AAPL", "MSFT", "NVDA", "JPM", "V"])


# ── Discovery orchestrator ───────────────────────────────────────────────

def test_orchestrator_merges_multi_source_hits(monkeypatch):
    from modular.trading.discover import orchestrator
    from modular.trading.discover.types import Discovery

    # Stub each scanner to return a known set
    def fake_insider(**kw):
        return [Discovery("AAPL", "insider", 0.8, "insider cluster", {})]
    def fake_earnings(**kw):
        return [Discovery("AAPL", "earnings", 0.7, "beat by 15%", {})]
    def fake_mq(**kw):
        return [Discovery("MSFT", "momentum-quality", 0.6, "factor combo", {})]
    def fake_sector(**kw):
        return []

    monkeypatch.setitem(orchestrator.SCANNERS, "insider", fake_insider)
    monkeypatch.setitem(orchestrator.SCANNERS, "earnings", fake_earnings)
    monkeypatch.setitem(orchestrator.SCANNERS, "momentum-quality", fake_mq)
    monkeypatch.setitem(orchestrator.SCANNERS, "sector", fake_sector)

    result = orchestrator.run(sources=["insider", "earnings",
                                        "momentum-quality", "sector"])
    # AAPL flagged by 2 sources → bonus, MSFT by 1 → no bonus
    by_sym = {e["symbol"]: e for e in result["ranked"]}
    assert "AAPL" in by_sym and "MSFT" in by_sym
    assert by_sym["AAPL"]["n_sources"] == 2
    assert by_sym["AAPL"]["bonus"] == 0.5
    assert by_sym["MSFT"]["bonus"] == 0.0
    # AAPL should rank above MSFT
    assert result["ranked"][0]["symbol"] == "AAPL"


def test_orchestrator_render_report_handles_empty():
    from modular.trading.discover import orchestrator
    md = orchestrator.render_report({
        "ranked": [], "per_source": {}, "n_unique": 0,
        "n_total_hits": 0, "notes": ["test note"],
    })
    assert "test note" in md


# ── Anomaly detector ────────────────────────────────────────────────────

def test_anomaly_volume_spike_detected(monkeypatch):
    from modular.trading.discover import anomaly
    # Synthetic 100-bar history: today's volume = 5× median
    rows = []
    for i in range(99):
        rows.append({"date": f"d{i}", "open": 100, "high": 101, "low": 99,
                     "close": 100, "volume": 1_000_000})
    rows.append({"date": "today", "open": 100, "high": 101, "low": 99,
                 "close": 100, "volume": 5_000_000})

    monkeypatch.setattr(
        "modular.trading.data.fetchers.fetch_market_data",
        lambda sym, **kw: {"data": rows, "error": None},
    )
    hits = anomaly.scan(["NVDA"], max_workers=1)
    types = [h.details.get("type") for h in hits]
    assert "volume_spike" in types


def test_anomaly_price_gap_detected(monkeypatch):
    from modular.trading.discover import anomaly
    rows = [{"date": f"d{i}", "open": 100, "high": 101, "low": 99,
             "close": 100, "volume": 1_000_000} for i in range(99)]
    # 5% gap up at open
    rows.append({"date": "today", "open": 105, "high": 106, "low": 104,
                 "close": 105, "volume": 1_000_000})

    monkeypatch.setattr(
        "modular.trading.data.fetchers.fetch_market_data",
        lambda sym, **kw: {"data": rows, "error": None},
    )
    hits = anomaly.scan(["AMD"], max_workers=1)
    types = [h.details.get("type") for h in hits]
    assert "price_gap" in types


def test_anomaly_returns_empty_for_short_history(monkeypatch):
    from modular.trading.discover import anomaly
    short_rows = [{"date": f"d{i}", "open": 100, "high": 100, "low": 100,
                   "close": 100, "volume": 1_000_000} for i in range(20)]
    monkeypatch.setattr(
        "modular.trading.data.fetchers.fetch_market_data",
        lambda sym, **kw: {"data": short_rows, "error": None},
    )
    hits = anomaly.scan(["X"], max_workers=1)
    assert hits == []


# ── Ranker ─────────────────────────────────────────────────────────────

@_skip_if_no_pandas
def test_ranker_combines_factor_and_discovery(stub_yfinance, tmp_path, monkeypatch):
    from modular.trading import ranker, factors
    from modular.trading.discover import orchestrator

    monkeypatch.setattr(factors, "_CACHE_PATH", tmp_path / "factors.json")
    # Stub orchestrator to avoid running real discovery
    monkeypatch.setattr(
        orchestrator, "run",
        lambda **kw: {"ranked": [], "per_source": {}, "n_unique": 0,
                      "n_total_hits": 0, "notes": []},
    )

    rows = ranker.rank(symbols=["AAPL", "MSFT"], universe=None,
                       use_discovery=True, use_calibration=False)
    assert len(rows) == 2
    assert rows[0].aggregate_score >= rows[1].aggregate_score
    md = ranker.render_rank_report(rows)
    assert "Investment Ranking" in md


def test_ranker_handles_empty_universe(monkeypatch):
    from modular.trading import ranker, factors
    monkeypatch.setattr(factors, "scan_universe", lambda syms, **k: [])
    out = ranker.rank(symbols=[], use_discovery=False, use_calibration=False)
    assert out == []


# ── Monitor ──────────────────────────────────────────────────────────────

def test_monitor_alert_render():
    from modular.trading.monitor import Alert, render_alerts
    alerts = [
        Alert("critical", "NVDA", "STOP HIT", "Trade #5 at -8%", "stop"),
        Alert("warning", "AMD", "Vol spike", "5× 90d median", "anomaly"),
        Alert("info", "AAPL", "Earnings in 5 days", "Plan accordingly", "earnings"),
    ]
    md = render_alerts(alerts)
    assert "Critical" in md and "NVDA" in md
    assert "STOP HIT" in md


def test_monitor_render_alerts_empty():
    from modular.trading.monitor import render_alerts
    md = render_alerts([])
    assert "quiet" in md.lower()


def test_monitor_dispatch_no_alerts_skips():
    from modular.trading.monitor import dispatch_to_bridges
    r = dispatch_to_bridges([])
    assert r["sent"] == 0


def test_monitor_scan_with_no_data_returns_empty(monkeypatch, tmp_path):
    """End-to-end: no watchlist, no open trades → empty alert list."""
    from modular.trading import monitor
    # Point state DB to tmp
    monkeypatch.setattr(monitor, "_STATE_DB", tmp_path / "monitor.db")
    # Stub paper_trader to return no trades / no watchlist
    import modular.trading.paper_trader as pt
    monkeypatch.setattr(pt, "list_trades", lambda **kw: [])
    monkeypatch.setattr(pt, "watchlist_list", lambda **kw: [])

    alerts = monitor.scan()
    assert alerts == []
    # Run was recorded
    last = monitor.last_run()
    assert last is not None
    assert last["n_symbols"] == 0


def test_monitor_anomaly_detection_when_watchlist_set(monkeypatch, tmp_path):
    """Smoke: with synthetic anomaly data, scan produces an alert."""
    from modular.trading import monitor
    monkeypatch.setattr(monitor, "_STATE_DB", tmp_path / "monitor.db")

    import modular.trading.paper_trader as pt
    monkeypatch.setattr(pt, "list_trades", lambda **kw: [])
    monkeypatch.setattr(pt, "watchlist_list",
                        lambda **kw: [{"symbol": "NVDA", "added_at": "x", "note": ""}])

    # Force volume-spike via fetcher stub
    rows = [{"date": f"d{i}", "open": 100, "high": 101, "low": 99,
             "close": 100, "volume": 1_000_000} for i in range(99)]
    rows.append({"date": "today", "open": 100, "high": 101, "low": 99,
                 "close": 100, "volume": 5_000_000})
    monkeypatch.setattr(
        "modular.trading.data.fetchers.fetch_market_data",
        lambda sym, **kw: {"data": rows, "error": None},
    )
    monkeypatch.setattr(
        "modular.trading.data.fetchers.fetch_current_price",
        lambda sym: {"price": 100.0},
    )

    alerts = monitor.scan(skip_insider=True)
    titles = [a.title for a in alerts]
    assert any("Volume spike" in t for t in titles)
