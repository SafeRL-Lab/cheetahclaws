"""
monitor.py — periodic market-monitoring scan + alert dispatcher.

What it does in one `scan()` call:
  1. Anomaly detection on watchlist (volume spike / gap / vol z-score)
  2. Stop-loss + take-profit checks on managed-portfolio holdings
  3. Earnings-calendar check (upcoming earnings within blackout window)
  4. New insider activity since last scan (delta detection)

Every alert is structured (`Alert` dataclass) and has a severity.
Optional dispatch to Telegram / WeChat / Slack via cheetahclaws's
existing bridges.

This module is **not** a continuously-running daemon. It runs one scan
when called. To run periodically:
  - manually: invoke `/trading monitor scan` whenever you want
  - cron-style: pair with cheetahclaws's `/monitor` system or external cron
  - bridge-driven: any of the bridges can be configured to call it

Frequency note: yfinance is 15-20 min delayed for free tier, so running
more often than every 5-10 minutes is wasted effort.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import paper_trader, earnings as earnings_mod
from .data import fetchers
from .discover import anomaly as anomaly_mod


_STATE_DB = Path.home() / ".cheetahclaws" / "trading" / "monitor_state.db"


@dataclass
class Alert:
    severity: str        # "info" | "warning" | "critical"
    symbol:   str
    title:    str
    detail:   str
    source:   str        # "anomaly" | "stop" | "tp" | "earnings" | "insider"
    payload:  dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(self.severity, "•")
        return f"{emoji} [{self.symbol}] {self.title}\n   {self.detail}"


@contextmanager
def _state_conn() -> Iterator[sqlite3.Connection]:
    _STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_STATE_DB))
    c.row_factory = sqlite3.Row
    try:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS monitor_seen_filings (
                symbol      TEXT NOT NULL,
                accession   TEXT NOT NULL,
                seen_at     TEXT NOT NULL,
                PRIMARY KEY (symbol, accession)
            );
            CREATE TABLE IF NOT EXISTS monitor_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at  TEXT NOT NULL,
                duration_s  REAL,
                n_symbols   INTEGER,
                n_alerts    INTEGER
            );
        """)
        yield c
        c.commit()
    finally:
        c.close()


# ── Individual checks ────────────────────────────────────────────────────

def _check_anomalies(symbols: list[str]) -> list[Alert]:
    hits = anomaly_mod.scan(symbols)
    alerts: list[Alert] = []
    for h in hits:
        sev = "warning" if h.score >= 0.5 else "info"
        alerts.append(Alert(
            severity=sev, symbol=h.symbol,
            title=h.reason, detail=str(h.details),
            source="anomaly", payload=h.details,
        ))
    return alerts


def _check_stops_and_tps() -> list[Alert]:
    """Stop-loss / take-profit check on open paper trades."""
    alerts: list[Alert] = []
    open_trades = paper_trader.list_trades(status="open", limit=500)
    for t in open_trades:
        if not t.entry_price or t.entry_price <= 0:
            continue
        info = fetchers.fetch_current_price(t.symbol)
        cur = info.get("price") if isinstance(info, dict) else None
        if not cur:
            continue
        # For long-style signals: BUY/OVERWEIGHT
        if t.signal in ("BUY", "OVERWEIGHT"):
            change_pct = (cur - t.entry_price) / t.entry_price * 100.0
            if t.stop_loss_pct is not None and change_pct <= -abs(t.stop_loss_pct):
                alerts.append(Alert(
                    severity="critical", symbol=t.symbol,
                    title=f"STOP HIT (-{t.stop_loss_pct}%)",
                    detail=f"Entry ${t.entry_price:.2f} → ${cur:.2f} ({change_pct:+.2f}%). Trade #{t.id}.",
                    source="stop",
                    payload={"trade_id": t.id, "entry": t.entry_price,
                             "current": cur, "change_pct": change_pct},
                ))
            elif t.take_profit_pct is not None and change_pct >= abs(t.take_profit_pct):
                alerts.append(Alert(
                    severity="warning", symbol=t.symbol,
                    title=f"TAKE-PROFIT HIT (+{t.take_profit_pct}%)",
                    detail=f"Entry ${t.entry_price:.2f} → ${cur:.2f} ({change_pct:+.2f}%). Trade #{t.id}.",
                    source="tp",
                    payload={"trade_id": t.id, "entry": t.entry_price,
                             "current": cur, "change_pct": change_pct},
                ))
    return alerts


def _check_earnings_blackouts() -> list[Alert]:
    """Earnings within 3 days for any open paper position."""
    alerts: list[Alert] = []
    open_trades = paper_trader.list_trades(status="open", limit=500)
    seen_symbols = set()
    for t in open_trades:
        if t.symbol in seen_symbols:
            continue
        seen_symbols.add(t.symbol)
        e = earnings_mod.upcoming_earnings(t.symbol)
        days = e.get("days_until")
        if days is None or days > 3:
            continue
        alerts.append(Alert(
            severity="critical" if days <= 1 else "warning",
            symbol=t.symbol,
            title=f"EARNINGS in {days} days ({e.get('date', 'TBD')})",
            detail=(f"Open paper position #{t.id} ({t.signal}, "
                    f"size {t.position_size_pct or '—'}%). "
                    f"Decide: hold through, trim, or exit pre-print."),
            source="earnings", payload=e,
        ))
    return alerts


def _check_insider_diff(symbols: list[str], days: int = 7) -> list[Alert]:
    """Surface NEW Form 4 filings since last scan (delta detection)."""
    alerts: list[Alert] = []
    from .alt_data import insider as ins
    with _state_conn() as c:
        seen_keys: set[tuple[str, str]] = {
            (r["symbol"], r["accession"])
            for r in c.execute("SELECT symbol, accession FROM monitor_seen_filings").fetchall()
        }

        new_records: list[tuple[str, str, str]] = []
        for sym in symbols:
            try:
                filings = ins.fetch_recent_insider_filings(sym, days=days, max_filings=10)
            except Exception:
                continue
            time.sleep(0.12)
            for f in filings:
                key = (sym, f["accession"])
                if key in seen_keys:
                    continue
                new_records.append((sym, f["accession"],
                                    datetime.now(timezone.utc).isoformat(timespec="seconds")))
                # Single new filing per ticker is normal; cluster of >=3 means alert.
            new_per_sym = sum(1 for r in new_records if r[0] == sym)
            if new_per_sym >= 3:
                alerts.append(Alert(
                    severity="warning", symbol=sym,
                    title=f"Insider activity cluster ({new_per_sym} new Form 4 filings)",
                    detail=f"{new_per_sym} new filings in last {days}d. Verify direction at SEC.",
                    source="insider",
                    payload={"new_filings": new_per_sym, "days": days},
                ))

        # Persist newly-seen filings
        if new_records:
            c.executemany(
                "INSERT OR IGNORE INTO monitor_seen_filings (symbol, accession, seen_at) "
                "VALUES (?, ?, ?)",
                new_records,
            )
    return alerts


# ── Public scan API ─────────────────────────────────────────────────────

def scan(
    extra_symbols: list[str] | None = None,
    skip_insider: bool = False,
) -> list[Alert]:
    """Run one full monitoring cycle. Returns flat list of alerts.

    Universe = paper-trade open positions + watchlist + extra_symbols.
    """
    started = time.time()
    syms: set[str] = set()
    syms.update(t.symbol for t in paper_trader.list_trades(status="open", limit=500))
    syms.update(w["symbol"] for w in paper_trader.watchlist_list())
    if extra_symbols:
        syms.update(s.upper().strip() for s in extra_symbols if s.strip())

    if not syms:
        with _state_conn() as c:
            c.execute(
                "INSERT INTO monitor_runs (started_at, duration_s, n_symbols, n_alerts) "
                "VALUES (?, ?, 0, 0)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 time.time() - started),
            )
        return []

    sym_list = sorted(syms)
    alerts: list[Alert] = []
    alerts.extend(_check_anomalies(sym_list))
    alerts.extend(_check_stops_and_tps())
    alerts.extend(_check_earnings_blackouts())
    if not skip_insider:
        alerts.extend(_check_insider_diff(sym_list))

    with _state_conn() as c:
        c.execute(
            "INSERT INTO monitor_runs (started_at, duration_s, n_symbols, n_alerts) "
            "VALUES (?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),
             round(time.time() - started, 2),
             len(sym_list), len(alerts)),
        )
    return alerts


def render_alerts(alerts: list[Alert]) -> str:
    if not alerts:
        return "_No alerts. All quiet on the watched front._"

    lines = [f"# Market Monitor — {len(alerts)} alert(s)", ""]
    by_sev: dict[str, list[Alert]] = {"critical": [], "warning": [], "info": []}
    for a in alerts:
        by_sev.setdefault(a.severity, []).append(a)
    for sev in ("critical", "warning", "info"):
        items = by_sev.get(sev, [])
        if not items:
            continue
        emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}[sev]
        lines.append(f"## {emoji} {sev.title()} ({len(items)})")
        for a in items:
            lines.append(f"- **{a.symbol}** [{a.source}] — {a.title}")
            lines.append(f"  - {a.detail}")
        lines.append("")
    return "\n".join(lines)


def dispatch_to_bridges(alerts: list[Alert],
                        bridges: list[str] | None = None,
                        config: dict | None = None) -> dict[str, Any]:
    """Push alerts to configured bridges (Telegram / WeChat / Slack).

    Soft-fails if a bridge isn't configured; returns per-bridge status.
    """
    if not alerts:
        return {"sent": 0, "channels": [], "skipped": "no alerts"}
    if bridges is None:
        bridges = ["telegram", "slack", "wechat"]

    config = config or {}
    body = render_alerts(alerts)
    out = {"sent": 0, "channels": [], "errors": {}}

    for ch in bridges:
        try:
            if ch == "telegram":
                from cheetahclaws.bridges import telegram as tg
                token = config.get("telegram_token")
                chat = config.get("telegram_chat_id")
                if token and chat:
                    tg._tg_send(token, chat, body)
                    out["sent"] += 1
                    out["channels"].append("telegram")
            elif ch == "slack":
                from cheetahclaws.bridges import slack as sl
                token = config.get("slack_token")
                ch_id = config.get("slack_channel")
                if token and ch_id:
                    sl._slack_send(token, ch_id, body)
                    out["sent"] += 1
                    out["channels"].append("slack")
            elif ch == "wechat":
                from cheetahclaws.bridges import wechat as wc
                if config.get("wechat_token"):
                    # Push to filehelper as a self-message
                    if hasattr(wc, "_wx_send"):
                        wc._wx_send(config, "filehelper", body)
                        out["sent"] += 1
                        out["channels"].append("wechat")
        except Exception as e:
            out.setdefault("errors", {})[ch] = f"{type(e).__name__}: {e}"

    return out


def last_run() -> dict | None:
    """Return diagnostic info about the most recent monitor run."""
    with _state_conn() as c:
        row = c.execute(
            "SELECT * FROM monitor_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return dict(row)
