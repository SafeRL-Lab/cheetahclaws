"""
paper_trader.py — SQLite-backed paper trading store for /trading.

Why this exists: without persistent tracking of agent recommendations, the
multi-agent analysis pipeline is unfalsifiable — every run is independent,
nobody ever measures whether "BUY High Confidence" actually beat "HOLD Low".
This module records every recommendation, periodically refreshes its market
price, and exposes the data to calibration.py and verifier.py.

Schema:
  paper_trades              — one row per recommendation
  paper_trade_snapshots     — periodic price snapshots for open trades
  trading_watchlist         — symbols the user wants /trading scan to cover

All paths and DB are scoped to ~/.cheetahclaws/trading/paper_trades.db.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


_DB_PATH = Path.home() / ".cheetahclaws" / "trading" / "paper_trades.db"
_DB_LOCK = threading.Lock()


# ── Schema ─────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT    NOT NULL,
    symbol              TEXT    NOT NULL,
    market              TEXT,
    signal              TEXT    NOT NULL,
    confidence          TEXT    NOT NULL,
    entry_price         REAL,
    position_size_pct   REAL,
    stop_loss_pct       REAL,
    take_profit_pct     REAL,
    time_horizon        TEXT,
    thesis              TEXT,
    sector              TEXT,
    source_run_id       TEXT,
    status              TEXT    NOT NULL DEFAULT 'open',
    closed_at           TEXT,
    close_price         REAL,
    realized_return_pct REAL,
    close_reason        TEXT
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status);
CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol ON paper_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_paper_trades_created_at ON paper_trades(created_at);

CREATE TABLE IF NOT EXISTS paper_trade_snapshots (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id                INTEGER NOT NULL,
    snapshot_at             TEXT    NOT NULL,
    price                   REAL    NOT NULL,
    unrealized_return_pct   REAL,
    FOREIGN KEY (trade_id) REFERENCES paper_trades(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_snapshots_trade_id ON paper_trade_snapshots(trade_id);

CREATE TABLE IF NOT EXISTS trading_watchlist (
    symbol      TEXT PRIMARY KEY,
    added_at    TEXT NOT NULL,
    note        TEXT
);
"""


# ── Connection management ──────────────────────────────────────────────────

@contextmanager
def _conn(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path) if db_path is not None else _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with _DB_LOCK:
        c = sqlite3.connect(str(path))
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        try:
            yield c
            c.commit()
        finally:
            c.close()


def init_db(db_path: Path | None = None) -> None:
    """Create tables if they don't exist. Safe to call repeatedly."""
    with _conn(db_path) as c:
        c.executescript(_SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Record types ───────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    id: int | None
    created_at: str
    symbol: str
    market: str | None
    signal: str
    confidence: str
    entry_price: float | None
    position_size_pct: float | None
    stop_loss_pct: float | None
    take_profit_pct: float | None
    time_horizon: str | None
    thesis: str | None
    sector: str | None
    source_run_id: str | None
    status: str
    closed_at: str | None
    close_price: float | None
    realized_return_pct: float | None
    close_reason: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TradeRecord":
        return cls(**{k: row[k] for k in row.keys()})


# ── Open / record / close ──────────────────────────────────────────────────

VALID_SIGNALS = {"BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"}
VALID_CONFIDENCE = {"High", "Medium", "Low"}


def open_trade(
    symbol: str,
    signal: str,
    confidence: str,
    entry_price: float | None = None,
    position_size_pct: float | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    time_horizon: str | None = None,
    thesis: str | None = None,
    sector: str | None = None,
    market: str | None = None,
    source_run_id: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Record a new paper trade. Returns the trade id."""
    if signal not in VALID_SIGNALS:
        raise ValueError(f"Invalid signal: {signal!r}. Must be one of {VALID_SIGNALS}")
    if confidence not in VALID_CONFIDENCE:
        raise ValueError(f"Invalid confidence: {confidence!r}. Must be one of {VALID_CONFIDENCE}")
    init_db(db_path)
    with _conn(db_path) as c:
        cur = c.execute(
            """
            INSERT INTO paper_trades (
                created_at, symbol, market, signal, confidence,
                entry_price, position_size_pct, stop_loss_pct, take_profit_pct,
                time_horizon, thesis, sector, source_run_id, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """,
            (_now(), symbol.upper(), market, signal, confidence,
             entry_price, position_size_pct, stop_loss_pct, take_profit_pct,
             time_horizon, thesis, sector, source_run_id),
        )
        return cur.lastrowid


def close_trade(
    trade_id: int,
    close_price: float,
    close_reason: str = "manual",
    db_path: Path | None = None,
) -> TradeRecord | None:
    """Close an open trade and compute realized return."""
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT * FROM paper_trades WHERE id = ? AND status = 'open'",
            (trade_id,),
        ).fetchone()
        if row is None:
            return None
        entry = row["entry_price"]
        realized = None
        if entry and entry > 0:
            realized = (close_price - entry) / entry * 100.0
            # SELL/UNDERWEIGHT signals profit when price drops — invert.
            if row["signal"] in ("SELL", "UNDERWEIGHT"):
                realized = -realized
        c.execute(
            """
            UPDATE paper_trades
               SET status = 'closed',
                   closed_at = ?,
                   close_price = ?,
                   realized_return_pct = ?,
                   close_reason = ?
             WHERE id = ?
            """,
            (_now(), close_price, realized, close_reason, trade_id),
        )
        return TradeRecord.from_row(
            c.execute("SELECT * FROM paper_trades WHERE id = ?", (trade_id,)).fetchone()
        )


def add_snapshot(
    trade_id: int,
    price: float,
    db_path: Path | None = None,
) -> float | None:
    """Record a price snapshot for an open trade. Returns unrealized return %."""
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT entry_price, signal FROM paper_trades WHERE id = ? AND status = 'open'",
            (trade_id,),
        ).fetchone()
        if row is None:
            return None
        unreal = None
        if row["entry_price"] and row["entry_price"] > 0:
            unreal = (price - row["entry_price"]) / row["entry_price"] * 100.0
            if row["signal"] in ("SELL", "UNDERWEIGHT"):
                unreal = -unreal
        c.execute(
            """
            INSERT INTO paper_trade_snapshots (trade_id, snapshot_at, price, unrealized_return_pct)
            VALUES (?, ?, ?, ?)
            """,
            (trade_id, _now(), price, unreal),
        )
        return unreal


# ── Query ─────────────────────────────────────────────────────────────────

def list_trades(
    status: str | None = None,
    symbol: str | None = None,
    limit: int = 100,
    db_path: Path | None = None,
) -> list[TradeRecord]:
    init_db(db_path)
    sql = "SELECT * FROM paper_trades WHERE 1=1"
    params: list[Any] = []
    if status:
        sql += " AND status = ?"
        params.append(status)
    if symbol:
        sql += " AND symbol = ?"
        params.append(symbol.upper())
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with _conn(db_path) as c:
        return [TradeRecord.from_row(r) for r in c.execute(sql, params).fetchall()]


def get_trade(trade_id: int, db_path: Path | None = None) -> TradeRecord | None:
    init_db(db_path)
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM paper_trades WHERE id = ?", (trade_id,)).fetchone()
        return TradeRecord.from_row(row) if row else None


def open_position_summary(db_path: Path | None = None) -> dict[str, Any]:
    """Aggregate stats for open positions: count, exposure %, by-sector breakdown."""
    init_db(db_path)
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT symbol, sector, position_size_pct FROM paper_trades WHERE status = 'open'"
        ).fetchall()
    n = len(rows)
    total_pct = sum((r["position_size_pct"] or 0.0) for r in rows)
    by_sector: dict[str, float] = {}
    for r in rows:
        sec = r["sector"] or "Unknown"
        by_sector[sec] = by_sector.get(sec, 0.0) + (r["position_size_pct"] or 0.0)
    return {
        "open_count": n,
        "total_exposure_pct": total_pct,
        "by_sector_pct": by_sector,
        "symbols": [r["symbol"] for r in rows],
    }


# ── Watchlist ──────────────────────────────────────────────────────────────

def watchlist_add(symbol: str, note: str | None = None, db_path: Path | None = None) -> None:
    init_db(db_path)
    with _conn(db_path) as c:
        c.execute(
            "INSERT OR REPLACE INTO trading_watchlist (symbol, added_at, note) VALUES (?, ?, ?)",
            (symbol.upper(), _now(), note),
        )


def watchlist_remove(symbol: str, db_path: Path | None = None) -> bool:
    init_db(db_path)
    with _conn(db_path) as c:
        cur = c.execute(
            "DELETE FROM trading_watchlist WHERE symbol = ?", (symbol.upper(),)
        )
        return cur.rowcount > 0


def watchlist_list(db_path: Path | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT symbol, added_at, note FROM trading_watchlist ORDER BY added_at"
        ).fetchall()
        return [dict(r) for r in rows]


# ── Auto-record from analyze output ────────────────────────────────────────

def record_from_analysis(
    symbol: str,
    analysis_text: str,
    db_path: Path | None = None,
) -> int | None:
    """Parse the multi-agent analysis output and open a paper trade.

    Looks for the Phase 5 'RATING:' / 'Plan:' / 'Conviction:' block and
    extracts the structured fields. Returns trade id, or None if the
    text doesn't contain a recognizable rating.
    """
    parsed = _parse_phase5(analysis_text)
    if not parsed:
        return None
    try:
        return open_trade(
            symbol=symbol,
            signal=parsed["signal"],
            confidence=parsed["confidence"],
            entry_price=parsed.get("entry_price"),
            position_size_pct=parsed.get("position_size_pct"),
            stop_loss_pct=parsed.get("stop_loss_pct"),
            take_profit_pct=parsed.get("take_profit_pct"),
            time_horizon=parsed.get("time_horizon"),
            thesis=parsed.get("thesis"),
            sector=parsed.get("sector"),
            db_path=db_path,
        )
    except ValueError:
        return None


def _parse_phase5(text: str) -> dict[str, Any] | None:
    """Extract structured fields from the Phase 5 analysis block.

    Permissive — handles models that wander on formatting. Returns None
    if RATING is missing, since signal is the one mandatory field.
    """
    import re

    out: dict[str, Any] = {}

    # Permissive separator class — handles "RATING: BUY", "**RATING**: BUY",
    # "RATING — BUY", and other model-specific formatting noise.
    sep = r"[:\s\*\-—–]*?"

    rating_match = re.search(
        rf"RATING{sep}(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)\b",
        text, re.IGNORECASE,
    )
    if not rating_match:
        return None
    out["signal"] = rating_match.group(1).upper()

    conf_match = re.search(
        rf"Conviction{sep}(High|Medium|Low)\b",
        text, re.IGNORECASE,
    )
    out["confidence"] = (conf_match.group(1).capitalize() if conf_match else "Medium")

    pos_match = re.search(rf"Position\s*Size{sep}([0-9.]+)\s*%", text, re.IGNORECASE)
    if pos_match:
        out["position_size_pct"] = float(pos_match.group(1))

    stop_match = re.search(rf"Stop\s*Loss{sep}-?([0-9.]+)\s*%", text, re.IGNORECASE)
    if stop_match:
        out["stop_loss_pct"] = float(stop_match.group(1))

    tp_match = re.search(rf"Take\s*Profit{sep}\+?([0-9.]+)\s*%", text, re.IGNORECASE)
    if tp_match:
        out["take_profit_pct"] = float(tp_match.group(1))

    horizon_match = re.search(
        rf"Time\s*Horizon{sep}([^\n]+)", text, re.IGNORECASE,
    )
    if horizon_match:
        out["time_horizon"] = horizon_match.group(1).strip().rstrip("*").strip()[:50]

    thesis_match = re.search(
        rf"Summary{sep}([^\n]+(?:\n[^\n#]+){{0,2}})",
        text, re.IGNORECASE,
    )
    if thesis_match:
        out["thesis"] = thesis_match.group(1).strip()[:1000]

    return out
