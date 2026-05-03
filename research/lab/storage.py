"""research/lab/storage.py — SQLite persistence for lab runs.

Five additive tables in ``~/.cheetahclaws/research_lab.db`` (separate
file from the daemon's sessions.db so the F-2 daemon work won't
collide):

  lab_runs        — one row per run (topic, status, started_at, ...)
  lab_stages      — one row per stage transition (which stage, outcome,
                    timestamps, agent role attribution)
  lab_messages    — every agent utterance (role, content, ts) for the
                    debate log; reviewer critiques, author rebuttals,
                    PI decisions all live here
  lab_artifacts   — versioned artifacts (research questions, outline,
                    section drafts, citations table, final report)
  lab_budget      — token / USD spend per run

Schema is additive and idempotent.  No migrations needed for v0.

Threading: SQLite ``check_same_thread=False`` plus a connection lock —
the orchestrator runs on a worker thread, the slash command queries
status from the REPL thread, both touch the same DB.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

DEFAULT_DB_PATH = Path.home() / ".cheetahclaws" / "research_lab.db"
DEFAULT_OUTPUT_DIR = Path.home() / ".cheetahclaws" / "research_papers"

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS lab_runs (
        run_id        TEXT PRIMARY KEY,
        topic         TEXT NOT NULL,
        status        TEXT NOT NULL,        -- pending|running|paused|done|failed|aborted
        current_stage TEXT,
        budget_tokens INTEGER,                -- max tokens for this run
        budget_cost_cents INTEGER,            -- max cost in USD cents
        max_rounds    INTEGER NOT NULL DEFAULT 5,
        created_at    REAL NOT NULL,
        updated_at    REAL NOT NULL,
        completed_at  REAL,
        error         TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_lab_runs_status ON lab_runs(status)",
    """CREATE TABLE IF NOT EXISTS lab_stages (
        run_id     TEXT NOT NULL,
        stage      TEXT NOT NULL,
        round      INTEGER NOT NULL DEFAULT 0,
        started_at REAL NOT NULL,
        ended_at   REAL,
        outcome    TEXT,                      -- advance|revise|abort|pending
        notes      TEXT,
        PRIMARY KEY (run_id, stage, round)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_lab_stages_run ON lab_stages(run_id)",
    """CREATE TABLE IF NOT EXISTS lab_messages (
        run_id   TEXT NOT NULL,
        ts       REAL NOT NULL,
        stage    TEXT NOT NULL,
        round    INTEGER NOT NULL DEFAULT 0,
        role     TEXT NOT NULL,               -- pi|questioner|surveyor|designer|writer|reviewer_n|lay_reader|user
        kind     TEXT NOT NULL,               -- draft|critique|decision|note
        content  TEXT NOT NULL,
        meta     TEXT                         -- JSON sidecar (model, tokens, etc.)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_lab_messages_run ON lab_messages(run_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_lab_messages_stage ON lab_messages(run_id, stage)",
    """CREATE TABLE IF NOT EXISTS lab_artifacts (
        run_id   TEXT NOT NULL,
        kind     TEXT NOT NULL,               -- rq|outline|section_<name>|citations|report
        version  INTEGER NOT NULL DEFAULT 1,
        content  TEXT NOT NULL,
        ts       REAL NOT NULL,
        PRIMARY KEY (run_id, kind, version)
    )""",
    """CREATE TABLE IF NOT EXISTS lab_budget (
        run_id        TEXT PRIMARY KEY,
        tokens_used   INTEGER NOT NULL DEFAULT 0,
        cost_cents    INTEGER NOT NULL DEFAULT 0,
        started_at    REAL NOT NULL,
        last_updated  REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS lab_experiments (
        run_id        TEXT NOT NULL,
        attempt       INTEGER NOT NULL,
        started_at    REAL NOT NULL,
        ended_at      REAL,
        exit_code     INTEGER,
        duration_s    REAL,
        timed_out     INTEGER NOT NULL DEFAULT 0,
        code          TEXT NOT NULL,
        stdout        TEXT,
        stderr        TEXT,
        artifacts     TEXT,                  -- JSON list of relative paths
        notes         TEXT,
        PRIMARY KEY (run_id, attempt)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_lab_experiments_run ON lab_experiments(run_id)",
]


# ── Records ────────────────────────────────────────────────────────────────


@dataclass
class RunRecord:
    run_id: str
    topic: str
    status: str
    current_stage: Optional[str]
    budget_tokens: Optional[int]
    budget_cost_cents: Optional[int]
    max_rounds: int
    created_at: float
    updated_at: float
    completed_at: Optional[float] = None
    error: Optional[str] = None


@dataclass
class StageRecord:
    run_id: str
    stage: str
    round: int
    started_at: float
    ended_at: Optional[float]
    outcome: Optional[str]
    notes: Optional[str]


@dataclass
class MessageRecord:
    run_id: str
    ts: float
    stage: str
    round: int
    role: str
    kind: str
    content: str
    meta: Optional[dict] = None


@dataclass
class ArtifactRecord:
    run_id: str
    kind: str
    version: int
    content: str
    ts: float


@dataclass
class ExperimentRecord:
    run_id: str
    attempt: int
    started_at: float
    ended_at: Optional[float]
    exit_code: Optional[int]
    duration_s: Optional[float]
    timed_out: bool
    code: str
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    artifacts: list[str] = field(default_factory=list)
    notes: Optional[str] = None


# ── Storage class ──────────────────────────────────────────────────────────


class LabStorage:
    """Single-instance SQLite-backed storage for lab runs.

    Public methods are chunked into clear concerns: run lifecycle,
    stages, messages, artifacts, budget.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._txn() as cur:
            for stmt in _SCHEMA:
                cur.execute(stmt)

    @contextmanager
    def _txn(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ── Run lifecycle ─────────────────────────────────────────────────

    def create_run(self, *, topic: str,
                   budget_tokens: Optional[int] = None,
                   budget_cost_cents: Optional[int] = None,
                   max_rounds: int = 5) -> RunRecord:
        run_id = "lab_" + uuid.uuid4().hex[:12]
        now = time.time()
        with self._txn() as cur:
            cur.execute(
                """INSERT INTO lab_runs
                   (run_id, topic, status, current_stage,
                    budget_tokens, budget_cost_cents, max_rounds,
                    created_at, updated_at)
                   VALUES (?, ?, 'pending', NULL, ?, ?, ?, ?, ?)""",
                (run_id, topic, budget_tokens, budget_cost_cents,
                 max_rounds, now, now),
            )
            cur.execute(
                """INSERT INTO lab_budget
                   (run_id, tokens_used, cost_cents, started_at, last_updated)
                   VALUES (?, 0, 0, ?, ?)""",
                (run_id, now, now),
            )
        return self.get_run(run_id)  # type: ignore[return-value]

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        with self._txn() as cur:
            cur.execute("SELECT * FROM lab_runs WHERE run_id = ?", (run_id,))
            row = cur.fetchone()
        return _row_to_run(row) if row else None

    def list_runs(self, *, status: Optional[str] = None,
                  limit: int = 50) -> list[RunRecord]:
        with self._txn() as cur:
            if status:
                cur.execute(
                    "SELECT * FROM lab_runs WHERE status = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM lab_runs "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            return [_row_to_run(r) for r in cur.fetchall()]

    def update_run_status(self, run_id: str, status: str,
                          *, current_stage: Optional[str] = None,
                          error: Optional[str] = None) -> None:
        now = time.time()
        with self._txn() as cur:
            if status in ("done", "failed", "aborted"):
                cur.execute(
                    """UPDATE lab_runs SET status = ?, current_stage = ?,
                       updated_at = ?, completed_at = ?, error = ?
                       WHERE run_id = ?""",
                    (status, current_stage, now, now, error, run_id),
                )
            else:
                cur.execute(
                    """UPDATE lab_runs SET status = ?, current_stage = ?,
                       updated_at = ?, error = COALESCE(?, error)
                       WHERE run_id = ?""",
                    (status, current_stage, now, error, run_id),
                )

    # ── Stages ────────────────────────────────────────────────────────

    def start_stage(self, run_id: str, stage: str, round_: int = 0) -> None:
        now = time.time()
        with self._txn() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO lab_stages
                   (run_id, stage, round, started_at, ended_at, outcome, notes)
                   VALUES (?, ?, ?, ?, NULL, 'pending', NULL)""",
                (run_id, stage, round_, now),
            )

    def end_stage(self, run_id: str, stage: str, round_: int,
                  outcome: str, notes: Optional[str] = None) -> None:
        now = time.time()
        with self._txn() as cur:
            cur.execute(
                """UPDATE lab_stages
                   SET ended_at = ?, outcome = ?, notes = ?
                   WHERE run_id = ? AND stage = ? AND round = ?""",
                (now, outcome, notes, run_id, stage, round_),
            )

    def list_stages(self, run_id: str) -> list[StageRecord]:
        with self._txn() as cur:
            cur.execute(
                "SELECT * FROM lab_stages WHERE run_id = ? "
                "ORDER BY started_at ASC",
                (run_id,),
            )
            rows = cur.fetchall()
        return [_row_to_stage(r) for r in rows]

    # ── Messages ──────────────────────────────────────────────────────

    def append_message(self, run_id: str, *, stage: str, round_: int,
                       role: str, kind: str, content: str,
                       meta: Optional[dict] = None) -> None:
        with self._txn() as cur:
            cur.execute(
                """INSERT INTO lab_messages
                   (run_id, ts, stage, round, role, kind, content, meta)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, time.time(), stage, round_, role, kind, content,
                 json.dumps(meta) if meta else None),
            )

    def list_messages(self, run_id: str, *, stage: Optional[str] = None,
                      limit: int = 500) -> list[MessageRecord]:
        with self._txn() as cur:
            if stage:
                cur.execute(
                    "SELECT * FROM lab_messages WHERE run_id = ? AND stage = ? "
                    "ORDER BY ts ASC LIMIT ?",
                    (run_id, stage, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM lab_messages WHERE run_id = ? "
                    "ORDER BY ts ASC LIMIT ?",
                    (run_id, limit),
                )
            rows = cur.fetchall()
        return [_row_to_msg(r) for r in rows]

    # ── Artifacts ────────────────────────────────────────────────────

    def put_artifact(self, run_id: str, kind: str, content: str) -> int:
        """Append a new versioned artifact; returns the version number."""
        now = time.time()
        with self._txn() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM lab_artifacts "
                "WHERE run_id = ? AND kind = ?",
                (run_id, kind),
            )
            v = cur.fetchone()["v"] + 1
            cur.execute(
                """INSERT INTO lab_artifacts
                   (run_id, kind, version, content, ts)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_id, kind, v, content, now),
            )
        return v

    def get_latest_artifact(self, run_id: str, kind: str) -> Optional[ArtifactRecord]:
        with self._txn() as cur:
            cur.execute(
                """SELECT * FROM lab_artifacts
                   WHERE run_id = ? AND kind = ?
                   ORDER BY version DESC LIMIT 1""",
                (run_id, kind),
            )
            row = cur.fetchone()
        return _row_to_artifact(row) if row else None

    def list_artifacts(self, run_id: str) -> list[ArtifactRecord]:
        with self._txn() as cur:
            cur.execute(
                "SELECT * FROM lab_artifacts WHERE run_id = ? "
                "ORDER BY ts ASC",
                (run_id,),
            )
            return [_row_to_artifact(r) for r in cur.fetchall()]

    # ── Budget ───────────────────────────────────────────────────────

    def add_budget(self, run_id: str, *, tokens: int = 0, cost_cents: int = 0) -> None:
        now = time.time()
        with self._txn() as cur:
            cur.execute(
                """UPDATE lab_budget
                   SET tokens_used = tokens_used + ?,
                       cost_cents  = cost_cents + ?,
                       last_updated = ?
                   WHERE run_id = ?""",
                (tokens, cost_cents, now, run_id),
            )

    def get_budget(self, run_id: str) -> tuple[int, int]:
        """Return (tokens_used, cost_cents)."""
        with self._txn() as cur:
            cur.execute(
                "SELECT tokens_used, cost_cents FROM lab_budget "
                "WHERE run_id = ?",
                (run_id,),
            )
            row = cur.fetchone()
        if not row:
            return 0, 0
        return int(row["tokens_used"]), int(row["cost_cents"])

    # ── Experiments ──────────────────────────────────────────────────

    def record_experiment(self, *, run_id: str, attempt: int,
                          code: str,
                          exit_code: Optional[int] = None,
                          stdout: Optional[str] = None,
                          stderr: Optional[str] = None,
                          duration_s: Optional[float] = None,
                          timed_out: bool = False,
                          artifacts: Optional[list[str]] = None,
                          notes: Optional[str] = None) -> None:
        now = time.time()
        with self._txn() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO lab_experiments
                   (run_id, attempt, started_at, ended_at, exit_code,
                    duration_s, timed_out, code, stdout, stderr,
                    artifacts, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, attempt, now, now, exit_code,
                 duration_s, 1 if timed_out else 0,
                 code, stdout, stderr,
                 json.dumps(artifacts or []), notes),
            )

    def list_experiments(self, run_id: str) -> list[ExperimentRecord]:
        with self._txn() as cur:
            cur.execute(
                "SELECT * FROM lab_experiments WHERE run_id = ? "
                "ORDER BY attempt ASC",
                (run_id,),
            )
            rows = cur.fetchall()
        return [_row_to_experiment(r) for r in rows]

    def get_latest_experiment(self, run_id: str) -> Optional[ExperimentRecord]:
        with self._txn() as cur:
            cur.execute(
                "SELECT * FROM lab_experiments WHERE run_id = ? "
                "ORDER BY attempt DESC LIMIT 1",
                (run_id,),
            )
            row = cur.fetchone()
        return _row_to_experiment(row) if row else None


# ── Row converters ────────────────────────────────────────────────────────


def _row_to_run(r) -> RunRecord:
    return RunRecord(
        run_id=r["run_id"], topic=r["topic"], status=r["status"],
        current_stage=r["current_stage"],
        budget_tokens=r["budget_tokens"],
        budget_cost_cents=r["budget_cost_cents"],
        max_rounds=int(r["max_rounds"]),
        created_at=r["created_at"], updated_at=r["updated_at"],
        completed_at=r["completed_at"], error=r["error"],
    )


def _row_to_stage(r) -> StageRecord:
    return StageRecord(
        run_id=r["run_id"], stage=r["stage"], round=int(r["round"]),
        started_at=r["started_at"], ended_at=r["ended_at"],
        outcome=r["outcome"], notes=r["notes"],
    )


def _row_to_msg(r) -> MessageRecord:
    return MessageRecord(
        run_id=r["run_id"], ts=r["ts"], stage=r["stage"],
        round=int(r["round"]), role=r["role"],
        kind=r["kind"], content=r["content"],
        meta=json.loads(r["meta"]) if r["meta"] else None,
    )


def _row_to_artifact(r) -> ArtifactRecord:
    return ArtifactRecord(
        run_id=r["run_id"], kind=r["kind"], version=int(r["version"]),
        content=r["content"], ts=r["ts"],
    )


def _row_to_experiment(r) -> ExperimentRecord:
    arts = []
    if r["artifacts"]:
        try:
            arts = json.loads(r["artifacts"]) or []
        except Exception:
            arts = []
    return ExperimentRecord(
        run_id=r["run_id"], attempt=int(r["attempt"]),
        started_at=r["started_at"], ended_at=r["ended_at"],
        exit_code=r["exit_code"],
        duration_s=r["duration_s"],
        timed_out=bool(r["timed_out"]),
        code=r["code"], stdout=r["stdout"], stderr=r["stderr"],
        artifacts=arts, notes=r["notes"],
    )
