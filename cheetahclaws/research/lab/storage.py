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


def _slugify(topic: str, *, max_len: int = 60) -> str:
    """Topic → filesystem-safe ASCII slug.

    Lowercase, ASCII alphanum + '-' only, single-hyphen separators,
    truncated at ``max_len`` (preferring a hyphen boundary).  CJK or
    other non-ASCII characters are dropped — if that leaves the slug
    empty (e.g. a pure-Chinese topic) we fall back to "untitled" so
    the run_id short suffix still makes the directory unique.
    """
    import re as _re
    s = _re.sub(r"[^A-Za-z0-9]+", "-", topic).strip("-").lower()
    s = _re.sub(r"-+", "-", s)
    if len(s) > max_len:
        truncated = s[:max_len]
        # Prefer cutting at a hyphen so the slug ends on a word boundary.
        if "-" in truncated:
            truncated = truncated.rsplit("-", 1)[0]
        s = truncated
    return s or "untitled"


def human_dir_name(run_id: str, topic: str, created_at: float) -> str:
    """Compose a directory name like
    ``2026-05-07_18-15_post-transformer-architectures-survey_b16036de``.

    The trailing run_id-suffix guarantees uniqueness across two runs
    with identical topic + minute — without it, a backlog with the
    same topic queued twice would race on the same path.
    """
    import datetime as _dt
    dt = _dt.datetime.fromtimestamp(created_at).strftime("%Y-%m-%d_%H-%M")
    slug = _slugify(topic)
    # run_id format is "lab_<12 hex>"; keep last 8 hex for a compact
    # but still-unique suffix.
    short = run_id.replace("lab_", "")[:8]
    return f"{dt}_{slug}_{short}"


def output_dir_for(run_id: str, topic: str, created_at: float,
                   *, root: Optional[Path] = None) -> Path:
    """Resolve the absolute output directory for a run."""
    return (root or DEFAULT_OUTPUT_DIR) / human_dir_name(run_id, topic, created_at)

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
    # ── Phase A additions: meta-loop iterations + topic backlog ─────────
    """CREATE TABLE IF NOT EXISTS lab_iterations (
        run_id        TEXT NOT NULL,
        iter_n        INTEGER NOT NULL,         -- 1-based; iter 1 = first /lab iterate call
        target_score  REAL,                     -- score the iterate call was targeting
        score_avg     REAL,                     -- self-review avg, populated after scoring
        score_breakdown TEXT,                   -- JSON {dim → avg}
        revise_stage  TEXT,                     -- stage we rolled back to
        delta         REAL,                     -- score - prev_score
        started_at    REAL NOT NULL,
        ended_at      REAL,
        status        TEXT NOT NULL,            -- pending|scoring|reverting|running|done|failed|skipped
        notes         TEXT,
        PRIMARY KEY (run_id, iter_n)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_lab_iterations_run ON lab_iterations(run_id)",
    """CREATE TABLE IF NOT EXISTS lab_backlog (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        topic          TEXT NOT NULL,
        status         TEXT NOT NULL,           -- pending|running|done|failed|skipped
        run_id         TEXT,                    -- linked once started
        iterate        INTEGER NOT NULL DEFAULT 0,   -- 0 = single-shot, 1 = auto-iterate after finalize
        target_score   REAL,                    -- only meaningful when iterate=1
        max_iterations INTEGER NOT NULL DEFAULT 5,
        added_at       REAL NOT NULL,
        started_at     REAL,
        ended_at       REAL,
        priority       INTEGER NOT NULL DEFAULT 0,
        notes          TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_lab_backlog_status ON lab_backlog(status, priority DESC, added_at)",
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

    # ── Iterations (Phase A meta-loop) ────────────────────────────────

    def add_iteration(self, *, run_id: str, iter_n: int,
                      target_score: Optional[float] = None,
                      revise_stage: Optional[str] = None,
                      notes: Optional[str] = None) -> None:
        now = time.time()
        with self._txn() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO lab_iterations
                   (run_id, iter_n, target_score, revise_stage,
                    started_at, status, notes)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
                (run_id, iter_n, target_score, revise_stage, now, notes),
            )

    def update_iteration(self, *, run_id: str, iter_n: int,
                         status: Optional[str] = None,
                         score_avg: Optional[float] = None,
                         score_breakdown: Optional[dict] = None,
                         delta: Optional[float] = None,
                         revise_stage: Optional[str] = None,
                         notes: Optional[str] = None,
                         mark_done: bool = False) -> None:
        # Preserve existing fields when caller passes None for them.
        sets: list[str] = []
        vals: list = []
        if status is not None:
            sets.append("status = ?"); vals.append(status)
        if score_avg is not None:
            sets.append("score_avg = ?"); vals.append(score_avg)
        if score_breakdown is not None:
            sets.append("score_breakdown = ?")
            vals.append(json.dumps(score_breakdown))
        if delta is not None:
            sets.append("delta = ?"); vals.append(delta)
        if revise_stage is not None:
            sets.append("revise_stage = ?"); vals.append(revise_stage)
        if notes is not None:
            sets.append("notes = ?"); vals.append(notes)
        if mark_done:
            sets.append("ended_at = ?"); vals.append(time.time())
        if not sets:
            return
        vals.extend([run_id, iter_n])
        with self._txn() as cur:
            cur.execute(
                f"UPDATE lab_iterations SET {', '.join(sets)} "
                "WHERE run_id = ? AND iter_n = ?",
                vals,
            )

    def list_iterations(self, run_id: str) -> list[dict]:
        with self._txn() as cur:
            cur.execute(
                "SELECT * FROM lab_iterations WHERE run_id = ? "
                "ORDER BY iter_n ASC",
                (run_id,),
            )
            rows = cur.fetchall()
        out = []
        for r in rows:
            out.append({
                "run_id": r["run_id"],
                "iter_n": int(r["iter_n"]),
                "target_score": r["target_score"],
                "score_avg": r["score_avg"],
                "score_breakdown": json.loads(r["score_breakdown"])
                                    if r["score_breakdown"] else None,
                "revise_stage": r["revise_stage"],
                "delta": r["delta"],
                "started_at": r["started_at"],
                "ended_at": r["ended_at"],
                "status": r["status"],
                "notes": r["notes"],
            })
        return out

    def latest_iteration_n(self, run_id: str) -> int:
        """Return the highest iter_n recorded for ``run_id`` (0 if none)."""
        with self._txn() as cur:
            cur.execute(
                "SELECT MAX(iter_n) AS m FROM lab_iterations WHERE run_id = ?",
                (run_id,),
            )
            row = cur.fetchone()
        return int(row["m"] or 0)

    # ── Backlog (Phase A topic queue) ─────────────────────────────────

    def add_backlog(self, *, topic: str,
                    iterate: bool = False,
                    target_score: Optional[float] = None,
                    max_iterations: int = 5,
                    priority: int = 0,
                    notes: Optional[str] = None) -> int:
        now = time.time()
        with self._txn() as cur:
            cur.execute(
                """INSERT INTO lab_backlog
                   (topic, status, iterate, target_score, max_iterations,
                    added_at, priority, notes)
                   VALUES (?, 'pending', ?, ?, ?, ?, ?, ?)""",
                (topic, 1 if iterate else 0, target_score, max_iterations,
                 now, priority, notes),
            )
            return int(cur.lastrowid)

    def list_backlog(self, *, status: Optional[str] = None,
                     limit: int = 100) -> list[dict]:
        q = "SELECT * FROM lab_backlog"
        args: list = []
        if status:
            q += " WHERE status = ?"; args.append(status)
        q += " ORDER BY priority DESC, added_at ASC LIMIT ?"; args.append(limit)
        with self._txn() as cur:
            cur.execute(q, args)
            rows = cur.fetchall()
        return [_row_to_backlog(r) for r in rows]

    def claim_next_backlog(self) -> Optional[dict]:
        """Atomically pull the highest-priority pending item and mark it
        running. Returns None if the queue is empty."""
        now = time.time()
        with self._txn() as cur:
            cur.execute(
                "SELECT * FROM lab_backlog WHERE status = 'pending' "
                "ORDER BY priority DESC, added_at ASC LIMIT 1",
            )
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute(
                """UPDATE lab_backlog
                   SET status = 'running', started_at = ?
                   WHERE id = ?""",
                (now, row["id"]),
            )
        return _row_to_backlog(row)

    def update_backlog(self, *, item_id: int,
                       status: Optional[str] = None,
                       run_id: Optional[str] = None,
                       notes: Optional[str] = None,
                       mark_ended: bool = False) -> None:
        sets: list[str] = []
        vals: list = []
        if status is not None:
            sets.append("status = ?"); vals.append(status)
        if run_id is not None:
            sets.append("run_id = ?"); vals.append(run_id)
        if notes is not None:
            sets.append("notes = ?"); vals.append(notes)
        if mark_ended:
            sets.append("ended_at = ?"); vals.append(time.time())
        if not sets:
            return
        vals.append(item_id)
        with self._txn() as cur:
            cur.execute(
                f"UPDATE lab_backlog SET {', '.join(sets)} WHERE id = ?",
                vals,
            )

    def remove_backlog(self, item_id: int) -> bool:
        with self._txn() as cur:
            cur.execute("DELETE FROM lab_backlog WHERE id = ?", (item_id,))
            return cur.rowcount > 0

    def reset_running_backlog(self) -> int:
        """Return any stuck ``running`` items to ``pending`` (used at
        daemon startup so a crashed daemon doesn't permanently lose work)."""
        with self._txn() as cur:
            cur.execute(
                "UPDATE lab_backlog SET status = 'pending', started_at = NULL "
                "WHERE status = 'running'"
            )
            return cur.rowcount


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


def _row_to_backlog(r) -> dict:
    return {
        "id": int(r["id"]),
        "topic": r["topic"],
        "status": r["status"],
        "run_id": r["run_id"],
        "iterate": bool(r["iterate"]),
        "target_score": r["target_score"],
        "max_iterations": int(r["max_iterations"]),
        "added_at": r["added_at"],
        "started_at": r["started_at"],
        "ended_at": r["ended_at"],
        "priority": int(r["priority"]),
        "notes": r["notes"],
    }
