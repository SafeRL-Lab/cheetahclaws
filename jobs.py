"""
jobs.py — Persistent job registry for remote control (phone → computer).

Every query received via Telegram/Slack/console creates a Job.
Jobs are tracked through their lifecycle: queued → running → done/failed.
Each Job records which tools were used (steps), result preview, and timing.

Usage (by bridges):
    import jobs
    job = jobs.create("帮我总结实验", source="telegram")
    jobs.start(job.id)
    jobs.add_step(job.id, "Bash", "pytest tests/")
    jobs.finish_step(job.id, "Bash", "5 passed")
    jobs.complete(job.id, result_preview="All tests passed. Summary: ...")
    jobs.fail(job.id, "pytest: command not found")

Usage (from phone via !jobs / !job <id> / !retry <id>):
    jobs.list_recent(10)      → [Job, ...]
    jobs.get(job_id)          → Job | None
    jobs.format_dashboard()   → str (mobile-friendly)
    jobs.format_detail(job_id) → str
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

_JOBS_PATH = Path.home() / ".cheetahclaws" / "jobs.json"
_MAX_JOBS = 100          # keep last N jobs on disk
_MAX_STEPS = 30          # max steps to record per job
_RESULT_PREVIEW = 600    # chars of result to store
_lock = threading.Lock()


# ── Data model ──────────────────────────────────────────────────────────────

class Job:
    __slots__ = (
        "id", "title", "prompt", "status", "source",
        "steps", "step_count", "current_step",
        "result", "error",
        "created_at", "started_at", "done_at", "duration_s",
        "retry_of",
    )

    def __init__(self, id: str, title: str, prompt: str,
                 status: str = "queued", source: str = "console",
                 steps: list | None = None, step_count: int = 0,
                 current_step: str = "",
                 result: str = "", error: str = "",
                 created_at: str = "", started_at: str = "", done_at: str = "",
                 duration_s: float = 0.0, retry_of: str = ""):
        self.id = id
        self.title = title
        self.prompt = prompt
        self.status = status          # queued|running|done|failed|cancelled
        self.source = source          # telegram|slack|console
        self.steps = steps or []      # list of step dicts: {name, preview, status}
        self.step_count = step_count  # total steps completed
        self.current_step = current_step
        self.result = result
        self.error = error
        self.created_at = created_at or _now()
        self.started_at = started_at
        self.done_at = done_at
        self.duration_s = duration_s
        self.retry_of = retry_of      # id of original job if this is a retry

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(**{k: d.get(k, "") for k in cls.__slots__})

    # ── Formatting ──────────────────────────────────────────────────────────

    def status_icon(self) -> str:
        return {
            "queued":    "⏳",
            "running":   "🔄",
            "done":      "✅",
            "failed":    "❌",
            "cancelled": "🚫",
        }.get(self.status, "❓")

    def age_str(self) -> str:
        """Human-readable age, e.g. '2m ago', 'just now'."""
        ts = self.done_at or self.started_at or self.created_at
        if not ts:
            return ""
        try:
            dt = datetime.fromisoformat(ts)
            secs = (datetime.now() - dt).total_seconds()
            if secs < 10:
                return "just now"
            if secs < 120:
                return f"{int(secs)}s ago"
            if secs < 7200:
                return f"{int(secs // 60)}m ago"
            return f"{int(secs // 3600)}h ago"
        except Exception:
            return ""

    def one_liner(self) -> str:
        icon = self.status_icon()
        age = self.age_str()
        steps_info = f" ({self.step_count} steps)" if self.step_count else ""
        dur = f" {self.duration_s:.0f}s" if self.duration_s else ""
        cur = f" — {self.current_step}" if self.status == "running" and self.current_step else ""
        err = f" — {self.error[:50]}" if self.status == "failed" and self.error else ""
        return f"{icon} #{self.id}  [{age}]  \"{self.title}\"{steps_info}{dur}{cur}{err}"

    def detail_card(self) -> str:
        lines = [
            f"{self.status_icon()} Job #{self.id} — {self.status.upper()}",
            f"'{self.prompt[:100]}'",
            "─" * 36,
        ]
        if self.started_at:
            lines.append(f"Started: {self.age_str()}  |  Source: {self.source}")
        if self.duration_s:
            lines.append(f"Duration: {self.duration_s:.1f}s")
        if self.steps:
            lines.append("")
            lines.append("Steps:")
            for s in self.steps[-_MAX_STEPS:]:
                icon = "✅" if s.get("status") == "done" else ("🔄" if s.get("status") == "running" else "○")
                preview = s.get("preview", "")
                lines.append(f"  {icon} {s['name']}" + (f": {preview[:40]}" if preview else ""))
        if self.result:
            lines.append("")
            lines.append("Result:")
            lines.append(self.result[:400])
        if self.error:
            lines.append("")
            lines.append(f"Error: {self.error[:200]}")
        if self.status == "failed":
            lines.append("")
            lines.append(f"↩ Retry with: !retry {self.id}")
        return "\n".join(lines)


# ── Storage ──────────────────────────────────────────────────────────────────
#
# F-2 swapped the JSON-file backend for the SQLite ``jobs`` table.  The
# legacy ``~/.cheetahclaws/jobs.json`` is migrated on first access and
# kept readable for one release as a fallback.  Public API
# (``create``, ``start``, ``get``, ``list_recent`` …) is unchanged.

_MIGRATION_KEY = "jobs_migrated_from_json"
_migration_done_in_process = False  # process-wide guard, avoids re-checking


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_migrated() -> None:
    """Idempotent one-shot import of the legacy JSON file.

    Tracked by a row in ``schema_meta`` so it survives across processes.
    A process-wide bool short-circuits subsequent calls.

    Note: this migration is **one-way**.  Once the schema_meta marker is
    set, the JSON file is never read again — subsequent edits to
    ``~/.cheetahclaws/jobs.json`` are ignored.  The file is left on disk
    so that users still on the prior release (or anyone holding a
    backup-style script that scrapes it) can read it, but it is no
    longer the source of truth.  To redo the migration, delete the
    ``jobs_migrated_from_json`` row from ``schema_meta`` AND the rows in
    the ``jobs`` table you want re-imported.
    """
    global _migration_done_in_process
    if _migration_done_in_process:
        return
    from cc_daemon.schema import get_conn
    conn = get_conn()
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key=?", (_MIGRATION_KEY,)
    ).fetchone()
    if row is None and _JOBS_PATH.exists():
        try:
            legacy = json.loads(_JOBS_PATH.read_text(encoding="utf-8"))
        except Exception:
            legacy = []
        for d in legacy if isinstance(legacy, list) else []:
            try:
                _persist(Job.from_dict(d), conn=conn)
            except Exception:
                continue
    if row is None:
        from datetime import timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        conn.execute(
            "INSERT INTO schema_meta (key, value, updated_at) "
            "VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
            "value=excluded.value, updated_at=excluded.updated_at",
            (_MIGRATION_KEY, "1", now),
        )
        conn.commit()
    _migration_done_in_process = True


def _row_to_job(row) -> Job:
    return Job(
        id=row["id"],
        title=row["title"] or "",
        prompt=row["prompt"] or "",
        status=row["status"],
        source=row["source"] or "",
        steps=json.loads(row["steps_json"]) if row["steps_json"] else [],
        step_count=row["step_count"] or 0,
        current_step=row["current_step"] or "",
        result=row["result"] or "",
        error=row["error"] or "",
        created_at=row["created_at"] or "",
        started_at=row["started_at"] or "",
        done_at=row["done_at"] or "",
        duration_s=row["duration_s"] or 0.0,
        retry_of=row["retry_of"] or "",
    )


def _persist(job: Job, conn=None) -> None:
    """INSERT or UPDATE the row for *job*.  Caller passes *conn* during
    migration to avoid re-entering ``get_conn`` from inside a transaction."""
    from cc_daemon.schema import get_conn
    c = conn if conn is not None else get_conn()
    c.execute(
        "INSERT INTO jobs (id, title, prompt, source, status, created_at, "
        "  started_at, done_at, duration_s, steps_json, step_count, "
        "  current_step, result, error, retry_of) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET "
        "  title=excluded.title, prompt=excluded.prompt, "
        "  source=excluded.source, status=excluded.status, "
        "  started_at=excluded.started_at, done_at=excluded.done_at, "
        "  duration_s=excluded.duration_s, steps_json=excluded.steps_json, "
        "  step_count=excluded.step_count, current_step=excluded.current_step,"
        "  result=excluded.result, error=excluded.error, "
        "  retry_of=excluded.retry_of",
        (job.id, job.title, job.prompt, job.source, job.status,
         job.created_at, job.started_at, job.done_at, job.duration_s,
         json.dumps(job.steps, ensure_ascii=False),
         job.step_count, job.current_step, job.result,
         job.error, job.retry_of),
    )
    if conn is None:
        c.commit()


def _prune_to_max(conn=None) -> None:
    """Keep only the most-recent ``_MAX_JOBS`` rows."""
    from cc_daemon.schema import get_conn
    c = conn if conn is not None else get_conn()
    excess = c.execute(
        "SELECT COUNT(*) FROM jobs"
    ).fetchone()[0] - _MAX_JOBS
    if excess > 0:
        c.execute(
            "DELETE FROM jobs WHERE id IN ("
            "  SELECT id FROM jobs ORDER BY created_at LIMIT ?"
            ")",
            (excess,),
        )
        if conn is None:
            c.commit()


def _get_all() -> list[Job]:
    _ensure_migrated()
    from cc_daemon.schema import get_conn
    rows = get_conn().execute(
        "SELECT * FROM jobs ORDER BY created_at"
    ).fetchall()
    return [_row_to_job(r) for r in rows]


def _update(job: Job) -> None:
    _ensure_migrated()
    with _lock:
        _persist(job)
        _prune_to_max()


# ── Public API ───────────────────────────────────────────────────────────────

def create(prompt: str, source: str = "console", retry_of: str = "") -> Job:
    """Create a new job in 'queued' state."""
    job = Job(
        id=uuid.uuid4().hex[:6],
        title=prompt[:60].replace("\n", " "),
        prompt=prompt,
        status="queued",
        source=source,
        retry_of=retry_of,
    )
    _update(job)
    return job


def start(job_id: str) -> None:
    """Mark job as running."""
    job = get(job_id)
    if job:
        job.status = "running"
        job.started_at = _now()
        _update(job)


def add_step(job_id: str, tool_name: str, preview: str = "") -> None:
    """Record a tool invocation (ToolStart event)."""
    job = get(job_id)
    if not job:
        return
    # Mark previous step as done if still running
    if job.steps and job.steps[-1].get("status") == "running":
        job.steps[-1]["status"] = "done"
    # Add new step
    if len(job.steps) < _MAX_STEPS:
        job.steps.append({
            "name": tool_name,
            "preview": preview[:80],
            "status": "running",
        })
    job.current_step = f"{tool_name}: {preview[:40]}" if preview else tool_name
    job.step_count = sum(1 for s in job.steps if s.get("status") == "done")
    _update(job)


def finish_step(job_id: str, tool_name: str, result_preview: str = "") -> None:
    """Record tool completion (ToolEnd event)."""
    job = get(job_id)
    if not job:
        return
    for s in reversed(job.steps):
        if s["name"] == tool_name and s.get("status") == "running":
            s["status"] = "done"
            if result_preview:
                s["result"] = result_preview[:80]
            break
    job.step_count = sum(1 for s in job.steps if s.get("status") == "done")
    job.current_step = ""
    _update(job)


def stream_result(job_id: str, chunk: str) -> None:
    """Append a chunk to the rolling result preview."""
    job = get(job_id)
    if not job:
        return
    job.result = (job.result + chunk)[-_RESULT_PREVIEW:]
    _update(job)


def complete(job_id: str, result_preview: str = "") -> None:
    """Mark job as successfully done."""
    job = get(job_id)
    if not job:
        return
    job.status = "done"
    job.done_at = _now()
    if result_preview:
        job.result = result_preview[-_RESULT_PREVIEW:]
    if job.started_at:
        try:
            job.duration_s = (
                datetime.fromisoformat(job.done_at) -
                datetime.fromisoformat(job.started_at)
            ).total_seconds()
        except Exception:
            pass
    # Mark any still-running steps as done
    for s in job.steps:
        if s.get("status") == "running":
            s["status"] = "done"
    job.step_count = sum(1 for s in job.steps if s.get("status") == "done")
    job.current_step = ""
    _update(job)


def fail(job_id: str, error: str) -> None:
    """Mark job as failed."""
    job = get(job_id)
    if not job:
        return
    job.status = "failed"
    job.done_at = _now()
    job.error = error[:300]
    if job.started_at:
        try:
            job.duration_s = (
                datetime.fromisoformat(job.done_at) -
                datetime.fromisoformat(job.started_at)
            ).total_seconds()
        except Exception:
            pass
    for s in job.steps:
        if s.get("status") == "running":
            s["status"] = "failed"
    job.current_step = ""
    _update(job)


def cancel(job_id: str) -> None:
    """Mark job as cancelled."""
    job = get(job_id)
    if not job:
        return
    job.status = "cancelled"
    job.done_at = _now()
    _update(job)


# ── Query ────────────────────────────────────────────────────────────────────

def get(job_id: str) -> Optional[Job]:
    _ensure_migrated()
    from cc_daemon.schema import get_conn
    row = get_conn().execute(
        "SELECT * FROM jobs WHERE id=?", (job_id,)
    ).fetchone()
    return _row_to_job(row) if row is not None else None


def list_recent(n: int = 10) -> list[Job]:
    """Return last N jobs, newest first."""
    _ensure_migrated()
    from cc_daemon.schema import get_conn
    rows = get_conn().execute(
        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (n,)
    ).fetchall()
    return [_row_to_job(r) for r in rows]


def list_running() -> list[Job]:
    _ensure_migrated()
    from cc_daemon.schema import get_conn
    rows = get_conn().execute(
        "SELECT * FROM jobs WHERE status='running' ORDER BY started_at"
    ).fetchall()
    return [_row_to_job(r) for r in rows]


# ── Dashboard formatting ──────────────────────────────────────────────────────

def format_dashboard(n: int = 8) -> str:
    jobs = list_recent(n)
    if not jobs:
        return "📊 No jobs yet. Send me something to do!"

    running = [j for j in jobs if j.status == "running"]
    recent = [j for j in jobs if j.status != "running"]

    lines = ["📊 Job Dashboard"]
    lines.append("─" * 36)

    if running:
        for j in running:
            lines.append(j.one_liner())

    if recent:
        if running:
            lines.append("")
        for j in recent[:6]:
            lines.append(j.one_liner())

    lines.append("")
    lines.append("!job <id>  !retry <id>  !cancel")
    return "\n".join(lines)


def format_detail(job_id: str) -> str:
    job = get(job_id)
    if not job:
        return f"❓ Job #{job_id} not found."
    return job.detail_card()
