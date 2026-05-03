"""web/lab_api.py — HTTP routes for the research lab UI.

Mounted under ``/api/lab/*`` from web/server.py. We expose JSON
endpoints so the frontend (web/lab.html) can drive the lab without
opening the cheetahclaws REPL.

Endpoints:

  POST   /api/lab/runs                   start a new run
  GET    /api/lab/runs                   list recent runs
  GET    /api/lab/runs/<id>              run detail (incl. stages)
  GET    /api/lab/runs/<id>/messages     recent agent messages
  GET    /api/lab/runs/<id>/report       final markdown report
  GET    /api/lab/runs/<id>/experiments  experiment log + artifacts
  GET    /api/lab/runs/<id>/artifacts/<fn>  download a workspace file
  POST   /api/lab/runs/<id>/abort        request cancellation

Auth: this module reuses the ``_check_auth`` cookie/JWT machinery from
web/server.py via the caller; lab_api itself is auth-agnostic and just
gets a ``user_id`` parameter.

v0 is single-tenant — runs aren't scoped per user yet. Multi-tenant is
Phase 4.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Optional

# Single-process state shared with commands/lab_cmd.py — both surfaces
# can spawn / abort the same runs.
_cancel_flags: dict[str, threading.Event] = {}
_run_threads: dict[str, threading.Thread] = {}


# Path patterns we recognize.  Order matters: more-specific patterns first.
_RUN_DETAIL    = re.compile(r"^/api/lab/runs/(?P<id>lab_[a-f0-9]+)$")
_RUN_MESSAGES  = re.compile(r"^/api/lab/runs/(?P<id>lab_[a-f0-9]+)/messages$")
_RUN_REPORT    = re.compile(r"^/api/lab/runs/(?P<id>lab_[a-f0-9]+)/report$")
_RUN_EXPS      = re.compile(r"^/api/lab/runs/(?P<id>lab_[a-f0-9]+)/experiments$")
_RUN_ARTIFACT  = re.compile(
    r"^/api/lab/runs/(?P<id>lab_[a-f0-9]+)/artifacts/(?P<fn>[\w\.\-]+)$",
)
_RUN_ABORT     = re.compile(r"^/api/lab/runs/(?P<id>lab_[a-f0-9]+)/abort$")


def matches_lab_path(path: str) -> bool:
    return path.startswith("/api/lab/") or path == "/api/lab"


# ── Public dispatcher ─────────────────────────────────────────────────────


def dispatch(path: str, method: str, query: dict, body_json: dict,
             config: dict) -> tuple[int, str, bytes]:
    """Top-level lab API dispatcher.

    Returns (status_code, content_type, body_bytes).  The caller in
    web/server.py writes the response with its existing _send_http
    helper. We don't touch sockets here so unit tests can drive this
    function directly.
    """
    # POST /api/lab/runs
    if path == "/api/lab/runs":
        if method == "POST":
            return _start_run(body_json, config)
        if method == "GET":
            return _list_runs(query)
        return _err(405, "method not allowed")

    m = _RUN_ABORT.match(path)
    if m:
        if method != "POST":
            return _err(405, "method not allowed")
        return _abort_run(m.group("id"))

    m = _RUN_MESSAGES.match(path)
    if m:
        return _get_messages(m.group("id"), query)

    m = _RUN_REPORT.match(path)
    if m:
        return _get_report(m.group("id"))

    m = _RUN_EXPS.match(path)
    if m:
        return _get_experiments(m.group("id"))

    m = _RUN_ARTIFACT.match(path)
    if m:
        return _get_artifact(m.group("id"), m.group("fn"))

    m = _RUN_DETAIL.match(path)
    if m:
        return _run_detail(m.group("id"))

    return _err(404, f"unknown lab endpoint: {path}")


# ── Endpoint impls ────────────────────────────────────────────────────────


def _start_run(body: dict, config: dict) -> tuple[int, str, bytes]:
    topic = (body.get("topic") or "").strip()
    if not topic:
        return _err(400, "topic is required")
    budget_tokens = int(body.get("budget_tokens",
                                  config.get("lab_budget_tokens", 5_000_000)))
    budget_cost_cents = int(body.get("budget_cost_cents",
                                       config.get("lab_budget_cost_cents", 5000)))
    max_rounds = int(body.get("max_rounds",
                                config.get("lab_max_rounds", 5)))
    role_override = body.get("role_override") or config.get("lab_role_override") or {}

    from research.lab.storage import LabStorage
    from research.lab.orchestrator import (
        _drive, LabRun, LabState, Stage,
    )
    from research.lab.roles import build_default_assignment
    from research.lab.convergence import ConvergenceConfig

    storage = LabStorage()
    rec = storage.create_run(
        topic=topic,
        budget_tokens=budget_tokens,
        budget_cost_cents=budget_cost_cents,
        max_rounds=max_rounds,
    )
    cancel = threading.Event()
    _cancel_flags[rec.run_id] = cancel

    def _runner():
        roles = build_default_assignment(config, override=role_override)
        state = LabState(run_id=rec.run_id, topic=topic, stage=Stage.QUESTIONING)
        run = LabRun(
            state=state, storage=storage, roles=roles, config=config,
            convergence=ConvergenceConfig(max_rounds=max_rounds),
        )
        storage.update_run_status(rec.run_id, "running",
                                   current_stage=state.stage.value)
        try:
            _drive(run, cancel_check=cancel.is_set)
            if state.cancel_requested:
                storage.update_run_status(rec.run_id, "aborted",
                                          current_stage=state.stage.value)
            else:
                storage.update_run_status(rec.run_id, "done",
                                          current_stage=state.stage.value)
        except Exception as exc:
            storage.update_run_status(rec.run_id, "failed",
                                      current_stage=state.stage.value,
                                      error=str(exc))

    t = threading.Thread(target=_runner, name=f"lab-{rec.run_id}", daemon=True)
    _run_threads[rec.run_id] = t
    t.start()

    return _ok({
        "run_id": rec.run_id,
        "topic": topic,
        "budget_tokens": budget_tokens,
        "budget_cost_cents": budget_cost_cents,
        "max_rounds": max_rounds,
    })


def _list_runs(query: dict) -> tuple[int, str, bytes]:
    from research.lab.storage import LabStorage
    s = LabStorage()
    limit = int(query.get("limit", 50))
    status_filter = query.get("status") or None
    runs = s.list_runs(status=status_filter, limit=limit)
    out = []
    for r in runs:
        tok, cents = s.get_budget(r.run_id)
        out.append({
            "run_id": r.run_id,
            "topic": r.topic,
            "status": r.status,
            "current_stage": r.current_stage,
            "tokens_used": tok,
            "tokens_budget": r.budget_tokens,
            "cost_cents": cents,
            "cost_cents_budget": r.budget_cost_cents,
            "created_at": r.created_at,
            "completed_at": r.completed_at,
            "error": r.error,
        })
    return _ok({"runs": out})


def _run_detail(run_id: str) -> tuple[int, str, bytes]:
    from research.lab.storage import LabStorage
    s = LabStorage()
    r = s.get_run(run_id)
    if r is None:
        return _err(404, f"no such run: {run_id}")
    tok, cents = s.get_budget(run_id)
    stages = s.list_stages(run_id)
    return _ok({
        "run_id": r.run_id,
        "topic": r.topic,
        "status": r.status,
        "current_stage": r.current_stage,
        "tokens_used": tok,
        "tokens_budget": r.budget_tokens,
        "cost_cents": cents,
        "cost_cents_budget": r.budget_cost_cents,
        "max_rounds": r.max_rounds,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
        "completed_at": r.completed_at,
        "error": r.error,
        "stages": [
            {
                "stage": st.stage, "round": st.round,
                "started_at": st.started_at, "ended_at": st.ended_at,
                "outcome": st.outcome, "notes": st.notes,
            }
            for st in stages
        ],
        "in_process_cancellable": run_id in _cancel_flags,
    })


def _get_messages(run_id: str, query: dict) -> tuple[int, str, bytes]:
    from research.lab.storage import LabStorage
    s = LabStorage()
    if s.get_run(run_id) is None:
        return _err(404, f"no such run: {run_id}")
    limit = int(query.get("limit", 100))
    stage = query.get("stage") or None
    msgs = s.list_messages(run_id, stage=stage, limit=limit)
    return _ok({
        "messages": [
            {
                "ts": m.ts, "stage": m.stage, "round": m.round,
                "role": m.role, "kind": m.kind,
                # Cap content per message to keep response sizes sane.
                "content": (m.content[:8000]
                             + ("…\n[+truncated]"
                                if len(m.content) > 8000 else "")),
                "meta": m.meta,
            }
            for m in msgs
        ]
    })


def _get_report(run_id: str) -> tuple[int, str, bytes]:
    from research.lab.storage import LabStorage, DEFAULT_OUTPUT_DIR
    s = LabStorage()
    r = s.get_run(run_id)
    if r is None:
        return _err(404, f"no such run: {run_id}")
    art = s.get_latest_artifact(run_id, "report")
    if art:
        return (200, "text/markdown; charset=utf-8",
                art.content.encode("utf-8"))
    # Fallback to file on disk
    path = DEFAULT_OUTPUT_DIR / run_id / "report.md"
    if path.exists():
        return (200, "text/markdown; charset=utf-8",
                path.read_bytes())
    return _err(404, "report not yet available")


def _get_experiments(run_id: str) -> tuple[int, str, bytes]:
    from research.lab.storage import LabStorage
    s = LabStorage()
    if s.get_run(run_id) is None:
        return _err(404, f"no such run: {run_id}")
    exps = s.list_experiments(run_id)
    return _ok({
        "experiments": [
            {
                "attempt": e.attempt, "exit_code": e.exit_code,
                "duration_s": e.duration_s, "timed_out": e.timed_out,
                "stdout": (e.stdout or "")[:8000],
                "stderr": (e.stderr or "")[:4000],
                "code": (e.code or "")[:4000],
                "artifacts": e.artifacts,
                "started_at": e.started_at, "ended_at": e.ended_at,
            }
            for e in exps
        ]
    })


def _get_artifact(run_id: str, filename: str) -> tuple[int, str, bytes]:
    """Serve a file from the run's workspace dir.  Read-only, sandboxed
    to the workspace; basic path-traversal guard."""
    from research.lab.storage import DEFAULT_OUTPUT_DIR
    if "/" in filename or "\\" in filename or filename.startswith("."):
        return _err(400, "invalid filename")
    path = DEFAULT_OUTPUT_DIR / run_id / "workspace" / filename
    try:
        path = path.resolve()
        ws_root = (DEFAULT_OUTPUT_DIR / run_id / "workspace").resolve()
        if not str(path).startswith(str(ws_root)):
            return _err(400, "path traversal blocked")
    except Exception:
        return _err(400, "invalid path")
    if not path.exists() or not path.is_file():
        return _err(404, "artifact not found")
    ext = path.suffix.lower()
    content_type = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".pdf": "application/pdf", ".svg": "image/svg+xml",
        ".csv": "text/csv", ".tsv": "text/tab-separated-values",
        ".json": "application/json", ".log": "text/plain",
        ".txt": "text/plain",
    }.get(ext, "application/octet-stream")
    return (200, content_type, path.read_bytes())


def _abort_run(run_id: str) -> tuple[int, str, bytes]:
    flag = _cancel_flags.get(run_id)
    if flag is None:
        return _err(404, "no in-process run for that id")
    flag.set()
    return _ok({"ok": True, "run_id": run_id,
                "message": "cancellation requested; current stage will finish"})


# ── Helpers ───────────────────────────────────────────────────────────────


def _ok(obj: Any) -> tuple[int, str, bytes]:
    return (200, "application/json", json.dumps(obj).encode("utf-8"))


def _err(status: int, msg: str) -> tuple[int, str, bytes]:
    return (status, "application/json",
            json.dumps({"error": msg}).encode("utf-8"))
