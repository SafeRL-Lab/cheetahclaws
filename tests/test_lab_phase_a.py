"""Phase A tests: resume, iterate (meta-loop), backlog, daemon worker.

These tests stub the LLM with a controllable fake so we can drive the
full state machine deterministically without hitting any real API.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from cheetahclaws.research.lab import iterate, resume, backlog
from cheetahclaws.research.lab.iterate import (
    DIMENSIONS,
    DIMENSION_TO_STAGE,
    IterationConfig,
    aggregate_scores,
    overall_score,
    parse_review_scores,
    weakest_dimension,
)
from cheetahclaws.research.lab.orchestrator import LLMResponse, Stage
from cheetahclaws.research.lab.storage import LabStorage


# ── Fake LLM ──────────────────────────────────────────────────────────────


class FakeLLM:
    """Records every call; returns a programmable response per role.

    ``set_response(role_name, text)`` sets a sticky reply.  Default reply
    is a benign "OK" so unmatched calls don't blow up.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._by_role: dict[str, str] = {}
        self.default_text = "OK"

    def set_response(self, role_name: str, text: str) -> None:
        self._by_role[role_name] = text

    def __call__(self, *, role_name, model, system, user, config):
        self.calls.append({
            "role": role_name, "model": model,
            "system": system, "user": user,
        })
        text = self._by_role.get(role_name, self.default_text)
        return LLMResponse(text=text, tokens_in=10, tokens_out=10, cost_cents=1)


@pytest.fixture
def storage(tmp_path):
    db_path = tmp_path / "lab.db"
    s = LabStorage(db_path=db_path)
    yield s
    s.close()


# ── Score parser tests ────────────────────────────────────────────────────


def test_parse_review_scores_clean():
    txt = "novelty: 6\nrigor: 7.5\nclarity: 8\nevidence: 5"
    out = parse_review_scores(txt)
    assert out == {"novelty": 6.0, "rigor": 7.5, "clarity": 8.0, "evidence": 5.0}


def test_parse_review_scores_clamps_range():
    txt = "novelty: 11\nrigor: -1\nclarity: 5\nevidence: 5"
    out = parse_review_scores(txt)
    # 11 → 10, -1 → 0 (regex actually rejects -1 since pattern doesn't allow
    # minus, so missing → defaults to 5; that's fine)
    assert out["novelty"] == 10.0
    assert out["rigor"] == 5.0    # "-1" doesn't match → default
    assert out["clarity"] == 5.0
    assert out["evidence"] == 5.0


def test_parse_review_scores_missing_dim_defaults_to_5():
    txt = "novelty: 7"
    out = parse_review_scores(txt)
    assert out == {"novelty": 7.0, "rigor": 5.0, "clarity": 5.0, "evidence": 5.0}


def test_parse_review_scores_empty():
    assert parse_review_scores("") == {d: 5.0 for d in DIMENSIONS}
    assert parse_review_scores("garbage no dims here") == {d: 5.0 for d in DIMENSIONS}


def test_aggregate_and_overall():
    per_reviewer = [
        {"novelty": 6, "rigor": 7, "clarity": 8, "evidence": 5},
        {"novelty": 8, "rigor": 7, "clarity": 6, "evidence": 5},
    ]
    by_dim = aggregate_scores(per_reviewer)
    assert by_dim["novelty"] == 7.0
    assert by_dim["rigor"] == 7.0
    assert overall_score(by_dim) == pytest.approx(6.5)


def test_weakest_dimension():
    by_dim = {"novelty": 8, "rigor": 5, "clarity": 9, "evidence": 7}
    assert weakest_dimension(by_dim) == "rigor"


# ── State reconstruction tests ────────────────────────────────────────────


def _seed_run(storage, *, stage: Stage = Stage.FINALIZATION):
    """Create a fully-populated run record so resume has something to read."""
    rec = storage.create_run(topic="carbon-aware ML training")
    storage.update_run_status(rec.run_id, "done", current_stage=stage.value)
    storage.put_artifact(rec.run_id, "rq", "Q1\nQ2\nQ3")
    storage.put_artifact(rec.run_id, "survey", "Survey summary")
    storage.put_artifact(rec.run_id, "outline", "1. intro\n2. methods")
    storage.put_artifact(rec.run_id, "experiment_code_v1", "print('hi')")
    storage.put_artifact(rec.run_id, "experiment_code_v2", "print('v2')")
    storage.record_experiment(
        run_id=rec.run_id, attempt=2, code="print('v2')",
        exit_code=0, stdout="RESULT: {\"acc\": 0.91}\n", stderr="",
        duration_s=1.2, artifacts=["fig1.png"],
    )
    storage.put_artifact(rec.run_id, "results_section", "## Results\n0.91")
    storage.put_artifact(rec.run_id, "draft_full",
                         "# Paper\n## Abstract\n…\n## References")
    storage.put_artifact(rec.run_id, "report", "# FINAL REPORT\n…")
    return rec


def test_reconstruct_state_full(storage):
    rec = _seed_run(storage)
    state = resume.reconstruct_state(storage, rec.run_id)
    # Default resume = continue from saved stage (FINALIZATION here).
    # Everything *before* FINALIZATION should be loaded:
    assert state.stage == Stage.FINALIZATION
    assert state.research_questions == ["Q1", "Q2", "Q3"]
    assert state.survey_summary == "Survey summary"
    assert state.outline.startswith("1. intro")
    assert state.experiment_code == "print('v2')"
    assert state.experiment_attempt == 2
    assert state.experiment_result is not None
    assert state.experiment_result.exit_code == 0
    assert state.results_section.startswith("## Results")
    assert state.section_drafts.get("full_body", "").startswith("# Paper")


def test_reconstruct_state_rewind_to_drafting(storage):
    """Rewinding to DRAFTING should drop the draft + verification artifacts
    (so the orchestrator regenerates them) but keep everything earlier."""
    rec = _seed_run(storage)
    state = resume.reconstruct_state(storage, rec.run_id, start_stage=Stage.DRAFTING)
    assert state.stage == Stage.DRAFTING
    assert state.outline                   # earlier — kept
    assert state.experiment_code           # earlier — kept
    assert state.results_section           # produced by ANALYSIS, before DRAFTING — kept
    assert state.section_drafts == {}      # produced by DRAFTING itself — dropped


def test_reconstruct_state_rewind_to_questioning(storage):
    rec = _seed_run(storage)
    state = resume.reconstruct_state(storage, rec.run_id, start_stage=Stage.QUESTIONING)
    assert state.stage == Stage.QUESTIONING
    assert state.research_questions == []
    assert state.survey_summary == ""
    assert state.outline == ""
    assert state.experiment_code == ""
    assert state.section_drafts == {}


def test_reconstruct_state_unknown_run_raises(storage):
    with pytest.raises(ValueError):
        resume.reconstruct_state(storage, "lab_does_not_exist")


# ── Iteration: scoring ────────────────────────────────────────────────────


def test_score_report_aggregates_across_reviewers(storage):
    rec = _seed_run(storage)
    fake = FakeLLM()
    # Three reviewers with different opinions.
    fake.set_response("reviewer_1",
                      "novelty: 6\nrigor: 7\nclarity: 7\nevidence: 5")
    fake.set_response("reviewer_2",
                      "novelty: 7\nrigor: 8\nclarity: 6\nevidence: 6")
    fake.set_response("reviewer_3",
                      "novelty: 8\nrigor: 7\nclarity: 8\nevidence: 7")

    from cheetahclaws.research.lab.orchestrator import LabRun, LabState
    from cheetahclaws.research.lab.roles import build_default_assignment
    roles = build_default_assignment({})
    state = LabState(run_id=rec.run_id, topic=rec.topic, stage=Stage.FINALIZATION)
    run = LabRun(state=state, storage=storage, roles=roles, config={},
                 call_llm=fake)

    by_dim = iterate.score_report(
        run=run, report_text="# FAKE\nbody",
        reviewers=roles.reviewers[:3], call_llm=fake,
    )
    assert by_dim["novelty"] == 7.0
    assert by_dim["rigor"] == pytest.approx(7.333, abs=0.01)
    assert by_dim["evidence"] == 6.0


# ── Iteration: end-to-end (with rewind to weakest) ────────────────────────


def test_run_one_iteration_converges_when_score_is_high(storage, monkeypatch):
    rec = _seed_run(storage)
    fake = FakeLLM()
    # Above target on all dims.
    for i in range(1, 4):
        fake.set_response(f"reviewer_{i}",
                          "novelty: 8\nrigor: 8\nclarity: 8\nevidence: 8")

    cfg = IterationConfig(target_score=7.0, max_iterations=3)
    result = iterate.run_one_iteration(
        run_id=rec.run_id, iter_n=1, config={}, iter_cfg=cfg,
        storage_obj=storage, call_llm=fake,
    )
    assert result.score_avg == 8.0
    assert result.revise_stage is None  # converged

    rows = storage.list_iterations(rec.run_id)
    assert len(rows) == 1
    assert rows[0]["status"] == "done"
    assert rows[0]["score_avg"] == 8.0


def test_run_one_iteration_routes_low_evidence_to_experiment(storage, monkeypatch):
    rec = _seed_run(storage)
    fake = FakeLLM()
    # Low evidence — should rewind to EXPERIMENT.
    for i in range(1, 4):
        fake.set_response(f"reviewer_{i}",
                          "novelty: 7\nrigor: 7\nclarity: 7\nevidence: 3")

    # Patch resume_run to a no-op so the test doesn't actually rerun the
    # orchestrator (we're testing routing, not the full pipeline).
    called = {}
    def _noop_resume(**kwargs):
        called.update(kwargs)
        class _R: pass
        return _R()
    monkeypatch.setattr(iterate, "resume_run", _noop_resume)

    cfg = IterationConfig(target_score=7.0, max_iterations=3)
    result = iterate.run_one_iteration(
        run_id=rec.run_id, iter_n=1, config={}, iter_cfg=cfg,
        storage_obj=storage, call_llm=fake,
    )
    assert result.revise_stage == DIMENSION_TO_STAGE["evidence"]
    assert result.revise_stage == Stage.EXPERIMENT
    assert called["start_stage"] == Stage.EXPERIMENT
    assert called["run_id"] == rec.run_id


def test_iterate_until_converged_stops_at_target(storage, monkeypatch):
    rec = _seed_run(storage)
    fake = FakeLLM()
    # First call: low score → rewind. Second call: high score → converge.
    rounds = {"n": 0}
    def _resp_for(role):
        rounds["n"] += 1
        if rounds["n"] <= 3:   # 3 reviewers per round
            return "novelty: 7\nrigor: 5\nclarity: 7\nevidence: 7"
        return "novelty: 8\nrigor: 8\nclarity: 8\nevidence: 8"
    # Replace FakeLLM with a callable that varies per call.
    class StatefulLLM:
        def __init__(self):
            self.calls = 0
        def __call__(self, *, role_name, model, system, user, config):
            self.calls += 1
            text = _resp_for(role_name)
            return LLMResponse(text=text, tokens_in=10, tokens_out=10, cost_cents=1)
    llm = StatefulLLM()

    monkeypatch.setattr(iterate, "resume_run", lambda **kw: None)

    cfg = IterationConfig(target_score=7.0, max_iterations=5)
    history = iterate.iterate_until_converged(
        run_id=rec.run_id, config={}, iter_cfg=cfg,
        storage_obj=storage, call_llm=llm,
    )
    assert len(history) == 2
    assert history[0].revise_stage is not None       # iter 1 needed revision
    assert history[1].revise_stage is None           # iter 2 converged


def test_iterate_until_converged_stops_at_max(storage, monkeypatch):
    rec = _seed_run(storage)
    class LowLLM:
        def __call__(self, *, role_name, model, system, user, config):
            return LLMResponse(
                text="novelty: 4\nrigor: 4\nclarity: 4\nevidence: 4",
                tokens_in=10, tokens_out=10, cost_cents=1,
            )
    monkeypatch.setattr(iterate, "resume_run", lambda **kw: None)
    cfg = IterationConfig(target_score=7.0, max_iterations=3)
    history = iterate.iterate_until_converged(
        run_id=rec.run_id, config={}, iter_cfg=cfg,
        storage_obj=storage, call_llm=LowLLM(),
    )
    assert len(history) == 3   # hit the cap
    assert all(r.revise_stage is not None for r in history)


# ── Backlog tests ─────────────────────────────────────────────────────────


def test_backlog_add_list_remove(storage):
    mgr = backlog.BacklogManager(storage)
    a = mgr.add(topic="A", priority=0)
    b = mgr.add(topic="B", priority=5)
    c = mgr.add(topic="C", iterate=True, target_score=6.5)
    items = mgr.list()
    assert len(items) == 3
    # priority desc → B (5) first
    assert items[0]["topic"] == "B"
    assert items[1]["topic"] == "A" or items[1]["topic"] == "C"
    assert mgr.remove(a) is True
    assert mgr.remove(99999) is False
    assert len(mgr.list()) == 2


def test_claim_next_backlog_atomic(storage):
    storage.add_backlog(topic="A", priority=0)
    storage.add_backlog(topic="B", priority=10)
    first = storage.claim_next_backlog()
    second = storage.claim_next_backlog()
    third = storage.claim_next_backlog()
    assert first["topic"] == "B"
    assert second["topic"] == "A"
    assert third is None
    # Both claimed items now show running.
    items = storage.list_backlog()
    statuses = sorted(it["status"] for it in items)
    assert statuses == ["running", "running"]


def test_reset_running_backlog(storage):
    storage.add_backlog(topic="A")
    storage.claim_next_backlog()
    n = storage.reset_running_backlog()
    assert n == 1
    items = storage.list_backlog()
    assert items[0]["status"] == "pending"


# ── Worker loop (with stubbed orchestrator) ───────────────────────────────


def test_worker_picks_up_pending_item_and_marks_done(storage, monkeypatch):
    """Worker should claim → run → mark done; idempotent on stop."""
    storage.add_backlog(topic="topic X")

    # Fake out run_one_lab_session so we don't run the real orchestrator.
    seen_topics = []
    class FakeRun:
        def __init__(self, topic):
            self.state = type("S", (), {"run_id": "lab_fake_" + topic[:5],
                                          "topic": topic, "stage": "done"})()
    def _fake_run(*, topic, **kwargs):
        seen_topics.append(topic)
        return FakeRun(topic)
    monkeypatch.setattr(backlog, "run_one_lab_session", _fake_run)

    stop = threading.Event()
    t = threading.Thread(
        target=backlog.run_backlog_worker,
        kwargs={"config": {}, "stop_event": stop, "poll_interval_s": 0.05,
                "storage_obj": storage},
        daemon=True,
    )
    t.start()
    # Give it time to claim + finish.
    deadline = time.time() + 3
    while time.time() < deadline:
        items = storage.list_backlog()
        if items and items[0]["status"] == "done":
            break
        time.sleep(0.05)
    stop.set()
    t.join(timeout=2)

    assert seen_topics == ["topic X"]
    items = storage.list_backlog()
    assert len(items) == 1
    assert items[0]["status"] == "done"
    assert items[0]["run_id"] is not None


def test_worker_stops_promptly_on_empty_queue(storage):
    stop = threading.Event()
    t = threading.Thread(
        target=backlog.run_backlog_worker,
        kwargs={"config": {}, "stop_event": stop, "poll_interval_s": 0.05,
                "storage_obj": storage},
        daemon=True,
    )
    t.start()
    time.sleep(0.2)
    stop.set()
    t.join(timeout=2)
    assert not t.is_alive()


def test_daemon_singleton(storage, monkeypatch):
    """start_daemon is idempotent; stop_daemon clears the slot."""
    # Make the worker tight + harmless so the test runs fast.
    monkeypatch.setattr(backlog, "run_one_lab_session",
                        lambda **kw: type("R", (),
                            {"state": type("S", (), {"run_id": "lab_x"})()})())
    h1 = backlog.start_daemon(config={}, storage_obj=storage)
    h2 = backlog.start_daemon(config={}, storage_obj=storage)
    assert h1 is h2          # idempotent
    assert backlog.stop_daemon() is True
    assert backlog.stop_daemon() is False  # already stopped
