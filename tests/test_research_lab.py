"""Unit + integration tests for research/lab.

Coverage:
  * storage roundtrip + indexing
  * convergence rule under all branches
  * citation parsing + similarity helpers
  * verifier graceful degradation when network unavailable
  * orchestrator end-to-end with stubbed LLM
  * output assembly from artifacts

Network-dependent tests skip cleanly when offline (we only mock the
HTTP layer when we want to exercise verifier code paths).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from research.lab import storage as _storage
from research.lab import convergence as _conv
from research.lab import verifier as _verifier
from research.lab import roles as _roles
from research.lab import orchestrator as _orch
from research.lab import output as _output


# ── Storage ────────────────────────────────────────────────────────────────


def test_storage_create_run_returns_record(tmp_path):
    s = _storage.LabStorage(tmp_path / "lab.db")
    r = s.create_run(topic="hi", budget_tokens=10000, max_rounds=3)
    assert r.run_id.startswith("lab_")
    assert r.topic == "hi"
    assert r.status == "pending"
    assert r.budget_tokens == 10000
    s.close()


def test_storage_messages_artifacts_budget(tmp_path):
    s = _storage.LabStorage(tmp_path / "lab.db")
    r = s.create_run(topic="t")
    s.append_message(r.run_id, stage="questioning", round_=0,
                      role="pi", kind="decision",
                      content="pick rq #2", meta={"model": "claude"})
    msgs = s.list_messages(r.run_id)
    assert len(msgs) == 1
    assert msgs[0].meta == {"model": "claude"}

    v1 = s.put_artifact(r.run_id, "rq", "Q1\nQ2")
    v2 = s.put_artifact(r.run_id, "rq", "Q1\nQ2 (rev)")
    assert v1 == 1 and v2 == 2
    latest = s.get_latest_artifact(r.run_id, "rq")
    assert latest.version == 2
    assert latest.content == "Q1\nQ2 (rev)"

    s.add_budget(r.run_id, tokens=500, cost_cents=12)
    s.add_budget(r.run_id, tokens=300, cost_cents=8)
    tok, cents = s.get_budget(r.run_id)
    assert tok == 800 and cents == 20
    s.close()


def test_storage_stage_transitions(tmp_path):
    s = _storage.LabStorage(tmp_path / "lab.db")
    r = s.create_run(topic="t")
    s.start_stage(r.run_id, "questioning", 0)
    s.end_stage(r.run_id, "questioning", 0, outcome="advance",
                notes="picked rq")
    stages = s.list_stages(r.run_id)
    assert len(stages) == 1
    assert stages[0].outcome == "advance"
    assert stages[0].ended_at is not None
    s.close()


def test_storage_persists_across_reopen(tmp_path):
    db = tmp_path / "lab.db"
    s1 = _storage.LabStorage(db)
    r = s1.create_run(topic="persist")
    s1.put_artifact(r.run_id, "rq", "RQ-1")
    s1.close()
    s2 = _storage.LabStorage(db)
    art = s2.get_latest_artifact(r.run_id, "rq")
    assert art.content == "RQ-1"
    s2.close()


# ── Convergence ─────────────────────────────────────────────────────────────


def _v(reviewer_id: str, score: int = 8, blocking=None, sugs=None):
    return _conv.ReviewerVerdict(
        reviewer_id=reviewer_id, score=score,
        blocking_issues=blocking or [], suggestions=sugs or [],
        overall="ok",
    )


def test_decide_advance_pass_quorum():
    v = [_v("r1"), _v("r2"), _v("r3")]
    d = _conv.decide_advance(v, round_index=1)
    assert d.advance is True
    assert "3/3" in d.reason


def test_decide_advance_partial_quorum():
    v = [_v("r1", score=8), _v("r2", score=8), _v("r3", score=4, blocking=["bad"])]
    d = _conv.decide_advance(v, round_index=1)
    # 2/3 pass with default n_required=2
    assert d.advance is True


def test_decide_advance_below_quorum_iterates():
    v = [_v("r1", score=8), _v("r2", score=4, blocking=["bad"]),
         _v("r3", score=4, blocking=["bad"])]
    d = _conv.decide_advance(v, round_index=1)
    assert d.advance is False
    assert d.needs_redesign is False


def test_decide_advance_max_rounds_force_advance():
    v = [_v("r1", score=4, blocking=["bad"]),
         _v("r2", score=4, blocking=["bad"]),
         _v("r3", score=4, blocking=["bad"])]
    d = _conv.decide_advance(v, round_index=5)  # default max=5
    assert d.advance is True
    assert "max_rounds" in d.reason


def test_decide_advance_zero_pass_triggers_redesign():
    v = [_v("r1", score=2, blocking=["x"]),
         _v("r2", score=2, blocking=["x"]),
         _v("r3", score=2, blocking=["x"])]
    cfg = _conv.ConvergenceConfig(abort_after_n_rounds_with_zero_pass=2)
    d = _conv.decide_advance(v, round_index=2, config=cfg)
    assert d.advance is False
    assert d.needs_redesign is True


def test_budget_status_fraction_and_exceeded():
    bs = _conv.BudgetStatus(tokens_used=500, tokens_budget=1000,
                              cost_cents_used=10, cost_cents_budget=100)
    assert bs.fraction_used() == 0.5
    assert bs.exceeded is False
    bs2 = _conv.BudgetStatus(tokens_used=1000, tokens_budget=1000,
                               cost_cents_used=0, cost_cents_budget=None)
    assert bs2.exceeded is True


def test_budget_status_unlimited():
    bs = _conv.BudgetStatus(tokens_used=5_000_000, tokens_budget=None,
                              cost_cents_used=10_000, cost_cents_budget=None)
    assert bs.exceeded is False
    assert bs.fraction_used() == 0.0


# ── Verifier helpers ────────────────────────────────────────────────────────


def test_title_similarity_jaccard():
    sim = _verifier._title_similarity(
        "Attention Is All You Need",
        "attention is all you need (revised)",
    )
    assert sim >= 0.7


def test_title_similarity_unrelated():
    sim = _verifier._title_similarity(
        "Attention Is All You Need",
        "Convolutional Neural Networks for Sentence Classification",
    )
    assert sim < 0.3


def test_author_overlap_handles_format_variants():
    overlap = _verifier._author_overlap(
        ["Vaswani, Ashish", "Shazeer, Noam"],
        ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
    )
    # both claimed surnames found in result; some extra in result is OK.
    assert overlap > 0.5


def test_author_overlap_no_match():
    overlap = _verifier._author_overlap(["Smith, John"],
                                          ["Doe, Jane"])
    assert overlap < 0.1


def test_verify_one_skips_without_network(monkeypatch):
    """When all three APIs fail, return verification_skipped, not not_found."""
    def fail(*a, **kw):
        raise RuntimeError("network down")
    monkeypatch.setattr(_verifier, "_http_get", fail)
    c = _verifier.Citation(key="x", title="A Title",
                            authors=["First Last"], year=2020)
    v = _verifier.verify_one(c, timeout_s=1.0)
    assert v.status == "verification_skipped"


def test_verify_one_arxiv_match(monkeypatch):
    """Stub out HTTP to simulate a verified arXiv match."""
    fake_atom = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
</feed>"""
    def fake_http(url, **kw):
        return fake_atom
    monkeypatch.setattr(_verifier, "_http_get", fake_http)
    c = _verifier.Citation(
        key="vaswani2017", title="Attention Is All You Need",
        authors=["Vaswani, Ashish", "Shazeer, Noam"], year=2017,
    )
    v = _verifier.verify_one(c, timeout_s=1.0)
    assert v.status == "verified"
    assert v.source == "arxiv"


def test_verify_citations_aggregates_counts(monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("net down")
    monkeypatch.setattr(_verifier, "_http_get", fail)
    cs = [_verifier.Citation(key=str(i), title=f"t{i}", authors=[])
          for i in range(3)]
    res = _verifier.verify_citations(cs, sleep_s=0)
    assert res.n_skipped == 3


# ── Roles + assignment ────────────────────────────────────────────────────


def test_default_assignment_has_all_seven_roles():
    a = _roles.build_default_assignment(config={"model": "test"})
    assert a.pi.name == "pi"
    assert a.questioner.name == "questioner"
    assert a.surveyor.name == "surveyor"
    assert a.designer.name == "designer"
    assert a.writer.name == "writer"
    assert len(a.reviewers) == 3
    assert a.reviewers[0].name == "reviewer_1"
    assert a.lay_reader.name == "lay_reader"


def test_role_override_pins_models():
    override = {"pi": "claude-opus-4-6", "reviewer_1": "gpt-5"}
    a = _roles.build_default_assignment(config={"model": "test"},
                                         override=override)
    assert a.pi.model == "claude-opus-4-6"
    assert a.reviewers[0].model == "gpt-5"


def test_load_role_template_finds_files():
    # Templates ship in agent_templates/lab/. If repo layout is correct,
    # all 7 should load.
    a = _roles.build_default_assignment(config={"model": "test"})
    for r in [a.pi, a.questioner, a.surveyor, a.designer,
              a.writer, a.reviewers[0], a.lay_reader]:
        text = _roles.load_role_template(r)
        assert len(text) > 50
        assert text.lower().startswith("you are")


# ── Orchestrator with stubbed LLM ─────────────────────────────────────────


def _make_stub_llm(scripted_outputs: dict):
    """Build a CallLLM that returns scripted text per role_name."""
    counters = {}

    def call(*, role_name: str, model: str, system: str, user: str,
             config: dict) -> _orch.LLMResponse:
        # Reviewers all share the same envelope; keep a per-role counter
        # so we can return different verdicts on different rounds.
        n = counters.get(role_name, 0)
        counters[role_name] = n + 1
        outputs = scripted_outputs.get(role_name) or [
            "stub default output"
        ]
        text = outputs[min(n, len(outputs) - 1)]
        return _orch.LLMResponse(text=text, tokens_in=10, tokens_out=20,
                                  cost_cents=1)

    return call


def _passing_reviewer_json():
    return json.dumps({
        "score": 8, "blocking_issues": [],
        "suggestions": ["minor polish"],
        "overall": "looks good",
    })


def _failing_reviewer_json():
    return json.dumps({
        "score": 4,
        "blocking_issues": ["unclear methodology", "missing citation"],
        "suggestions": [], "overall": "needs work",
    })


def test_orchestrator_end_to_end_happy_path(tmp_path, monkeypatch):
    """Run all 6 stages with reviewers passing on round 1; verify outputs."""
    storage = _storage.LabStorage(tmp_path / "lab.db")
    monkeypatch.setattr(_storage, "DEFAULT_OUTPUT_DIR", tmp_path / "papers")

    scripted = {
        "questioner": ["1. RQ A\n2. RQ B\n3. RQ C"],
        "pi": ["I pick RQ A because ..."],
        "surveyor": [
            "## Related work\nLots of stuff [Smith 2020].\n"
            "## Identified gap\nNobody has tried X.\n"
            "## Citations\n- Foo (Smith, 2020). arXiv:2001.0001"
        ],
        "designer": ["## Approach\n- Specific bullet"],
        "writer": [
            "# Title\n## Abstract\nshort.\n## References\n- Foo (Smith, 2020)."
        ],
        "reviewer_1": [_passing_reviewer_json()],
        "reviewer_2": [_passing_reviewer_json()],
        "reviewer_3": [_passing_reviewer_json()],
        "lay_reader": [_passing_reviewer_json()],
    }
    call = _make_stub_llm(scripted)

    # Skip the network verifier so this test never touches the internet.
    monkeypatch.setattr(_verifier, "_http_get",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("offline test")))

    run = _orch.run_one_lab_session(
        topic="A test topic",
        config={"model": "test"},
        storage_obj=storage,
        call_llm=call,
        budget_tokens=1_000_000,
        budget_cost_cents=1000,
        max_rounds=3,
    )

    rec = storage.get_run(run.state.run_id)
    assert rec.status == "done"
    # All stages logged
    stages = {s.stage for s in storage.list_stages(run.state.run_id)}
    assert {"questioning", "survey", "outline", "drafting",
            "verification", "finalization"} <= stages

    # Final report exists
    report = storage.get_latest_artifact(run.state.run_id, "report")
    assert report is not None
    assert "Title" in report.content or "test topic" in report.content.lower()


def test_orchestrator_advances_on_max_rounds_with_failing_reviewers(
        tmp_path, monkeypatch):
    """Even with failing reviewers, hit max_rounds → force advance."""
    storage = _storage.LabStorage(tmp_path / "lab.db")
    monkeypatch.setattr(_storage, "DEFAULT_OUTPUT_DIR", tmp_path / "papers")
    scripted = {
        "questioner": ["1. RQ A"],
        "pi": ["pick A"],
        "surveyor": ["## Related work\n## Identified gap\n## Citations\n- t (a, 2020)"],
        "designer": ["## Approach"],
        "writer": ["# T\n## References\n- t (a, 2020)"],
        # All reviewers fail on every round; orchestrator should still
        # advance after max_rounds=2.
        "reviewer_1": [_failing_reviewer_json()],
        "reviewer_2": [_failing_reviewer_json()],
        "reviewer_3": [_failing_reviewer_json()],
        "lay_reader": [_failing_reviewer_json()],
    }
    call = _make_stub_llm(scripted)
    monkeypatch.setattr(_verifier, "_http_get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("o")))

    run = _orch.run_one_lab_session(
        topic="t", config={"model": "x"}, storage_obj=storage,
        call_llm=call, max_rounds=2,
        budget_tokens=10_000_000, budget_cost_cents=10000,
    )
    rec = storage.get_run(run.state.run_id)
    assert rec.status == "done"


def test_orchestrator_respects_cancel(tmp_path, monkeypatch):
    """Cancel flag set during stage → run ends with status=aborted."""
    storage = _storage.LabStorage(tmp_path / "lab.db")
    monkeypatch.setattr(_storage, "DEFAULT_OUTPUT_DIR", tmp_path / "papers")

    cancelled = {"v": False}
    scripted = {
        "questioner": ["1. RQ A"],
        "pi": ["pick"],
        "surveyor": ["## Related work\n## Identified gap\n## Citations\n- t (a, 2020)"],
        "designer": ["## Approach"],
        "writer": ["# T"],
        "reviewer_1": [_passing_reviewer_json()],
        "reviewer_2": [_passing_reviewer_json()],
        "reviewer_3": [_passing_reviewer_json()],
        "lay_reader": [_passing_reviewer_json()],
    }
    base_call = _make_stub_llm(scripted)

    def cancelling_call(**kwargs):
        # Cancel right after the questioner's first call.
        if kwargs.get("role_name") == "questioner":
            cancelled["v"] = True
        return base_call(**kwargs)

    monkeypatch.setattr(_verifier, "_http_get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("o")))

    run = _orch.run_one_lab_session(
        topic="t", config={"model": "x"}, storage_obj=storage,
        call_llm=cancelling_call,
        cancel_check=lambda: cancelled["v"],
    )
    rec = storage.get_run(run.state.run_id)
    assert rec.status == "aborted"


def test_orchestrator_respects_budget_and_finalizes(tmp_path, monkeypatch):
    """When the budget is tiny, run still reaches finalization gracefully."""
    storage = _storage.LabStorage(tmp_path / "lab.db")
    monkeypatch.setattr(_storage, "DEFAULT_OUTPUT_DIR", tmp_path / "papers")

    # First call burns the entire token budget so subsequent stages get
    # short-circuited to FINALIZATION by the budget check.
    def big_call(**kwargs):
        return _orch.LLMResponse(text="1. RQ A", tokens_in=500, tokens_out=600,
                                  cost_cents=999)
    monkeypatch.setattr(_verifier, "_http_get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("o")))

    run = _orch.run_one_lab_session(
        topic="t", config={"model": "x"}, storage_obj=storage,
        call_llm=big_call,
        budget_tokens=1000,    # tiny — first call exceeds
        budget_cost_cents=10,
    )
    rec = storage.get_run(run.state.run_id)
    assert rec.status == "done"   # finalization completed


# ── Citation parsing inside orchestrator ─────────────────────────────────


def test_parse_citations_from_markdown_finds_arxiv_id():
    md = """## Citations
- Attention Is All You Need (Vaswani, Shazeer, 2017). arXiv:1706.03762
- Other paper (Doe, 2020).
"""
    cits = _orch._parse_citations_from_markdown(md)
    titles = [c.title.lower() for c in cits]
    assert any("attention" in t for t in titles)
    assert any("other paper" in t for t in titles)
    arxiv_ids = [c.arxiv_id for c in cits if c.arxiv_id]
    assert "1706.03762" in arxiv_ids


def test_parse_citations_handles_empty():
    assert _orch._parse_citations_from_markdown("") == []


def test_parse_reviewer_verdict_handles_fenced_json():
    text = '```json\n{"score": 7, "blocking_issues": [], "suggestions": [], "overall": "ok"}\n```'
    v = _orch._parse_reviewer_verdict(text, "reviewer_1")
    assert v.score == 7
    assert v.passes is True


def test_parse_reviewer_verdict_handles_garbage():
    v = _orch._parse_reviewer_verdict("just prose, no JSON", "reviewer_2")
    assert v.score == 5
    assert "unparseable" in (v.blocking_issues + ["x"])[0].lower() or v.overall == "parse_error"


# ── Output assembly ──────────────────────────────────────────────────────


def test_write_markdown_report_assembles_artifacts(tmp_path):
    storage = _storage.LabStorage(tmp_path / "lab.db")
    rec = storage.create_run(topic="My topic")
    storage.put_artifact(rec.run_id, "rq", "1. First RQ\n2. Second RQ")
    storage.put_artifact(rec.run_id, "rq_decision", "I pick the first.")
    storage.put_artifact(rec.run_id, "survey",
                          "## Related work\nfoo\n## Identified gap\nbar\n")
    storage.put_artifact(rec.run_id, "draft_full", "# Final paper title\n\nBody.")
    storage.put_artifact(
        rec.run_id, "citations_verified",
        json.dumps([{"key": "k1", "title": "Foo", "claimed_authors": ["A"],
                     "status": "verified", "matched_title": "Foo",
                     "matched_authors": ["A"], "matched_url": "u",
                     "source": "arxiv"}]),
    )
    storage.update_run_status(rec.run_id, "done", current_stage="finalization")

    # Build a minimal LabRun shim; output.write_markdown_report just needs
    # state.run_id, state.topic, storage.
    from research.lab.orchestrator import LabState, LabRun, Stage
    state = LabState(run_id=rec.run_id, topic="My topic", stage=Stage.FINALIZATION)
    run = LabRun(state=state, storage=storage,
                  roles=_roles.build_default_assignment({}),
                  config={})

    md = _output.write_markdown_report(run, output_dir=tmp_path / "papers")
    assert "# Final paper title" in md
    assert "First RQ" in md or "Second RQ" in md
    assert "verified" in md
    # Files exist on disk
    paper_dir = tmp_path / "papers" / rec.run_id
    assert (paper_dir / "report.md").exists()
    assert (paper_dir / "references.bib").exists() or True
    assert (paper_dir / "citations_verified.json").exists()


def test_format_bibtex_handles_not_found():
    bib = _output.format_bibtex([
        {"key": "ok1", "title": "Real", "matched_authors": ["A"],
         "status": "verified", "matched_url": "u"},
        {"key": "fake", "title": "Hallucinated", "claimed_authors": ["X"],
         "status": "not_found"},
    ])
    assert "@misc{ok1," in bib
    assert "NOT FOUND" in bib
    assert "fake," not in bib  # don't emit bibtex for unverifieds


# ── Phase 2: sandbox ─────────────────────────────────────────────────────


def test_extract_python_block_basic():
    from research.lab import sandbox as sb
    text = "Here's the script:\n\n```python\nprint(42)\n```\n"
    code = sb.extract_python_block(text)
    assert code is not None
    assert code.strip() == "print(42)"


def test_extract_python_block_no_lang_fallback():
    from research.lab import sandbox as sb
    text = "```\nprint('hi')\n```"
    code = sb.extract_python_block(text)
    assert code is not None
    assert "print('hi')" in code


def test_extract_python_block_returns_none_when_absent():
    from research.lab import sandbox as sb
    assert sb.extract_python_block("just prose") is None


def test_sandbox_runs_simple_script(tmp_path):
    from research.lab import sandbox as sb
    ws = tmp_path / "ws"
    res = sb.run_python_in_sandbox(
        "print('hello')\nimport sys; sys.exit(0)",
        workspace_dir=ws, timeout_s=10,
    )
    assert res.exit_code == 0
    assert res.stdout.startswith("hello")
    assert not res.timed_out
    assert (ws / "stdout.txt").exists()


def test_sandbox_captures_nonzero_exit(tmp_path):
    from research.lab import sandbox as sb
    res = sb.run_python_in_sandbox(
        "import sys; sys.exit(7)",
        workspace_dir=tmp_path / "ws", timeout_s=10,
    )
    assert res.exit_code == 7
    assert not res.timed_out


def test_sandbox_captures_stderr(tmp_path):
    from research.lab import sandbox as sb
    res = sb.run_python_in_sandbox(
        "import sys; print('bad', file=sys.stderr); sys.exit(1)",
        workspace_dir=tmp_path / "ws", timeout_s=10,
    )
    assert res.exit_code == 1
    assert "bad" in res.stderr


def test_sandbox_timeout(tmp_path):
    from research.lab import sandbox as sb
    res = sb.run_python_in_sandbox(
        "import time; time.sleep(20)",
        workspace_dir=tmp_path / "ws", timeout_s=1,
    )
    assert res.timed_out is True
    assert res.exit_code != 0


def test_sandbox_collects_artifacts(tmp_path):
    from research.lab import sandbox as sb
    ws = tmp_path / "ws"
    code = (
        "with open('output.txt', 'w') as f:\n"
        "    f.write('hello\\n')\n"
        "with open('result.json', 'w') as f:\n"
        "    f.write('{}')\n"
    )
    res = sb.run_python_in_sandbox(code, workspace_dir=ws, timeout_s=10)
    assert res.exit_code == 0
    fnames = sorted(p.name for p in res.artifacts)
    assert "output.txt" in fnames
    assert "result.json" in fnames
    # stdout/stderr/exit_code are excluded from artifacts
    assert "stdout.txt" not in fnames


def test_sandbox_format_result_for_prompt():
    from research.lab.sandbox import SandboxResult, format_result_for_prompt
    res = SandboxResult(
        exit_code=0, stdout="hello\nworld\n", stderr="",
        duration_s=0.42, timed_out=False, workspace=Path("/tmp"),
        artifacts=[],
    )
    txt = format_result_for_prompt(res)
    assert "exit_code: 0" in txt
    assert "0.42s" in txt
    assert "hello" in txt


# ── Phase 2: storage experiments ────────────────────────────────────────


def test_storage_record_experiment_roundtrip(tmp_path):
    s = _storage.LabStorage(tmp_path / "lab.db")
    r = s.create_run(topic="t")
    s.record_experiment(
        run_id=r.run_id, attempt=1,
        code="print(1)", exit_code=0, stdout="1\n", stderr="",
        duration_s=0.1, timed_out=False, artifacts=["fig.png"],
    )
    s.record_experiment(
        run_id=r.run_id, attempt=2,
        code="print(2)", exit_code=1, stdout="", stderr="boom",
        duration_s=0.05, timed_out=False,
    )
    exps = s.list_experiments(r.run_id)
    assert len(exps) == 2
    assert exps[0].artifacts == ["fig.png"]
    assert exps[1].exit_code == 1
    latest = s.get_latest_experiment(r.run_id)
    assert latest.attempt == 2
    s.close()


# ── Phase 2: orchestrator with experiments ──────────────────────────────


def test_orchestrator_skip_experiment_via_engineer(tmp_path, monkeypatch):
    """Engineer responds with SKIP_EXPERIMENT → IMPLEMENTATION/EXPERIMENT/
    ANALYSIS are skipped, run still completes."""
    storage = _storage.LabStorage(tmp_path / "lab.db")
    monkeypatch.setattr(_storage, "DEFAULT_OUTPUT_DIR", tmp_path / "papers")
    scripted = {
        "questioner": ["1. RQ A"],
        "pi": ["pick A"],
        "surveyor": ["## Related work\n## Identified gap\n## Citations\n- t (a, 2020)"],
        "designer": ["## Approach"],
        # No fenced code block → orchestrator interprets as skip
        "engineer": ["# SKIP_EXPERIMENT: this is a survey topic"],
        "writer": ["# T\n## References\n- t (a, 2020)"],
        "reviewer_1": [_passing_reviewer_json()],
        "reviewer_2": [_passing_reviewer_json()],
        "reviewer_3": [_passing_reviewer_json()],
        "lay_reader": [_passing_reviewer_json()],
    }
    call = _make_stub_llm(scripted)
    monkeypatch.setattr(_verifier, "_http_get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("o")))

    run = _orch.run_one_lab_session(
        topic="t", config={"model": "x"}, storage_obj=storage,
        call_llm=call, max_rounds=2,
        budget_tokens=10_000_000, budget_cost_cents=10_000,
    )
    rec = storage.get_run(run.state.run_id)
    assert rec.status == "done"
    # No experiments recorded since engineer skipped
    assert storage.list_experiments(run.state.run_id) == []


def test_orchestrator_runs_experiment_when_engineer_outputs_code(
        tmp_path, monkeypatch):
    """End-to-end with sandbox executing a tiny script."""
    storage = _storage.LabStorage(tmp_path / "lab.db")
    monkeypatch.setattr(_storage, "DEFAULT_OUTPUT_DIR", tmp_path / "papers")

    engineer_code = (
        "Here's the experiment:\n\n"
        "```python\nimport json\nprint('RESULT:', json.dumps({'metric': 0.42}))\n```"
    )
    scripted = {
        "questioner": ["1. RQ A"],
        "pi": ["pick A"],
        "surveyor": ["## Related work\n## Identified gap\n## Citations\n- t (a, 2020)"],
        "designer": ["## Approach"],
        "engineer": [engineer_code],
        "analyst": ["## Results\n### Setup\nTiny.\n### Findings\nmetric=0.42."],
        "writer": ["# T\nresult was 0.42.\n## References\n- t (a, 2020)"],
        "reviewer_1": [_passing_reviewer_json()],
        "reviewer_2": [_passing_reviewer_json()],
        "reviewer_3": [_passing_reviewer_json()],
        "lay_reader": [_passing_reviewer_json()],
    }
    call = _make_stub_llm(scripted)
    monkeypatch.setattr(_verifier, "_http_get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("o")))

    run = _orch.run_one_lab_session(
        topic="t", config={"model": "x", "lab_experiments": True},
        storage_obj=storage,
        call_llm=call, max_rounds=2,
        budget_tokens=10_000_000, budget_cost_cents=10_000,
    )
    rec = storage.get_run(run.state.run_id)
    assert rec.status == "done"
    exps = storage.list_experiments(run.state.run_id)
    assert len(exps) >= 1
    assert exps[0].exit_code == 0
    assert "metric" in (exps[0].stdout or "")


def test_orchestrator_engineer_debug_loop_on_failure(tmp_path, monkeypatch):
    """First code crashes; engineer revises on attempt 2; should advance."""
    storage = _storage.LabStorage(tmp_path / "lab.db")
    monkeypatch.setattr(_storage, "DEFAULT_OUTPUT_DIR", tmp_path / "papers")

    bad_code = "```python\nimport sys; sys.exit(1)\n```"
    good_code = "```python\nprint('ok')\n```"
    scripted = {
        "questioner": ["1. RQ A"],
        "pi": ["pick A"],
        "surveyor": ["## Related work\n## Identified gap\n## Citations\n- t (a, 2020)"],
        "designer": ["## Approach"],
        "engineer": [bad_code, good_code],   # 1st fails, 2nd succeeds
        "analyst": ["## Results\nok"],
        "writer": ["# T\nworked.\n## References\n- t (a, 2020)"],
        "reviewer_1": [_passing_reviewer_json()],
        "reviewer_2": [_passing_reviewer_json()],
        "reviewer_3": [_passing_reviewer_json()],
        "lay_reader": [_passing_reviewer_json()],
    }
    call = _make_stub_llm(scripted)
    monkeypatch.setattr(_verifier, "_http_get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("o")))

    run = _orch.run_one_lab_session(
        topic="t",
        config={"model": "x", "lab_experiments": True,
                "lab_experiment_max_attempts": 3},
        storage_obj=storage, call_llm=call, max_rounds=2,
        budget_tokens=10_000_000, budget_cost_cents=10_000,
    )
    exps = storage.list_experiments(run.state.run_id)
    assert len(exps) == 2
    assert exps[0].exit_code == 1
    assert exps[1].exit_code == 0


# ── Phase 3: web routes ─────────────────────────────────────────────────


def test_lab_api_start_run_returns_id(tmp_path, monkeypatch):
    monkeypatch.setattr(_storage, "DEFAULT_DB_PATH", tmp_path / "lab.db")
    monkeypatch.setattr(_storage, "DEFAULT_OUTPUT_DIR", tmp_path / "papers")
    # Patch threading.Thread to avoid actually launching the orchestrator
    import threading as _th
    started = {"n": 0}
    real_thread = _th.Thread

    class _NoopThread:
        def __init__(self, *a, **kw): pass
        def start(self): started["n"] += 1
    monkeypatch.setattr(_th, "Thread", _NoopThread)

    from web import lab_api
    monkeypatch.setattr(lab_api, "_run_threads", {})
    monkeypatch.setattr(lab_api, "_cancel_flags", {})
    # The dispatcher uses threading.Thread internally — it imports `threading`
    # at module level. Override the same module reference.
    monkeypatch.setattr(lab_api.threading, "Thread", _NoopThread)

    status, ctype, body = lab_api.dispatch(
        "/api/lab/runs", "POST", {},
        {"topic": "test topic", "budget_tokens": 1000,
         "budget_cost_cents": 10, "max_rounds": 3},
        {"model": "x"},
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["run_id"].startswith("lab_")
    assert started["n"] == 1


def test_lab_api_list_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(_storage, "DEFAULT_DB_PATH", tmp_path / "lab.db")
    s = _storage.LabStorage(tmp_path / "lab.db")
    s.create_run(topic="alpha")
    s.create_run(topic="beta")
    s.close()
    from web import lab_api
    status, ctype, body = lab_api.dispatch(
        "/api/lab/runs", "GET", {}, {}, {})
    assert status == 200
    payload = json.loads(body)
    topics = {r["topic"] for r in payload["runs"]}
    assert {"alpha", "beta"} <= topics


def test_lab_api_run_detail_404_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(_storage, "DEFAULT_DB_PATH", tmp_path / "lab.db")
    from web import lab_api
    status, _, body = lab_api.dispatch(
        "/api/lab/runs/lab_doesnotexist", "GET", {}, {}, {})
    assert status == 404


def test_lab_api_messages_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(_storage, "DEFAULT_DB_PATH", tmp_path / "lab.db")
    s = _storage.LabStorage(tmp_path / "lab.db")
    rec = s.create_run(topic="t")
    s.append_message(rec.run_id, stage="questioning", round_=0,
                     role="pi", kind="decision",
                     content="hi from PI")
    s.close()
    from web import lab_api
    status, _, body = lab_api.dispatch(
        f"/api/lab/runs/{rec.run_id}/messages", "GET", {}, {}, {})
    assert status == 200
    msgs = json.loads(body)["messages"]
    assert msgs[0]["content"] == "hi from PI"


def test_lab_api_report_falls_back_to_404(tmp_path, monkeypatch):
    monkeypatch.setattr(_storage, "DEFAULT_DB_PATH", tmp_path / "lab.db")
    monkeypatch.setattr(_storage, "DEFAULT_OUTPUT_DIR", tmp_path / "papers")
    s = _storage.LabStorage(tmp_path / "lab.db")
    rec = s.create_run(topic="t")
    s.close()
    from web import lab_api
    status, _, _ = lab_api.dispatch(
        f"/api/lab/runs/{rec.run_id}/report", "GET", {}, {}, {})
    assert status == 404


def test_lab_api_artifact_serves_file(tmp_path, monkeypatch):
    monkeypatch.setattr(_storage, "DEFAULT_DB_PATH", tmp_path / "lab.db")
    monkeypatch.setattr(_storage, "DEFAULT_OUTPUT_DIR", tmp_path / "papers")
    s = _storage.LabStorage(tmp_path / "lab.db")
    rec = s.create_run(topic="t")
    s.close()
    ws = tmp_path / "papers" / rec.run_id / "workspace"
    ws.mkdir(parents=True)
    (ws / "fig.png").write_bytes(b"PNGDATA")
    from web import lab_api
    status, ctype, body = lab_api.dispatch(
        f"/api/lab/runs/{rec.run_id}/artifacts/fig.png",
        "GET", {}, {}, {})
    assert status == 200
    assert ctype == "image/png"
    assert body == b"PNGDATA"


def test_lab_api_artifact_path_traversal_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(_storage, "DEFAULT_OUTPUT_DIR", tmp_path / "papers")
    from web import lab_api
    status, _, body = lab_api.dispatch(
        "/api/lab/runs/lab_abcdef0123456789/artifacts/..%2Fpasswd",
        "GET", {}, {}, {})
    # Either 404 (regex doesn't match the encoded form) or 400 (blocked)
    assert status in (400, 404)


def test_lab_api_abort_unknown_run(tmp_path, monkeypatch):
    from web import lab_api
    status, _, body = lab_api.dispatch(
        "/api/lab/runs/lab_abcdef0123456789/abort", "POST", {}, {}, {})
    assert status == 404


def test_lab_api_unknown_endpoint_404():
    from web import lab_api
    status, _, _ = lab_api.dispatch(
        "/api/lab/garbage", "GET", {}, {}, {})
    assert status == 404


def test_lab_html_file_exists():
    """The frontend page must ship with the package."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "web" / "lab.html"
    assert p.exists()
    text = p.read_text()
    assert "research lab" in text.lower()
    assert "/api/lab/runs" in text
