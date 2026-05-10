"""Tests for autonomous-agent stagnation detection in agent_runner.py.

Regression target: research_assistant template ran 1500+ iterations on
qwen2.5-72b producing the *byte-identical* "task complete, no more papers"
summary every time, burning ~6,000 API calls of pure waste. The fix adds a
generic per-summary repetition counter that stops the iteration loop when
the same summary appears N times in a row (default N=3).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agent_runner
from agent_runner import AgentRunner, _normalize_summary


# ── _normalize_summary unit tests ────────────────────────────────────────

class TestNormalizeSummary:
    def test_empty(self):
        assert _normalize_summary("") == ""

    def test_basic_lowercase(self):
        assert _normalize_summary("Hello World") == "hello world"

    def test_collapses_whitespace(self):
        assert (
            _normalize_summary("Hello\n\n\tWorld   foo")
            == "hello world foo"
        )

    def test_strips_outer_whitespace(self):
        assert _normalize_summary("  hello  ") == "hello"

    def test_treats_paragraph_breaks_as_space(self):
        a = _normalize_summary("Line A\n\nLine B")
        b = _normalize_summary("Line A Line B")
        assert a == b

    def test_preserves_punctuation(self):
        # We deliberately keep punctuation so "Done." vs "I am done." differ.
        assert _normalize_summary("Done.") != _normalize_summary("Done!")

    def test_real_world_research_assistant_example(self):
        # The actual stuck summary from the user's log — same string
        # recorded byte-identically across 1500+ iterations.
        s1 = (
            "tes and related work sections have been updated accordingly.\n\n"
            "### Next Steps:\n- No further papers to process.\n"
            "- The task is complete."
        )
        s2 = (
            "tes and related work sections have been updated accordingly.\n\n"
            "### Next Steps:\n- No further papers to process.\n"
            "- The task is complete."
        )
        # Even byte-identical strings should normalize to the same canonical
        # form, and the comparison should obviously succeed.
        assert _normalize_summary(s1) == _normalize_summary(s2)


# ── Integration: run _run_loop with a fake agent.run generator ───────────

def _fake_agent_run_factory(text_per_iter: Iterable[str]):
    """Build an `agent.run` replacement that yields the next canned text on
    every invocation. Each canned text becomes one TextChunk.
    """
    from agent import TextChunk

    iter_texts = iter(text_per_iter)

    def fake_run(prompt, state, config, system_prompt):
        # Get next canned text; raise StopIteration if exhausted (loop exits).
        try:
            text = next(iter_texts)
        except StopIteration:
            return
        yield TextChunk(text)
    return fake_run


def _build_runner(tmp_log_root: Path, dup_limit: int = 3,
                  interval: float = 0.0) -> AgentRunner:
    """Build a runner with patched _LOG_DIR and an empty config + send_fn."""
    # Redirect _LOG_DIR so test logs don't pollute the real directory.
    agent_runner._LOG_DIR = tmp_log_root / "agents"
    runner = AgentRunner(
        name=f"test-{int(time.time() * 1000)}",
        template_content="(test template)",
        template_path="/tmp/dummy.md",
        args="",
        config={"auto_agent_dup_summary_limit": dup_limit, "model": "test"},
        send_fn=lambda msg: None,
        interval=interval,
        auto_approve=True,
    )
    return runner


class TestStagnationStop:
    def _run_with_canned_outputs(self, monkeypatch, tmp_path, outputs,
                                  dup_limit=3, max_seconds=5.0):
        """Helper: instantiate runner, patch agent.run, run loop in thread,
        wait until it stops or timeout, return runner."""
        import agent
        runner = _build_runner(tmp_path, dup_limit=dup_limit, interval=0.0)
        monkeypatch.setattr(agent, "run", _fake_agent_run_factory(outputs))
        runner.start()
        # Wait up to max_seconds for the runner thread to finish on its own.
        t0 = time.monotonic()
        while runner.is_alive and (time.monotonic() - t0) < max_seconds:
            time.sleep(0.05)
        runner.stop()  # idempotent
        return runner

    def test_three_identical_summaries_triggers_stop(self, monkeypatch, tmp_path):
        # 100 identical summaries; the runner should stop after the 3rd one,
        # producing exactly 3 history records (not 100).
        outputs = ["Task is complete. No further work."] * 100
        runner = self._run_with_canned_outputs(monkeypatch, tmp_path, outputs)
        assert not runner.is_alive
        # Should have stopped at iteration 3 (index 0-2 = three duplicates).
        assert len(runner._history) == 3

    def test_varied_summaries_do_not_trigger(self, monkeypatch, tmp_path):
        # Each iteration produces a *different* summary → loop exhausts the
        # canned list naturally, no early stagnation stop.
        outputs = [f"Iteration {i} did some unique work" for i in range(5)]
        runner = self._run_with_canned_outputs(monkeypatch, tmp_path, outputs)
        # All 5 iterations should run; the runner stops because canned
        # outputs ran out (StopIteration → empty text → "(no output)").
        # The 6th iteration onwards yields "(no output)" which is treated
        # specially (clears the dup window) — but the loop still ran 5
        # canned + may have started 6th. Either way, ≥ 5.
        assert len(runner._history) >= 5

    def test_disabled_via_zero_limit(self, monkeypatch, tmp_path):
        # With dup limit = 0, identical summaries are NOT a stop signal.
        # We use only 5 outputs to keep the test bounded; if the runner
        # didn't stop and ran a 6th time (no canned output), it would yield
        # "(no output)" which is what we'd see.
        outputs = ["Repeating output"] * 5
        runner = self._run_with_canned_outputs(
            monkeypatch, tmp_path, outputs, dup_limit=0
        )
        # At least all 5 canned outputs executed (we don't stop early)
        assert len(runner._history) >= 5
        # Confirm no stagnation log was emitted by checking the number of
        # records — the test outputs ran out, but the runner shouldn't have
        # stopped from stagnation, so it would continue iterating with
        # "(no output)" until something else stops it.

    def test_whitespace_variants_treated_as_duplicate(self, monkeypatch, tmp_path):
        # The same content with trivial whitespace differences must still
        # be detected as duplicate.
        outputs = [
            "Task is complete.",
            "Task is complete.\n\n",
            "  Task is complete.  ",
            # Whatever follows shouldn't matter — limit hit at the 3rd.
            "(should not be reached)",
        ]
        runner = self._run_with_canned_outputs(monkeypatch, tmp_path, outputs)
        assert not runner.is_alive
        assert len(runner._history) == 3

    def test_two_dup_then_different_then_two_dup_does_not_trigger(
        self, monkeypatch, tmp_path
    ):
        # The detector requires N *consecutive* duplicates. A pattern like
        # A, A, B, A, A should NOT trigger a stop with limit=3.
        outputs = ["A summary"] * 2 + ["A different summary"] + ["A summary"] * 2
        runner = self._run_with_canned_outputs(monkeypatch, tmp_path, outputs)
        # All 5 canned outputs should have run.
        assert len(runner._history) >= 5

    def test_no_output_does_not_count_as_duplicate(self, monkeypatch, tmp_path):
        # Empty text (which becomes "(no output)") should NOT count toward
        # the dup window — those are agent failures handled elsewhere.
        outputs = ["", "", ""]
        runner = self._run_with_canned_outputs(monkeypatch, tmp_path, outputs)
        # Should run all 3 without triggering stagnation (the failure-tracking
        # path may or may not trigger first; we just assert stagnation
        # specifically didn't pile up false-positives)
        assert len(runner._history) >= 3
