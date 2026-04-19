"""Tests for context_gc module."""
import pytest

from context_gc import (
    GCState, process_gc_call, apply_gc, _apply_snippet,
    _find_anchor_line, inject_notes, build_verbatim_audit_note,
    prepend_verbatim_audit,
)


class TestGCState:
    def test_defaults(self):
        gs = GCState()
        assert gs.trashed_ids == set()
        assert gs.snippets == {}
        assert gs.notes == {}


class TestProcessGCCall:
    def _make_config(self):
        return {"_gc_state": GCState()}

    def test_no_gc_state(self):
        result = process_gc_call({}, {})
        assert "Error" in result

    def test_trash(self):
        cfg = self._make_config()
        result = process_gc_call({"trash": ["r1", "r2"]}, cfg)
        assert "trashed 2 results" in result
        assert "r1" in cfg["_gc_state"].trashed_ids
        assert "r2" in cfg["_gc_state"].trashed_ids

    def test_notes(self):
        cfg = self._make_config()
        result = process_gc_call(
            {"notes": [{"name": "key", "content": "value"}]}, cfg
        )
        assert "1 notes saved" in result
        assert cfg["_gc_state"].notes["key"] == "value"

    def test_trash_notes(self):
        cfg = self._make_config()
        cfg["_gc_state"].notes["old"] = "data"
        result = process_gc_call({"trash_notes": ["old"]}, cfg)
        assert "1 notes removed" in result
        assert "old" not in cfg["_gc_state"].notes

    def test_keep_snippets(self):
        cfg = self._make_config()
        result = process_gc_call(
            {"keep_snippets": [{"id": "r1", "keep_after": "def main"}]}, cfg
        )
        assert "kept snippets for 1 results" in result
        assert "r1" in cfg["_gc_state"].snippets

    def test_snippet_ignored_if_trashed(self):
        cfg = self._make_config()
        cfg["_gc_state"].trashed_ids.add("r1")
        process_gc_call(
            {"keep_snippets": [{"id": "r1", "keep_after": "x"}]}, cfg
        )
        assert "r1" not in cfg["_gc_state"].snippets


class TestApplyGC:
    def test_no_changes(self):
        gs = GCState()
        msgs = [{"role": "user", "content": "hi"}]
        assert apply_gc(msgs, gs) is msgs

    def test_trash_tool_result(self):
        gs = GCState()
        gs.trashed_ids.add("tc1")
        msgs = [
            {"role": "tool", "tool_call_id": "tc1", "name": "Read", "content": "big data..."},
            {"role": "tool", "tool_call_id": "tc2", "name": "Grep", "content": "kept"},
        ]
        result = apply_gc(msgs, gs)
        assert "trashed by model" in result[0]["content"]
        assert result[1]["content"] == "kept"

    def test_snippet_applied(self):
        gs = GCState()
        gs.snippets["tc1"] = {"id": "tc1", "keep_after": "def main"}
        content = "import os\n\ndef main():\n    pass\n"
        msgs = [{"role": "tool", "tool_call_id": "tc1", "name": "Read", "content": content}]
        result = apply_gc(msgs, gs)
        assert "def main" in result[0]["content"]
        assert "import os" not in result[0]["content"]

    def test_non_tool_messages_pass_through(self):
        gs = GCState()
        gs.trashed_ids.add("x")
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = apply_gc(msgs, gs)
        assert len(result) == 2
        assert result[0]["content"] == "hello"


class TestApplySnippet:
    def test_keep_after(self):
        content = "line1\nline2\ndef main():\n    pass"
        result = _apply_snippet(content, {"keep_after": "def main"})
        assert "def main" in result
        assert "line1" not in result
        assert "2 lines trimmed" in result

    def test_keep_before(self):
        content = "line1\nline2\nclass Foo:\n    pass"
        result = _apply_snippet(content, {"keep_before": "class Foo"})
        assert "line1" in result
        assert "class Foo:\n    pass" not in result
        assert "2 lines trimmed" in result

    def test_keep_between(self):
        content = "a\nb\nSTART\nc\nd\nEND\ne\nf"
        result = _apply_snippet(content, {"keep_between": ["START", "END"]})
        assert "START" in result
        assert "END" in result
        assert "a\n" not in result

    def test_anchor_not_found(self):
        content = "some text"
        result = _apply_snippet(content, {"keep_after": "MISSING"})
        assert "GC warning" in result
        assert "some text" in result

    def test_empty_content(self):
        assert _apply_snippet("", {"keep_after": "x"}) == ""

    def test_keep_between_bad_anchors(self):
        result = _apply_snippet("text", {"keep_between": ["a"]})
        assert "needs exactly 2 anchors" in result


class TestFindAnchorLine:
    def test_found(self):
        assert _find_anchor_line(["a", "b", "c"], "b") == 1

    def test_not_found(self):
        assert _find_anchor_line(["a", "b"], "z") is None

    def test_start_from(self):
        assert _find_anchor_line(["a", "b", "a"], "a", start_from=1) == 2


class TestInjectNotes:
    def test_empty_notes(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert inject_notes(msgs, {}) is msgs

    def test_inject(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = inject_notes(msgs, {"key": "value"})
        assert "[Your working memory notes]" in result[0]["content"]
        assert "## key\nvalue" in result[0]["content"]
        assert "hello" in result[0]["content"]

    def test_injects_in_last_user_msg(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        result = inject_notes(msgs, {"n": "v"})
        assert "[Your working memory notes]" in result[2]["content"]
        assert result[0]["content"] == "first"


class TestVerbatimAudit:
    def test_empty(self):
        assert build_verbatim_audit_note([]) == ""

    def test_skips_trashed(self):
        msgs = [{"role": "tool", "tool_call_id": "t1", "name": "Read",
                 "content": "[Read result -- trashed by model]"}]
        assert build_verbatim_audit_note(msgs) == ""

    def test_skips_elided(self):
        msgs = [{"role": "tool", "tool_call_id": "t1", "name": "Read",
                 "content": '<tool_use_elided name="Read" brief="..."/>'}]
        assert build_verbatim_audit_note(msgs) == ""

    def test_includes_verbatim(self):
        msgs = [{"role": "tool", "tool_call_id": "r1", "name": "Read",
                 "content": "file content here"}]
        result = build_verbatim_audit_note(msgs)
        assert "r1 (Read)" in result
        assert "tk" in result

    def test_prepend(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "r1", "name": "Read", "content": "data"},
        ]
        result = prepend_verbatim_audit(msgs)
        assert "[Verbatim" in result[0]["content"]
