"""Unit tests for bridges/wechat_smart_reply.py + wechat_smart_reply_store.py.

We don't exercise the full WeChat poll loop here — the iLink protocol
needs a real account to drive end-to-end. These tests cover the logic
that's testable in isolation: gating rules, store backends, parsing,
candidate extraction, prompt construction, and the high-level entry
points with stubs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cheetahclaws.bridges import wechat_smart_reply as sr
from cheetahclaws.bridges import wechat_smart_reply_store as srs


# ── Identity heuristics ────────────────────────────────────────────────────

def test_is_filehelper():
    assert sr.is_filehelper("filehelper") is True
    assert sr.is_filehelper("wxid_alice") is False
    assert sr.is_filehelper("group@chatroom") is False


def test_is_group():
    assert sr.is_group("12345@chatroom") is True
    assert sr.is_group("wxid_alice") is False
    assert sr.is_group("filehelper") is False


# ── Gating: is_smart_reply_target ──────────────────────────────────────────

def test_target_off_by_default():
    assert sr.is_smart_reply_target("wxid_alice", {}) is False


def test_target_filehelper_excluded():
    cfg = {"wechat_smart_reply": True}
    assert sr.is_smart_reply_target("filehelper", cfg) is False


def test_target_group_excluded_by_default():
    cfg = {"wechat_smart_reply": True}
    assert sr.is_smart_reply_target("12345@chatroom", cfg) is False


def test_target_group_included_when_groups_on():
    cfg = {"wechat_smart_reply": True, "wechat_smart_reply_groups": True}
    assert sr.is_smart_reply_target("12345@chatroom", cfg, text="anything") is True


def test_target_whitelist_includes():
    cfg = {
        "wechat_smart_reply": True,
        "wechat_smart_reply_whitelist": ["wxid_alice", "wxid_bob"],
    }
    assert sr.is_smart_reply_target("wxid_alice", cfg) is True
    assert sr.is_smart_reply_target("wxid_carol", cfg) is False


def test_target_empty_whitelist_means_everyone():
    cfg = {"wechat_smart_reply": True}
    assert sr.is_smart_reply_target("wxid_random", cfg) is True


# ── Regression: self uid must always reach the agent ──────────────────────

def test_target_self_uid_excluded_even_when_in_whitelist():
    """The bot owner's own messages must never trigger smart-reply,
    even if their uid was (mistakenly) added to the whitelist. Without
    this, sending /draft from your phone would loop into smart-reply
    panels instead of reaching the agent."""
    my_uid = "o9cq80_xxx@im.wechat"
    cfg = {
        "wechat_smart_reply": True,
        "wechat_self_uid": my_uid,
        "wechat_smart_reply_whitelist": [my_uid, "wxid_alice"],
    }
    assert sr.is_smart_reply_target(my_uid, cfg) is False
    # Other whitelisted contacts still go through smart-reply.
    assert sr.is_smart_reply_target("wxid_alice", cfg) is True


def test_target_self_uid_unset_falls_through():
    """Empty wechat_self_uid disables the short-circuit (back-compat)."""
    cfg = {"wechat_smart_reply": True, "wechat_self_uid": ""}
    assert sr.is_smart_reply_target("any_uid", cfg) is True


# ── Group @-only rule ──────────────────────────────────────────────────────

def test_group_at_only_blocks_message_without_at():
    cfg = {
        "wechat_smart_reply": True,
        "wechat_smart_reply_groups": True,
        "wechat_smart_reply_groups_at_only": True,
        "wechat_self_nickname": "李明",
    }
    assert sr.is_smart_reply_target(
        "g@chatroom", cfg, text="今天天气不错") is False


def test_group_at_only_allows_message_with_at():
    cfg = {
        "wechat_smart_reply": True,
        "wechat_smart_reply_groups": True,
        "wechat_smart_reply_groups_at_only": True,
        "wechat_self_nickname": "李明",
    }
    assert sr.is_smart_reply_target(
        "g@chatroom", cfg, text="@李明 帮我看下这个") is True


def test_group_at_only_eos_boundary():
    cfg = {
        "wechat_smart_reply": True,
        "wechat_smart_reply_groups": True,
        "wechat_smart_reply_groups_at_only": True,
        "wechat_self_nickname": "李明",
    }
    # @nickname at end of string with no trailing char should still match
    assert sr.is_smart_reply_target("g@chatroom", cfg, text="hi @李明") is True


def test_group_at_only_substring_does_not_match():
    cfg = {
        "wechat_smart_reply": True,
        "wechat_smart_reply_groups": True,
        "wechat_smart_reply_groups_at_only": True,
        "wechat_self_nickname": "李",
    }
    # @李明 contains "李" but should not match nickname "李" because of
    # boundary rule (the next char after "李" is "明", not space/punct).
    assert sr.is_smart_reply_target("g@chatroom", cfg, text="@李明 hi") is False


def test_group_at_only_no_nickname_set_blocks_all():
    cfg = {
        "wechat_smart_reply": True,
        "wechat_smart_reply_groups": True,
        "wechat_smart_reply_groups_at_only": True,
        "wechat_self_nickname": "",
    }
    assert sr.is_smart_reply_target("g@chatroom", cfg, text="@anyone hi") is False


# ── Panel-ID generation ────────────────────────────────────────────────────

def test_n_to_id_first_few():
    assert srs.n_to_id(0) == "AA"
    assert srs.n_to_id(1) == "AB"
    assert srs.n_to_id(25) == "AZ"
    assert srs.n_to_id(26) == "BA"


def test_n_to_id_wraps():
    assert srs.n_to_id(26 * 26) == "AA"  # wraps after 676


# ── make_panel ─────────────────────────────────────────────────────────────

def _mk_panel(uid: str, *, panel_id: str = "AA",
              expires_in: float = 60) -> sr.PendingPanel:
    return sr.make_panel(uid, "Alice", "hi", ["a", "b", "c"],
                         panel_id=panel_id, timeout_s=expires_in)


# ── In-memory store ───────────────────────────────────────────────────────

def test_store_assign_next_id_monotonic():
    store = srs.InMemoryStore()
    assert store.assign_next_id() == "AA"
    assert store.assign_next_id() == "AB"
    assert store.assign_next_id() == "AC"


def test_store_put_and_take_active():
    store = srs.InMemoryStore()
    p = _mk_panel("wxid_alice")
    store.put(p)
    assert len(store) == 1
    got = store.take_active()
    assert got is not None
    assert got.target_uid == "wxid_alice"


def test_store_take_returns_most_recent():
    store = srs.InMemoryStore()
    p1 = _mk_panel("wxid_alice", panel_id="AA")
    time.sleep(0.01)
    p2 = _mk_panel("wxid_bob", panel_id="AB")
    store.put(p1)
    store.put(p2)
    got = store.take_active()
    assert got.target_uid == "wxid_bob"


def test_store_skips_expired():
    store = srs.InMemoryStore()
    store.put(_mk_panel("wxid_alice", expires_in=-1))  # already expired
    assert store.take_active() is None


def test_store_consume_by_uid():
    store = srs.InMemoryStore()
    store.put(_mk_panel("wxid_alice"))
    assert store.consume("wxid_alice") is not None
    assert store.consume("wxid_alice") is None  # idempotent
    assert len(store) == 0


def test_store_get_by_id():
    store = srs.InMemoryStore()
    store.put(_mk_panel("wxid_alice", panel_id="AA"))
    p = store.get_by_id("AA")
    assert p is not None
    assert p.target_uid == "wxid_alice"
    assert store.get_by_id("ZZ") is None


def test_store_consume_by_id():
    store = srs.InMemoryStore()
    store.put(_mk_panel("wxid_alice", panel_id="AA"))
    assert store.consume_by_id("AA") is not None
    assert store.consume_by_id("AA") is None  # already consumed


def test_store_list_active_returns_oldest_first():
    store = srs.InMemoryStore()
    store.put(_mk_panel("u1", panel_id="AA"))
    time.sleep(0.01)
    store.put(_mk_panel("u2", panel_id="AB"))
    time.sleep(0.01)
    store.put(_mk_panel("u3", panel_id="AC"))
    listed = store.list_active()
    assert [p.panel_id for p in listed] == ["AA", "AB", "AC"]


def test_store_sweep_expired_returns_count():
    store = srs.InMemoryStore()
    store.put(_mk_panel("u1", panel_id="AA", expires_in=-1))
    store.put(_mk_panel("u2", panel_id="AB", expires_in=60))
    swept = store.sweep_expired()
    assert swept == 1
    assert store.list_active()[0].panel_id == "AB"


def test_store_history_write_and_recent():
    store = srs.InMemoryStore()
    store.write_reply(to_uid="u1", to_label="A", text="hi", source="candidate_1")
    store.write_reply(to_uid="u2", to_label="B", text="bye", source="freeform")
    rows = store.recent_replies(n=10)
    assert len(rows) == 2
    assert rows[0].text == "bye"  # newest first


def test_store_history_excludes_uid():
    store = srs.InMemoryStore()
    store.write_reply(to_uid="u1", to_label="A", text="hi")
    store.write_reply(to_uid="u2", to_label="B", text="bye")
    rows = store.recent_replies(n=10, exclude_uid="u1")
    assert [r.to_uid for r in rows] == ["u2"]


def test_store_history_pruning():
    store = srs.InMemoryStore()
    old_ts = time.time() - (40 * 86400)
    store.write_reply(to_uid="u", to_label="A", text="ancient", ts=old_ts)
    store.write_reply(to_uid="u", to_label="A", text="recent")
    pruned = store.prune_history(older_than_days=30)
    assert pruned == 1
    assert [r.text for r in store.recent_replies(n=5)] == ["recent"]


# ── SQLite store ──────────────────────────────────────────────────────────


def test_sqlite_store_schema_initializes(tmp_path):
    store = srs.SqliteStore(tmp_path / "wx.db")
    try:
        # Schema should be created idempotently — second open works fine.
        store2 = srs.SqliteStore(tmp_path / "wx.db")
        store2.stop()
    finally:
        store.stop()


def test_sqlite_store_panel_roundtrip(tmp_path):
    store = srs.SqliteStore(tmp_path / "wx.db")
    try:
        pid = store.assign_next_id()
        p = sr.make_panel("u1", "Alice", "hi", ["A", "B", "C"], panel_id=pid)
        store.put(p)
        got = store.get_by_id(pid)
        assert got is not None
        assert got.candidates == ["A", "B", "C"]
        assert got.target_label == "Alice"
    finally:
        store.stop()


def test_sqlite_store_persists_across_reopen(tmp_path):
    db = tmp_path / "wx.db"
    store1 = srs.SqliteStore(db)
    pid = store1.assign_next_id()
    store1.put(sr.make_panel("u1", "Alice", "hi", ["a", "b", "c"], panel_id=pid))
    store1.stop()

    store2 = srs.SqliteStore(db)
    try:
        assert store2.get_by_id(pid) is not None
        # ID counter persists too
        assert store2.assign_next_id() == "AB"
    finally:
        store2.stop()


def test_sqlite_store_history_persists(tmp_path):
    db = tmp_path / "wx.db"
    s1 = srs.SqliteStore(db)
    s1.write_reply(to_uid="u1", to_label="Alice", text="好的", source="candidate_1")
    s1.stop()
    s2 = srs.SqliteStore(db)
    try:
        rows = s2.recent_replies(n=5)
        assert len(rows) == 1
        assert rows[0].text == "好的"
    finally:
        s2.stop()


def test_make_store_falls_back_to_memory_when_sqlite_blocked(tmp_path,
                                                               monkeypatch):
    import sqlite3 as _sqlite3
    real_connect = _sqlite3.connect

    def boom(*args, **kwargs):
        raise _sqlite3.OperationalError("simulated")

    monkeypatch.setattr(_sqlite3, "connect", boom)
    store = srs.make_store(db_path=tmp_path / "wx.db")
    monkeypatch.setattr(_sqlite3, "connect", real_connect)
    assert isinstance(store, srs.InMemoryStore)


# ── ParsedAction parsing ─────────────────────────────────────────────────


def test_parse_no_panel_returns_noop():
    store = srs.InMemoryStore()
    assert sr.parse_filehelper_input("1", store).kind == "noop"


def test_parse_queue_command():
    store = srs.InMemoryStore()
    assert sr.parse_filehelper_input("q", store).kind == "list"
    assert sr.parse_filehelper_input("queue", store).kind == "list"
    assert sr.parse_filehelper_input("队列", store).kind == "list"


def test_parse_numeric_choice_uses_latest_active():
    store = srs.InMemoryStore()
    pid = store.assign_next_id()
    store.put(sr.make_panel("u1", "Alice", "hi",
                            ["A", "B", "C"], panel_id=pid))
    a = sr.parse_filehelper_input("2", store)
    assert a.kind == "send"
    assert a.text == "B"
    assert a.panel_id == pid


def test_parse_skip_uses_latest_active():
    store = srs.InMemoryStore()
    pid = store.assign_next_id()
    store.put(sr.make_panel("u1", "Alice", "hi",
                            ["A", "B", "C"], panel_id=pid))
    for tok in ("x", "X", "skip", "/skip", "跳过"):
        a = sr.parse_filehelper_input(tok, store)
        assert a.kind == "skip"
        assert a.panel_id == pid


def test_parse_freeform_uses_latest_active():
    store = srs.InMemoryStore()
    pid = store.assign_next_id()
    store.put(sr.make_panel("u1", "Alice", "hi",
                            ["A", "B", "C"], panel_id=pid))
    a = sr.parse_filehelper_input("我自己写的", store)
    assert a.kind == "send"
    assert a.text == "我自己写的"


def test_parse_explicit_panel_id_addressing():
    store = srs.InMemoryStore()
    p1 = store.assign_next_id()  # AA
    p2 = store.assign_next_id()  # AB
    store.put(sr.make_panel("u1", "Alice", "hi", ["A", "B", "C"], panel_id=p1))
    store.put(sr.make_panel("u2", "Bob",   "hi", ["X", "Y", "Z"], panel_id=p2))
    # latest is u2 (AB); but explicit AA addresses Alice's panel
    a = sr.parse_filehelper_input("AA 1", store)
    assert a.kind == "send"
    assert a.panel_id == "AA"
    assert a.text == "A"


def test_parse_panel_id_alone_lists_panel():
    store = srs.InMemoryStore()
    p1 = store.assign_next_id()
    store.put(sr.make_panel("u1", "Alice", "hi", ["A", "B", "C"], panel_id=p1))
    a = sr.parse_filehelper_input("AA", store)
    assert a.kind == "list"
    assert a.panel_id == "AA"


def test_parse_unknown_panel_id_returns_noop():
    store = srs.InMemoryStore()
    p1 = store.assign_next_id()
    store.put(sr.make_panel("u1", "Alice", "hi", ["A", "B", "C"], panel_id=p1))
    a = sr.parse_filehelper_input("ZZ 1", store)
    # ZZ doesn't exist → _classify_choice with panel=None → noop, panel_id="ZZ"
    assert a.kind == "noop"
    assert a.panel_id == "ZZ"


# ── Candidate extraction ──────────────────────────────────────────────────

def test_parse_candidates_clean_format():
    text = "1. 好的\n2. 周末出差\n3. 在忙稍后"
    out = sr._parse_candidates(text)
    assert out == ["好的", "周末出差", "在忙稍后"]


def test_parse_candidates_strips_trailing_punct():
    text = "1. 好的。\n2. 不行！\n3. 在忙呢…"
    out = sr._parse_candidates(text)
    assert out[0] == "好的"
    assert out[1] == "不行"


def test_parse_candidates_handles_chinese_dot():
    text = "1、好\n2、行\n3、不"
    out = sr._parse_candidates(text)
    assert out == ["好", "行", "不"]


def test_parse_candidates_caps_at_three():
    text = "1. a\n2. b\n3. c\n4. d\n5. e"
    out = sr._parse_candidates(text)
    assert out == ["a", "b", "c"]


def test_parse_candidates_empty_on_garbage():
    assert sr._parse_candidates("just some prose, no numbered list") == []


# ── Prompt construction (style + contact context) ─────────────────────────

def test_build_prompt_without_extras():
    prompt = sr._build_prompt("hi", "Alice", contact=None, history=[])
    assert "hi" in prompt
    assert "Alice" in prompt
    assert "关于发件人" not in prompt
    assert "用户最近发出" not in prompt


def test_build_prompt_includes_contact_relationship():
    contact = sr.Contact(uid="u", label="Alice", relationship="close friend",
                         notes="她在找工作")
    prompt = sr._build_prompt("hi", "Alice", contact=contact, history=[])
    assert "close friend" in prompt
    assert "她在找工作" in prompt


def test_build_prompt_includes_history_examples():
    history = [
        sr.ReplyHistoryEntry(ts=time.time(), to_uid="u1", to_label="A",
                              text="哈哈好的", source="candidate_1"),
        sr.ReplyHistoryEntry(ts=time.time(), to_uid="u2", to_label="B",
                              text="晚点回你", source="freeform"),
    ]
    prompt = sr._build_prompt("hi", "Alice", contact=None, history=history)
    assert "哈哈好的" in prompt
    assert "晚点回你" in prompt
    assert "用户最近发出" in prompt


def test_build_prompt_history_capped_at_ten():
    history = [
        sr.ReplyHistoryEntry(ts=time.time(), to_uid="u", to_label=None,
                              text=f"reply {i}", source=None)
        for i in range(50)
    ]
    prompt = sr._build_prompt("hi", "Alice", contact=None, history=history)
    assert "reply 0" in prompt
    # 11+ should not appear (we cap at 10)
    assert "reply 15" not in prompt


# ── generate_candidates with stub ──────────────────────────────────────────

class _TextChunk:
    def __init__(self, text: str):
        self.text = text


def _stub_stream_3(**kwargs):
    yield _TextChunk("1. 行\n2. 在忙\n3. 晚点回")


def _stub_stream_2(**kwargs):
    yield _TextChunk("1. 好\n2. 在忙")


def _stub_stream_garbage(**kwargs):
    yield _TextChunk("This is not a list.")


def _stub_stream_raises(**kwargs):
    raise RuntimeError("boom")
    yield  # pragma: no cover


def test_generate_candidates_happy_path():
    out = sr.generate_candidates(
        "周末有空吗", "张三", {"auxiliary_model": "test"},
        _stream_fn=_stub_stream_3,
    )
    assert out == ["行", "在忙", "晚点回"]


def test_generate_candidates_partial_returned_as_is():
    out = sr.generate_candidates(
        "x", "Y", {"auxiliary_model": "test"}, _stream_fn=_stub_stream_2,
    )
    assert out == ["好", "在忙"]


def test_generate_candidates_returns_empty_on_garbage():
    out = sr.generate_candidates(
        "x", "Y", {"auxiliary_model": "test"}, _stream_fn=_stub_stream_garbage,
    )
    assert out == []


def test_generate_candidates_returns_empty_on_exception():
    out = sr.generate_candidates(
        "x", "Y", {"auxiliary_model": "test"}, _stream_fn=_stub_stream_raises,
    )
    assert out == []


def test_generate_candidates_threads_contact_into_prompt():
    captured = {}

    def capture_stream(**kwargs):
        captured["msg"] = kwargs["messages"][0]["content"]
        yield _TextChunk("1. ok\n2. fine\n3. sure")

    contact = sr.Contact(uid="u", label="Alice",
                         relationship="ex-coworker", notes="formal tone")
    sr.generate_candidates(
        "hi", "Alice", {"auxiliary_model": "test"},
        contact=contact, _stream_fn=capture_stream,
    )
    assert "ex-coworker" in captured["msg"]
    assert "formal tone" in captured["msg"]


# ── format_panel + format_queue ───────────────────────────────────────────

def test_format_panel_includes_id_label_message():
    p = sr.make_panel("wxid_alice", "Alice", "你好",
                      ["a", "b", "c"], panel_id="AA")
    out = sr.format_panel(p)
    assert "[AA]" in out
    assert "Alice" in out
    assert "你好" in out
    assert "[1] a" in out
    assert "q 看队列" in out


def test_format_queue_empty():
    out = sr.format_queue([])
    assert "没有" in out


def test_format_queue_lists_with_ids():
    p1 = sr.make_panel("u1", "Alice", "msg1", ["a", "b", "c"], panel_id="AA")
    p2 = sr.make_panel("u2", "Bob",   "msg2", ["x", "y", "z"], panel_id="AB")
    out = sr.format_queue([p1, p2])
    assert "[AA]" in out
    assert "[AB]" in out
    assert "Alice" in out
    assert "Bob" in out


def test_format_panel_truncates_long_message():
    long_msg = "x" * 300
    p = sr.make_panel("u", "U", long_msg, ["a", "b", "c"], panel_id="AA")
    out = sr.format_panel(p)
    assert "…" in out


# ── trigger_smart_reply (end-to-end with stubs) ──────────────────────────

def test_trigger_happy_path_with_in_memory_store():
    sent: list = []
    store = srs.InMemoryStore()
    cfg = {"auxiliary_model": "test"}
    ok = sr.trigger_smart_reply(
        target_uid="wxid_alice", target_label="Alice",
        message="周末有空", store=store, config=cfg,
        send_to_filehelper=lambda txt: sent.append(txt),
        generate_fn=lambda *a, **k: ["行", "忙", "晚点回"],
    )
    assert ok is True
    assert len(sent) == 1
    assert "[AA]" in sent[0]   # first panel gets first ID
    assert "Alice" in sent[0]
    assert len(store) == 1


def test_trigger_pads_short_list_to_three():
    store = srs.InMemoryStore()
    sent: list = []
    sr.trigger_smart_reply(
        target_uid="u", target_label="U",
        message="x", store=store, config={},
        send_to_filehelper=lambda txt: sent.append(txt),
        generate_fn=lambda *a, **k: ["哈哈"],
    )
    panel = store.take_active()
    assert len(panel.candidates) == 3


def test_trigger_returns_false_on_empty_generation():
    store = srs.InMemoryStore()
    sent: list = []
    ok = sr.trigger_smart_reply(
        target_uid="u", target_label="U", message="x",
        store=store, config={},
        send_to_filehelper=lambda txt: sent.append(txt),
        generate_fn=lambda *a, **k: [],
    )
    assert ok is False
    assert sent == []
    assert len(store) == 0


def test_trigger_passes_contact_into_generator(tmp_path):
    captured = {}

    def fake_gen(message, label, config, *, contact=None, history=None):
        captured["contact"] = contact
        captured["history"] = history
        return ["a", "b", "c"]

    contacts_path = tmp_path / "wx_contacts.json"
    contacts_path.write_text(json.dumps({
        "u": {"label": "Alice", "relationship": "friend"}
    }))
    contacts = sr.ContactsStore(path=contacts_path)

    store = srs.InMemoryStore()
    sr.trigger_smart_reply(
        target_uid="u", target_label="U", message="x",
        store=store, config={},
        send_to_filehelper=lambda txt: None,
        contacts=contacts,
        generate_fn=fake_gen,
    )
    assert captured["contact"] is not None
    assert captured["contact"].relationship == "friend"


# ── handle_filehelper_message ─────────────────────────────────────────────


def test_handle_no_active_panel_falls_through():
    store = srs.InMemoryStore()
    consumed = sr.handle_filehelper_message(
        "anything", store,
        send_to_target=lambda u, t: None,
        send_to_filehelper=lambda t: None,
    )
    assert consumed is False


def test_handle_numeric_sends_candidate_and_records_history():
    store = srs.InMemoryStore()
    pid = store.assign_next_id()
    store.put(sr.make_panel("wxid_alice", "Alice", "你好",
                            ["A", "B", "C"], panel_id=pid))
    sent_target: list = []
    sent_fh: list = []
    consumed = sr.handle_filehelper_message(
        "2", store,
        send_to_target=lambda u, t: sent_target.append((u, t)),
        send_to_filehelper=lambda t: sent_fh.append(t),
    )
    assert consumed is True
    assert sent_target == [("wxid_alice", "B")]
    history = store.recent_replies(n=5)
    assert len(history) == 1
    assert history[0].text == "B"
    assert history[0].source == "candidate_2"


def test_handle_skip_drops_panel_no_history():
    store = srs.InMemoryStore()
    pid = store.assign_next_id()
    store.put(sr.make_panel("wxid_alice", "Alice", "你好",
                            ["A", "B", "C"], panel_id=pid))
    sent_target: list = []
    consumed = sr.handle_filehelper_message(
        "x", store,
        send_to_target=lambda u, t: sent_target.append((u, t)),
        send_to_filehelper=lambda t: None,
    )
    assert consumed is True
    assert sent_target == []
    assert store.recent_replies(n=5) == []  # skip ≠ send → no history row


def test_handle_freeform_records_as_freeform():
    store = srs.InMemoryStore()
    pid = store.assign_next_id()
    store.put(sr.make_panel("u", "U", "hi", ["A", "B", "C"], panel_id=pid))
    sr.handle_filehelper_message(
        "我自己写的", store,
        send_to_target=lambda u, t: None,
        send_to_filehelper=lambda t: None,
    )
    history = store.recent_replies(n=5)
    assert history[0].source == "freeform"
    assert history[0].text == "我自己写的"


def test_handle_queue_command_lists_pending():
    store = srs.InMemoryStore()
    p1 = store.assign_next_id()
    p2 = store.assign_next_id()
    store.put(sr.make_panel("u1", "Alice", "hi", ["a", "b", "c"], panel_id=p1))
    store.put(sr.make_panel("u2", "Bob",   "hi", ["x", "y", "z"], panel_id=p2))
    sent_fh: list = []
    consumed = sr.handle_filehelper_message(
        "q", store,
        send_to_target=lambda u, t: None,
        send_to_filehelper=lambda t: sent_fh.append(t),
    )
    assert consumed is True
    assert any("Alice" in s and "Bob" in s for s in sent_fh)


def test_handle_explicit_panel_id_addressing():
    store = srs.InMemoryStore()
    p1 = store.assign_next_id()  # AA
    p2 = store.assign_next_id()  # AB
    store.put(sr.make_panel("u1", "Alice", "hi", ["A", "B", "C"], panel_id=p1))
    store.put(sr.make_panel("u2", "Bob",   "hi", ["X", "Y", "Z"], panel_id=p2))
    sent_target: list = []
    sr.handle_filehelper_message(
        "AA 1", store,
        send_to_target=lambda u, t: sent_target.append((u, t)),
        send_to_filehelper=lambda t: None,
    )
    # AA = Alice's panel; "1" = first candidate "A"
    assert sent_target == [("u1", "A")]
    # Bob's panel still pending
    assert store.get_by_id("AB") is not None


def test_handle_unknown_panel_id_returns_warning():
    store = srs.InMemoryStore()
    pid = store.assign_next_id()
    store.put(sr.make_panel("u1", "Alice", "hi", ["A", "B", "C"], panel_id=pid))
    sent_fh: list = []
    consumed = sr.handle_filehelper_message(
        "ZZ 1", store,
        send_to_target=lambda u, t: None,
        send_to_filehelper=lambda t: sent_fh.append(t),
    )
    # ZZ doesn't exist; parse_filehelper_input returns kind=noop with panel_id="ZZ"
    # but handle_filehelper_message should fall through (no known panel).
    assert consumed is False


# ── ContactsStore ─────────────────────────────────────────────────────────


def test_contacts_missing_file_returns_none(tmp_path):
    store = sr.ContactsStore(path=tmp_path / "nonexistent.json")
    assert store.get("anyone") is None
    assert store.all() == {}


def test_contacts_set_and_get_roundtrip(tmp_path):
    p = tmp_path / "wx_contacts.json"
    store = sr.ContactsStore(path=p)
    store.set(sr.Contact(uid="wxid_alice", label="Alice (friend)",
                         relationship="close friend", notes="loves coffee"))
    # File written
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["wxid_alice"]["relationship"] == "close friend"
    # Round-trip via fresh store
    fresh = sr.ContactsStore(path=p)
    got = fresh.get("wxid_alice")
    assert got is not None
    assert got.notes == "loves coffee"


def test_contacts_delete(tmp_path):
    p = tmp_path / "wx_contacts.json"
    store = sr.ContactsStore(path=p)
    store.set(sr.Contact(uid="u", label="L"))
    assert store.delete("u") is True
    assert store.delete("u") is False  # idempotent
    assert store.get("u") is None


def test_contacts_corrupt_file_returns_empty(tmp_path):
    p = tmp_path / "wx_contacts.json"
    p.write_text("not valid json")
    store = sr.ContactsStore(path=p)
    assert store.all() == {}
    assert store.get("any") is None


def test_contacts_reload_on_mtime_change(tmp_path):
    p = tmp_path / "wx_contacts.json"
    p.write_text(json.dumps({"u": {"label": "First"}}))
    store = sr.ContactsStore(path=p)
    assert store.get("u").label == "First"

    # Edit file out-of-band then reload
    time.sleep(0.05)
    p.write_text(json.dumps({"u": {"label": "Second"}}))
    # Force mtime change (some filesystems have low resolution)
    new_mtime = p.stat().st_mtime + 1
    os.utime(p, (new_mtime, new_mtime))
    assert store.get("u").label == "Second"


# ── Config defaults ───────────────────────────────────────────────────────


def test_config_defaults_present():
    from cheetahclaws.config import DEFAULTS
    assert DEFAULTS["wechat_smart_reply"] is False
    assert DEFAULTS["wechat_smart_reply_whitelist"] == []
    assert DEFAULTS["wechat_smart_reply_groups"] is False
    assert DEFAULTS["wechat_smart_reply_groups_at_only"] is False
    assert DEFAULTS["wechat_smart_reply_timeout_s"] == 300
    assert DEFAULTS["wechat_self_nickname"] == ""
