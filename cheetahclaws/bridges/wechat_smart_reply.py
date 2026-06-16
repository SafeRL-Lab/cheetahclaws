"""WeChat smart-reply panel — AI drafts, user approves on phone via 文件传输助手.

Flow:
  1. Inbound message from a whitelisted contact arrives.
  2. Generate 3 candidate replies via the auxiliary cheap model, conditioned on:
       * the contact's relationship/notes from ~/.cheetahclaws/wx_contacts.json
       * the user's recent confirmed replies (style mimicking)
  3. Send a panel to filehelper, tagged with a 2-letter ID:
        💬 [AA] 张三 → "周末有空吗"
        [1] 有的，周六下午行
        [2] 周末出差，下周可以吗
        [3] 在忙，晚点回你
        回 1/2/3 发送 · 直接打字自定义 · x 跳过 · q 看队列
  4. Filehelper input is interpreted against the active queue:
        - "1" / "2" / "3"           → send candidate from latest active panel
        - "x"                       → skip latest active panel
        - "<freeform>"              → send freeform reply for latest active panel
        - "AA 1" / "AA x" / "AA hi" → address panel by ID explicitly
        - "q" / "queue"             → list pending panels
  5. Confirmed sends are appended to wx_reply_history and feed style mimicking.
  6. Panels expire after wechat_smart_reply_timeout_s (default 5 min).
     The store's janitor sweeps expired rows.

Group rules:
  * Group chats off by default.
  * `wechat_smart_reply_groups: true` enables them.
  * `wechat_smart_reply_groups_at_only: true` further restricts groups to
    messages that contain @<wechat_self_nickname>.

Storage:
  * Panels + reply history persist in SQLite (~/.cheetahclaws/wx_smart_reply.db).
  * Contacts persist in JSON (~/.cheetahclaws/wx_contacts.json).
  * SQLite init failure auto-falls-back to in-memory; nothing crashes.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .wechat_smart_reply_store import (
    Contact,
    ContactsStore,
    DEFAULT_TIMEOUT_S,
    InMemoryStore,
    PendingPanel,
    ReplyHistoryEntry,
    SqliteStore,
    make_store,
    n_to_id,
)

# Re-export so callers (including tests) get a single import surface.
__all__ = [
    "Contact", "ContactsStore", "PendingPanel", "ReplyHistoryEntry",
    "InMemoryStore", "SqliteStore", "make_store", "n_to_id",
    "DEFAULT_TIMEOUT_S",
    "is_filehelper", "is_group", "is_smart_reply_target",
    "ParsedAction", "parse_filehelper_input",
    "format_panel", "format_queue",
    "generate_candidates",
    "trigger_smart_reply", "handle_filehelper_message",
]


# ── Identity heuristics ────────────────────────────────────────────────────

_FILEHELPER_UID = "filehelper"
_GROUP_SUFFIX = "@chatroom"


def is_filehelper(uid: str) -> bool:
    return uid == _FILEHELPER_UID


def is_group(uid: str) -> bool:
    return uid.endswith(_GROUP_SUFFIX)


def _matches_at_mention(text: str, nickname: str) -> bool:
    """True if the text contains `@<nickname>` (with word-ish boundary).

    WeChat clients typically render group @-mentions as `@<nickname>` followed
    by a space or end-of-message; we accept that or a CJK boundary.
    """
    if not nickname:
        return False
    # Allow nickname plus a trailing space, end-of-string, or punctuation.
    pattern = r"@" + re.escape(nickname) + r"(\s|$|[，,。.!！?？:：])"
    return bool(re.search(pattern, text))


def is_smart_reply_target(uid: str, config: dict, *, text: str = "") -> bool:
    """Return True iff a message from ``uid`` should go through smart reply.

    Rules (in order):
      * feature flag off                                  → False
      * filehelper                                         → False (would loop)
      * uid == wechat_self_uid                             → False (your own
        messages to the bot must always reach the agent — the bot
        couldn't usefully draft a reply to yourself)
      * group + groups disabled                            → False
      * group + groups_at_only + no @<nickname> in text    → False
      * whitelist set and uid not in it                    → False
      * otherwise                                          → True
    """
    if not config.get("wechat_smart_reply", False):
        return False
    if is_filehelper(uid):
        return False
    self_uid = (config.get("wechat_self_uid") or "").strip()
    if self_uid and uid == self_uid:
        return False
    if is_group(uid):
        if not config.get("wechat_smart_reply_groups", False):
            return False
        if config.get("wechat_smart_reply_groups_at_only", False):
            nickname = (config.get("wechat_self_nickname") or "").strip()
            if not nickname or not _matches_at_mention(text, nickname):
                return False
    whitelist = config.get("wechat_smart_reply_whitelist") or []
    if whitelist and uid not in whitelist:
        return False
    return True


# ── Filehelper input parsing ──────────────────────────────────────────────

# A panel ID is exactly 2 uppercase letters at the start of input,
# optionally followed by space + payload.
_PANEL_ID_RE = re.compile(r"^([A-Z]{2})(?:\s+(.*))?$")


@dataclass(frozen=True)
class ParsedAction:
    kind: str                 # "send" | "skip" | "list" | "noop"
    panel_id: Optional[str]   # explicit ID if user prefixed with "AA"; else None
    text: Optional[str]       # send-text (for kind == "send")


def parse_filehelper_input(text: str, store) -> ParsedAction:
    """Interpret a filehelper-incoming message against the active panel queue.

    Behaviour:
      * "q" / "queue"               → kind="list"
      * "<ID> <subcmd>"             → kind=send/skip/list, panel_id set
      * "1" / "2" / "3"             → send candidate from latest active panel
      * "x" / "skip" / "跳过"       → skip latest active panel
      * any other text              → freeform send for latest active panel
    Returns ``noop`` if there's no active panel and the input wasn't ``q``.
    """
    s = text.strip()
    if not s:
        return ParsedAction("noop", None, None)

    low = s.lower()
    if low in ("q", "queue", "/q", "/queue", "队列"):
        return ParsedAction("list", None, None)

    # Explicit panel-ID prefix: "AA 1", "AA x", "AA hello", or "AA" alone (=list)
    m = _PANEL_ID_RE.match(s)
    if m:
        pid = m.group(1)
        rest = (m.group(2) or "").strip()
        return _classify_choice(rest, panel=store.get_by_id(pid), panel_id=pid)

    # No panel-ID prefix → applies to the latest active panel
    active = store.take_active() if store is not None else None
    return _classify_choice(s, panel=active, panel_id=None)


def _classify_choice(payload: str, panel: Optional[PendingPanel],
                     panel_id: Optional[str]) -> ParsedAction:
    if panel is None:
        return ParsedAction("noop", panel_id, None)

    s = payload.strip()
    low = s.lower()

    if low in ("", "show", "preview"):
        # "AA" alone means "show me this panel" — surfaced as `list` with id
        # so the caller can reformat or ignore.
        return ParsedAction("list", panel_id, None)
    if low in ("x", "skip", "/skip", "/x", "跳过", "不回"):
        return ParsedAction("skip", panel_id or panel.panel_id, None)
    if s in ("1", "2", "3"):
        idx = int(s) - 1
        if 0 <= idx < len(panel.candidates):
            return ParsedAction("send", panel_id or panel.panel_id,
                                panel.candidates[idx])
    return ParsedAction("send", panel_id or panel.panel_id, s)


# ── Panel + queue formatting ──────────────────────────────────────────────


def format_panel(panel: PendingPanel) -> str:
    label = panel.target_label or panel.target_uid[:8]
    msg = panel.message[:200]
    if len(panel.message) > 200:
        msg += "…"
    lines = [f"💬 [{panel.panel_id}] {label} → 「{msg}」", ""]
    for i, cand in enumerate(panel.candidates, start=1):
        lines.append(f"[{i}] {cand}")
    lines.append("")
    lines.append("回 1/2/3 发送 · 直接打字自定义 · x 跳过 · q 看队列")
    return "\n".join(lines)


def format_queue(panels: list[PendingPanel]) -> str:
    """Render the pending-panel queue for filehelper.

    Sorted oldest-first so the user sees what's been waiting longest.
    """
    if not panels:
        return "📋 当前没有待处理的消息"
    now = time.time()
    lines = ["📋 待处理 (oldest first):", ""]
    for p in panels:
        age = max(0, int(now - p.created_at))
        ttl = max(0, int(p.expires_at - now))
        label = p.target_label or p.target_uid[:8]
        msg_preview = p.message[:40]
        if len(p.message) > 40:
            msg_preview += "…"
        lines.append(
            f"  [{p.panel_id}] {label} ({_fmt_secs(age)}前 · 还剩 {_fmt_secs(ttl)})"
        )
        lines.append(f"        「{msg_preview}」")
    lines.append("")
    lines.append("发 <ID> 1/2/3/x/freeform 处理 · 例如 AA 2")
    return "\n".join(lines)


def _fmt_secs(s: int) -> str:
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h{(s % 3600) // 60}m"


# ── Candidate generation ──────────────────────────────────────────────────


def _build_prompt(message: str, sender_label: str,
                  contact: Optional[Contact],
                  history: list[ReplyHistoryEntry]) -> str:
    parts: list[str] = ["你是用户的微信回复助手。用户刚收到一条消息，"
                        "你需要起草 3 个简短自然的候选回复。"]

    if contact and (contact.relationship or contact.notes):
        ctx_lines = [f"\n关于发件人 {contact.label or sender_label}："]
        if contact.relationship:
            ctx_lines.append(f"- 关系: {contact.relationship}")
        if contact.notes:
            ctx_lines.append(f"- 备注: {contact.notes}")
        parts.append("\n".join(ctx_lines))

    if history:
        examples = []
        for h in history[:10]:
            txt = (h.text or "").strip()
            if txt:
                examples.append(f"- {txt}")
        if examples:
            parts.append("\n用户最近发出的几条回复（请模仿这种语气和长度，"
                         "不要照抄内容）：\n" + "\n".join(examples))

    parts.append(f"\n收到的消息（来自 {sender_label}）：\n{message[:500]}")

    parts.append(
        "\n要求：\n"
        "- 每条回复 ≤ 30 字\n"
        "- 语气：日常、自然、像真人随手回的\n"
        "- 3 条之间风格略有差异（例如：肯定 / 委婉拒绝 / 模糊延后）\n"
        "- 不要使用 emoji 或复杂标点\n"
        "- 不要解释你的选择，只输出回复\n"
        "\n格式（严格遵守，每行一条，开头数字+点+空格）：\n"
        "1. <回复一>\n"
        "2. <回复二>\n"
        "3. <回复三>"
    )

    return "\n".join(parts)


_LIST_RE = re.compile(r"^\s*[1-3][\.\)、]\s*(.+)$")
# Defensive trim: cheap models occasionally cram a duplicate candidate onto
# the tail of the previous one with no newline, e.g.
#   "3. 不太确定1. 有空呀，想干嘛"
# This regex finds the *next* numbered marker mid-string so we can cut there.
_TAIL_NUM_RE = re.compile(r"\s*\d+\s*[\.\)、]\s")


def _parse_candidates(text: str) -> list[str]:
    out: list[str] = []
    seen: set[int] = set()
    for line in text.splitlines():
        m = _LIST_RE.match(line)
        if not m:
            continue
        # Track seen indices so a duplicate "1. ..." line later in the
        # response can't overwrite the real candidate 1.
        idx_match = re.match(r"^\s*([1-3])", line)
        idx = int(idx_match.group(1)) if idx_match else 0
        if idx and idx in seen:
            continue
        cand = m.group(1).strip()
        # Cut anything that looks like a second numbered item bleeding in.
        cut = _TAIL_NUM_RE.search(cand)
        if cut:
            cand = cand[: cut.start()].strip()
        cand = cand.rstrip("。，,.!?！？")
        if cand:
            if idx:
                seen.add(idx)
            out.append(cand)
    return out[:3]


def generate_candidates(
    message: str,
    sender_label: str,
    config: dict,
    *,
    contact: Optional[Contact] = None,
    history: Optional[list[ReplyHistoryEntry]] = None,
    _stream_fn: Optional[Callable] = None,
) -> list[str]:
    """Produce 3 candidate replies via the auxiliary cheap model.

    Returns ``[]`` on any failure — caller falls back or skips.
    """
    prompt = _build_prompt(message, sender_label, contact, history or [])

    if _stream_fn is None:
        try:
            from cheetahclaws import providers
            from cheetahclaws.auxiliary import get_auxiliary_model
        except Exception:
            return []
        _stream_fn = providers.stream
        model = get_auxiliary_model(config)
    else:
        model = config.get("auxiliary_model") or "test-model"

    try:
        chunks = []
        for event in _stream_fn(
            model=model,
            system="只输出 3 个候选回复，按要求的格式。不要任何额外说明。",
            messages=[{"role": "user", "content": prompt}],
            tool_schemas=[],
            config={**config, "max_tokens": 200, "thinking": False},
        ):
            t = getattr(event, "text", None)
            if t:
                chunks.append(t)
        return _parse_candidates("".join(chunks))
    except Exception:
        return []


# ── Panel constructor ──────────────────────────────────────────────────────


def make_panel(target_uid: str, target_label: str, message: str,
               candidates: list[str], *,
               panel_id: str,
               timeout_s: float = DEFAULT_TIMEOUT_S) -> PendingPanel:
    now = time.time()
    return PendingPanel(
        panel_id=panel_id,
        target_uid=target_uid,
        target_label=target_label,
        message=message,
        candidates=candidates,
        created_at=now,
        expires_at=now + timeout_s,
    )


# ── High-level entry points ────────────────────────────────────────────────


def trigger_smart_reply(
    target_uid: str,
    target_label: str,
    message: str,
    store,
    config: dict,
    *,
    send_to_filehelper: Callable[[str], None],
    contacts: Optional[ContactsStore] = None,
    generate_fn: Optional[Callable] = None,
) -> bool:
    """Generate candidates, store the panel, push it to filehelper.

    Returns True on success, False if generation failed.
    """
    contact = contacts.get(target_uid) if contacts else None
    history = store.recent_replies(n=10, exclude_uid=target_uid) \
        if hasattr(store, "recent_replies") else []
    gen = generate_fn or generate_candidates
    candidates = gen(
        message, target_label, config,
        contact=contact, history=history,
    )
    if not candidates:
        return False
    if len(candidates) < 3:
        # Pad with conservative fallbacks so the panel is always 3 wide.
        for f in ("好", "稍后回你", "我看一下"):
            if len(candidates) >= 3:
                break
            if f not in candidates:
                candidates.append(f)
    timeout_s = float(config.get("wechat_smart_reply_timeout_s", DEFAULT_TIMEOUT_S))
    pid = store.assign_next_id()
    panel = make_panel(target_uid, target_label, message, candidates[:3],
                       panel_id=pid, timeout_s=timeout_s)
    store.put(panel)
    send_to_filehelper(format_panel(panel))
    return True


def handle_filehelper_message(
    text: str,
    store,
    *,
    send_to_target: Callable[[str, str], None],
    send_to_filehelper: Callable[[str], None],
) -> bool:
    """Route a filehelper-incoming message against the active panel queue.

    Returns True if the message was consumed (caller should NOT pass it on
    to the agent), False if the input wasn't a smart-reply action and
    should fall through to the normal bridge dispatch.
    """
    action = parse_filehelper_input(text, store)

    if action.kind == "noop":
        return False

    if action.kind == "list":
        if action.panel_id:
            # "AA" alone — show that one panel
            p = store.get_by_id(action.panel_id)
            if p is None:
                send_to_filehelper(f"⚠ [{action.panel_id}] 已过期或不存在")
                return True
            send_to_filehelper(format_panel(p))
            return True
        send_to_filehelper(format_queue(store.list_active()))
        return True

    # send / skip always have a panel_id by this point — parse_filehelper_input
    # fills it in either from the explicit ID prefix or from the latest-active
    # panel. Defensive None handling kept for forward-compat.
    if not action.panel_id:
        return False
    panel = store.consume_by_id(action.panel_id)
    if panel is None:
        send_to_filehelper(f"⚠ [{action.panel_id}] 已过期或不存在")
        return True

    if action.kind == "skip":
        send_to_filehelper(f"⏭ 已跳过 [{panel.panel_id}] {panel.target_label}")
        return True

    if action.kind == "send" and action.text:
        send_to_target(panel.target_uid, action.text)
        if hasattr(store, "write_reply"):
            try:
                source = _classify_source(action.text, panel.candidates)
                store.write_reply(
                    to_uid=panel.target_uid,
                    to_label=panel.target_label,
                    text=action.text,
                    source=source,
                )
            except Exception:
                pass
        send_to_filehelper(
            f"✓ 已发送给 [{panel.panel_id}] {panel.target_label}：{action.text[:60]}"
        )
        return True

    return False


def _classify_source(text: str, candidates: list[str]) -> str:
    """Tag a confirmed reply as candidate_N (matched) or 'freeform'."""
    for i, c in enumerate(candidates, start=1):
        if text == c:
            return f"candidate_{i}"
    return "freeform"
