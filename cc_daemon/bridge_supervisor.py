"""bridge_supervisor.py — daemon-owned bridge worker lifecycle (RFC 0002 F-6/7/8).

Each bridge (Telegram, Slack, WeChat) runs as one thread inside the daemon
process so that:

  * Users can keep texting after the REPL is closed.
  * Runner notifications (RFC 0002 F-4 #2 — ``notify`` IPC from the runner
    subprocess) have a place to land outside the agent loop.
  * One bridge crashing doesn't drag the others down — each thread has
    its own stop_event and last_error slot.

Scope of the **F-6/7/8 skeleton**:

  * POSIX + Windows — pure-Python threading.Event + supervisor entry point.
    The actual bridge transport (HTTP long-poll for Telegram, etc.) is
    re-used unmodified from the existing ``bridges/<kind>.py`` modules,
    which means today's REPL bridges and the daemon's bridges share the
    same network code.
  * Outbound is fully landed: :func:`notify` dispatches a text into the
    running bridge's send path, surfaced as a JSON-RPC method
    (``bridge.send``) and consumed by the F-4 runner supervisor.
  * Inbound (phone → ``session.send`` → SSE) is **deferred** to a Phase 2
    of F-6.  Documented in the RFC §F-6 "Phase 2 — inbound refactor".
    Until Phase 2 lands, an enabled bridge still routes phone messages
    via the existing REPL ``run_query`` callback if a REPL is attached;
    otherwise the message is captured in the daemon log.

Feature-flag gating (RFC roadmap §"Bridge flag"):

  * ``CHEETAHCLAWS_ENABLE_F6`` → Telegram bridge in daemon.
  * ``CHEETAHCLAWS_ENABLE_F7`` → Slack bridge in daemon (depends on F-6).
  * ``CHEETAHCLAWS_ENABLE_F8`` → WeChat bridge in daemon (depends on F-6).

When the flag is off, ``enabled(kind)`` returns False and ``start(kind)``
raises RuntimeError. The REPL ``/telegram`` / ``/slack`` / ``/wechat``
commands keep using the in-process loop they always did.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


SUPPORTED_KINDS = ("telegram", "slack", "wechat")


# Feature-flag env vars per bridge — set independently so a deployment
# can run Telegram-in-daemon while Slack and WeChat still live in the REPL.
_FLAG_ENV = {
    "telegram": "CHEETAHCLAWS_ENABLE_F6",
    "slack":    "CHEETAHCLAWS_ENABLE_F7",
    "wechat":   "CHEETAHCLAWS_ENABLE_F8",
}


def enabled(kind: str) -> bool:
    """Return True iff the named bridge is allowed to run in-daemon.

    Mirrors :func:`cc_daemon.runner_supervisor.enabled`'s shape: truthy
    env vars (``1``/``true``/``yes``/``on``) flip the bridge on. The flag
    is per-kind because users may want to migrate one bridge at a time.
    """
    if kind not in SUPPORTED_KINDS:
        return False
    flag = os.environ.get(_FLAG_ENV[kind], "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _get_event_bus():
    """Lazy import — events module pulls in SQLite. Returns None if the
    daemon's bus isn't initialised (unit tests that exercise just this
    supervisor in isolation)."""
    try:
        from . import events
        return events.get_bus()
    except Exception:
        return None


# ── Worker registry ──────────────────────────────────────────────────────


@dataclass
class BridgeHandle:
    kind:          str
    config:        dict
    started_at:    float
    stop_event:    threading.Event
    thread:        threading.Thread             = field(repr=False)
    last_error:    str  = ""
    last_poll_at:  Optional[float] = None
    # Outbound sender — chosen by ``_resolve_sender`` at start time.
    # Signature: (config, text) -> bool. Returns True on success.
    sender:        Optional[Callable[[dict, str], bool]] = field(repr=False, default=None)
    # RFC 0002 F-6 Phase 2 — when True, the worker uses the slim
    # daemon-driven loop (inbound → session.send event, subscribe to
    # session_outbound for delivery) instead of the legacy supervisor
    # in ``bridges/<kind>.py``. Default False keeps Phase 1 semantics.
    daemon_phase2: bool = False

    def is_alive(self) -> bool:
        return self.thread.is_alive() and not self.stop_event.is_set()

    def session_id(self) -> str:
        """Phase 2 session identifier for this bridge. Format:
        ``"<kind-prefix>:<chat-key>"`` so per-chat sessions stay
        independent (one Telegram bot can talk to multiple chats with
        their own conversation histories — same is true for Slack
        channels and WeChat contacts).
        """
        if self.kind == "telegram":
            return f"tg:{self.config.get('telegram_chat_id', 0)}"
        if self.kind == "slack":
            return f"sl:{self.config.get('slack_channel', '')}"
        if self.kind == "wechat":
            return f"wc:{self.config.get('wechat_user_id', '')}"
        return f"{self.kind}:{id(self)}"


_handles: dict[str, BridgeHandle] = {}
_handles_lock = threading.Lock()


def get(kind: str) -> Optional[BridgeHandle]:
    with _handles_lock:
        h = _handles.get(kind)
        if h is not None and not h.thread.is_alive():
            # Reflect a crashed worker the same way runner_supervisor does:
            # keep the registry slot but flag the failure for observability.
            if not h.last_error:
                h.last_error = "thread exited"
        return h


def list_all() -> list[BridgeHandle]:
    with _handles_lock:
        return list(_handles.values())


# ── Outbound senders (per bridge) ───────────────────────────────────────


def _resolve_sender(kind: str) -> Callable[[dict, str], bool]:
    """Return the function that sends an outbound text for ``kind``.

    Reuses the network code from ``bridges/<kind>.py`` unmodified so the
    daemon and the REPL speak HTTP/long-poll identically; only the
    *ownership* of the loop changes.

    Senders are lazy-imported so a daemon that never starts WeChat
    doesn't pay for the WeChat transport at import time.
    """
    if kind == "telegram":
        from bridges import telegram as _tg
        def _send(cfg: dict, text: str) -> bool:
            token = cfg.get("telegram_token", "")
            chat_id = cfg.get("telegram_chat_id", 0)
            if not token or not chat_id:
                return False
            try:
                _tg._tg_send(token, int(chat_id), text)
                return True
            except Exception:
                return False
        return _send
    if kind == "slack":
        from bridges import slack as _sk
        def _send(cfg: dict, text: str) -> bool:
            token   = cfg.get("slack_token", "")
            channel = cfg.get("slack_channel", "")
            if not token or not channel:
                return False
            try:
                # bridges/slack.py exposes _slack_send(token, channel, text).
                _sk._slack_send(token, channel, text)
                return True
            except Exception:
                return False
        return _send
    if kind == "wechat":
        from bridges import wechat as _wc
        def _send(cfg: dict, text: str) -> bool:
            # WeChat send is per-user: ``wechat_user_id`` in config names
            # the destination contact (set by the WeChat poll loop when
            # the QR-login attaches; configurable for tests).
            user_id = cfg.get("wechat_user_id", "")
            if not user_id:
                return False
            try:
                _wc._wx_send(user_id, text, cfg)
                return True
            except Exception:
                return False
        return _send
    raise ValueError(f"unsupported bridge kind: {kind!r}")


# ── Bridge worker (one thread) ──────────────────────────────────────────


def _bridge_worker(handle: BridgeHandle) -> None:
    """The per-bridge supervisor loop.

    For F-6/7/8 we keep the existing supervisor functions in
    ``bridges/<kind>.py`` and just run them inside the daemon. Each bridge's
    supervisor is responsible for its own reconnect/backoff (Telegram's
    ``_tg_supervisor`` already does this).

    Phase 1 contract: the supervisor exits when its module-level stop
    flag is set. We coordinate via the bridge module's existing
    ``_<kind>_stop`` Event, mirrored by ``handle.stop_event`` so callers
    only see one shutdown surface.
    """
    bus = _get_event_bus()
    if bus is not None:
        try:
            bus.publish("bridge_started",
                        {"kind": handle.kind, "config": _safe_cfg(handle.config)})
        except Exception:
            pass

    try:
        # RFC 0002 F-6 Phase 2 — when ``daemon_phase2`` is set, the worker
        # runs the slim daemon-driven loop: inbound message → ``session.send``
        # event on the bus, subscribe to ``session_outbound`` events scoped
        # to this bridge's session_id and forward to the chat via
        # ``handle.sender``. The legacy ``bridges/<kind>.py`` supervisor is
        # bypassed (it's REPL-shaped and depends on session_ctx.run_query).
        if handle.daemon_phase2:
            _phase2_worker(handle)
            return

        if handle.kind == "telegram":
            from bridges import telegram as _tg
            # Tie our stop_event to the bridge's module-level Event so
            # ``stop()`` here is observed by the existing supervisor.
            _tg._telegram_stop = handle.stop_event
            _tg._tg_supervisor(
                handle.config.get("telegram_token", ""),
                int(handle.config.get("telegram_chat_id", 0) or 0),
                handle.config,
            )
        elif handle.kind == "slack":
            from bridges import slack as _sk
            _sk._slack_stop = handle.stop_event
            _sk._slack_supervisor(
                handle.config.get("slack_token", ""),
                handle.config.get("slack_channel", ""),
                handle.config,
            )
        elif handle.kind == "wechat":
            from bridges import wechat as _wc
            _wc._wechat_stop = handle.stop_event
            # WeChat starts its own auth path (QR login) inside
            # ``_wx_start_bridge``; the supervisor expects token+base_url
            # already attached to the config. If unset we surface a clear
            # error rather than letting _wx_supervisor crash on a None.
            token    = handle.config.get("wechat_token", "")
            base_url = handle.config.get("wechat_base_url", "")
            if not token or not base_url:
                handle.last_error = (
                    "wechat config missing 'wechat_token' / "
                    "'wechat_base_url'; run /wechat login first to "
                    "populate them.")
                return
            _wc._wx_supervisor(token, base_url, handle.config)
        else:
            handle.last_error = f"unsupported kind: {handle.kind!r}"
    except Exception as exc:
        handle.last_error = f"{type(exc).__name__}: {exc}"[:512]
        if bus is not None:
            try:
                bus.publish("bridge_crash",
                            {"kind": handle.kind,
                             "error": handle.last_error})
            except Exception:
                pass
    finally:
        # Persist the terminal state — the bridges row reflects "enabled"
        # at the moment of shutdown so observers know whether the loop
        # exited cleanly.
        _db_finalize_bridge(handle)
        if bus is not None:
            try:
                bus.publish("bridge_stopped",
                            {"kind": handle.kind,
                             "last_error": handle.last_error})
            except Exception:
                pass


_SECRET_KEY_FRAGMENTS = (
    "token", "secret", "api_key", "apikey",
    "password", "passwd", "auth",
)


def _safe_cfg(cfg: dict) -> dict:
    """Strip secrets before bus-publishing a config snapshot.

    The bridge config is merged with the daemon's full config at
    ``bridge.start`` time (so callers don't have to repeat tokens
    already stored under ``cc_config``), which means *provider*
    secrets — ``anthropic_api_key`` / ``openai_api_key`` / etc. — can
    bleed through if we don't redact them too. Match any key whose
    lowercase form contains a known secret fragment; values are kept
    only as their last 4 chars (or ``***`` for shorter strings).

    Chat IDs / channels / user IDs are intentionally NOT redacted —
    they're not secret on their own and operators want to see them in
    `daemon status` for triage.
    """
    out = {}
    for k, v in cfg.items():
        kl = k.lower() if isinstance(k, str) else ""
        is_secret = isinstance(v, str) and any(
            frag in kl for frag in _SECRET_KEY_FRAGMENTS)
        if is_secret:
            out[k] = "***" + v[-4:] if len(v) > 4 else "***"
        else:
            out[k] = v
    return out


# ── Lifecycle ────────────────────────────────────────────────────────────


def start(kind: str, config: dict, *, daemon_phase2: bool = False) -> BridgeHandle:
    """Spawn the named bridge inside the daemon. Idempotent: starting a
    bridge that's already running raises RuntimeError so the caller can
    decide whether to stop+restart or carry on."""
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"unsupported bridge kind: {kind!r}; "
                         f"expected one of {SUPPORTED_KINDS}")
    if not enabled(kind):
        raise RuntimeError(
            f"bridge {kind!r} not enabled — set {_FLAG_ENV[kind]}=1")
    if kind == "slack" and not enabled("telegram"):
        # F-7 depends on F-6 scaffolding; surface this as a clear error
        # rather than letting the import surprise the caller.
        raise RuntimeError("F-7 (slack) depends on F-6 (telegram); "
                           "enable CHEETAHCLAWS_ENABLE_F6 first")
    if kind == "wechat" and not enabled("telegram"):
        raise RuntimeError("F-8 (wechat) depends on F-6 (telegram); "
                           "enable CHEETAHCLAWS_ENABLE_F6 first")

    with _handles_lock:
        existing = _handles.get(kind)
        if existing is not None and existing.is_alive():
            raise RuntimeError(f"bridge {kind!r} already running")

    stop_event = threading.Event()
    handle = BridgeHandle(
        kind=kind, config=dict(config),
        started_at=time.time(),
        stop_event=stop_event,
        thread=threading.Thread(),  # placeholder, replaced below
        sender=_resolve_sender(kind),
        daemon_phase2=bool(daemon_phase2),
    )
    handle.thread = threading.Thread(
        target=_bridge_worker, args=(handle,),
        daemon=True, name=f"bridge-{kind}",
    )

    with _handles_lock:
        _handles[kind] = handle
    _db_upsert_bridge(handle, enabled_flag=True)
    handle.thread.start()
    return handle


def stop(kind: str, *, timeout_s: float = 5.0) -> bool:
    """Signal a bridge to shut down and join its thread."""
    with _handles_lock:
        handle = _handles.get(kind)
        if handle is None:
            return False

    handle.stop_event.set()
    handle.thread.join(timeout=timeout_s)
    alive = handle.thread.is_alive()
    if not alive:
        # Mark disabled in SQLite but keep the row for audit.
        _db_upsert_bridge(handle, enabled_flag=False)
        # Identity check — a concurrent bridge.start could have replaced
        # the slot with a fresh handle while we were joining (the old
        # handle's is_alive() goes False the moment we set stop_event,
        # which lets start() believe nothing is running). Pop only if
        # the slot still holds this exact handle.
        with _handles_lock:
            current = _handles.get(kind)
            if current is handle:
                _handles.pop(kind, None)
    return not alive


def stop_all(*, timeout_s: float = 5.0) -> int:
    """Stop every running bridge. Returns the number cleanly stopped."""
    kinds = [h.kind for h in list_all()]
    n = 0
    for kind in kinds:
        if stop(kind, timeout_s=timeout_s):
            n += 1
    return n


# ── Outbound mailbox (consumed by F-4 #2 + bridge.send RPC) ─────────────


def notify(kind: str, text: str) -> bool:
    """Send an outbound text via the named bridge. Returns True iff
    delivered, False if the bridge isn't running or the sender failed.

    This is the F-4 #2 hook: a runner subprocess's ``notify`` IPC frame
    feeds straight into here when the originator's bridge is running
    in-daemon. If no bridge is registered, the call is a quiet no-op
    so the runner's iteration log isn't spammed with "no bridge" errors
    every iteration.
    """
    if not text:
        return False
    if kind == "*":
        # Broadcast: deliver to every running bridge. Useful as a default
        # for runners that don't know which bridge their originator owns.
        any_ok = False
        for h in list_all():
            if h.is_alive() and h.sender is not None:
                try:
                    if h.sender(h.config, text):
                        any_ok = True
                except Exception as e:
                    h.last_error = f"notify: {type(e).__name__}: {e}"[:256]
        return any_ok

    handle = get(kind)
    if handle is None or not handle.is_alive() or handle.sender is None:
        return False
    try:
        ok = handle.sender(handle.config, text)
        if ok:
            handle.last_poll_at = time.time()
        return ok
    except Exception as e:
        handle.last_error = f"notify: {type(e).__name__}: {e}"[:256]
        return False


# ── SQLite persistence (bridges table from F-2) ─────────────────────────


def _db_upsert_bridge(handle: BridgeHandle, *, enabled_flag: bool) -> bool:
    """INSERT-or-UPDATE the bridges row for this kind."""
    try:
        from .schema import get_conn
        conn = get_conn()
        conn.execute(
            "INSERT INTO bridges (kind, enabled, config_json, "
            " last_poll_at, last_error) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(kind) DO UPDATE SET "
            "  enabled       = excluded.enabled, "
            "  config_json   = excluded.config_json, "
            "  last_poll_at  = excluded.last_poll_at, "
            "  last_error    = excluded.last_error",
            (
                handle.kind,
                1 if enabled_flag else 0,
                json.dumps(_safe_cfg(handle.config)),
                _iso_now(),
                handle.last_error or None,
            ),
        )
        conn.commit()
        return True
    except Exception:
        return False


def _db_finalize_bridge(handle: BridgeHandle) -> bool:
    """Mark a bridge row disabled at worker exit."""
    try:
        from .schema import get_conn
        conn = get_conn()
        conn.execute(
            "UPDATE bridges SET enabled = 0, last_poll_at = ?, "
            "last_error = ? WHERE kind = ?",
            (_iso_now(), handle.last_error or None, handle.kind),
        )
        conn.commit()
        return True
    except Exception:
        return False


def list_persisted() -> list[dict]:
    """Read the bridges table — useful for ``bridge.list`` even after a
    daemon restart wipes the in-memory registry."""
    try:
        from .schema import get_conn
        conn = get_conn()
        rows = conn.execute(
            "SELECT kind, enabled, config_json, last_poll_at, last_error "
            "FROM bridges ORDER BY kind"
        ).fetchall()
    except Exception:
        return []
    out: list[dict] = []
    for r in rows:
        try:
            cfg = json.loads(r["config_json"] or "{}")
        except (TypeError, ValueError):
            cfg = {}
        out.append({
            "kind":         r["kind"],
            "enabled":      bool(r["enabled"]),
            "config":       cfg,
            "last_poll_at": r["last_poll_at"],
            "last_error":   r["last_error"],
        })
    return out


# ── Phase 2 worker (RFC 0002 F-6 Phase 2) ───────────────────────────────


def _phase2_worker(handle: BridgeHandle) -> None:
    """Daemon-driven bridge loop.

    Replaces the legacy ``bridges/<kind>.py:_<kind>_supervisor`` REPL
    shape with three independent, composable pieces:

      1. **Outbound subscriber.**  Subscribe to the daemon event bus,
         filter for ``session_outbound`` events tagged with this bridge's
         ``session_id`` (or a broadcast event with no target list), call
         ``handle.sender(handle.config, text)`` to deliver.
      2. **Inbound poller.**  Re-use the per-kind transport function
         from ``bridges/<kind>.py`` (``_tg_api`` long-poll,
         ``_slack_api`` long-poll, ``_wx_get_updates``) to read messages
         off the wire. For every message, publish ``session_inbound``
         on the bus with ``origin=<kind>`` and ``session_id`` from
         ``handle.session_id()``.
      3. **Stop watcher.**  ``handle.stop_event`` cleanly shuts both
         threads down. The outbound subscriber's queue.get() honours the
         event via a periodic timeout; the inbound poller's network
         loop returns on the same flag.

    Permission routing: ``session_inbound`` carries ``origin=<kind>:<chat>``
    so the agent driver (REPL/Web/another bridge) can pin permission
    requests for this turn back to this bridge by stamping the same
    string as ``originator`` in the PermissionStore.  We don't drive
    the agent here — that's the responsibility of whatever subscribes
    to ``session_inbound``.
    """
    from . import events as _events_mod

    session_id = handle.session_id()
    origin_str = f"{handle.kind}:{session_id}"
    bus = _events_mod.get_bus()

    # ── Outbound subscriber ────────────────────────────────────────────
    out_q = bus.subscribe()

    def _outbound_loop():
        try:
            while not handle.stop_event.is_set():
                try:
                    ev = out_q.get(timeout=1.0)
                except Exception:
                    continue
                if not isinstance(ev, dict):
                    continue
                if ev.get("type") != "session_outbound":
                    continue
                data = ev.get("data") or {}
                if data.get("session_id") != session_id:
                    continue
                targets = data.get("target_bridges")
                if targets and handle.kind not in targets:
                    continue
                text = str(data.get("text", "")) or ""
                if not text or handle.sender is None:
                    continue
                try:
                    if handle.sender(handle.config, text):
                        handle.last_poll_at = time.time()
                except Exception as e:
                    handle.last_error = (
                        f"phase2 outbound: {type(e).__name__}: {e}")[:256]
        finally:
            try:
                bus.unsubscribe(out_q)
            except Exception:
                pass

    out_t = threading.Thread(target=_outbound_loop, daemon=True,
                              name=f"bridge-phase2-out-{handle.kind}")
    out_t.start()

    # ── Inbound poller (per-kind) ──────────────────────────────────────
    try:
        if handle.kind == "telegram":
            _phase2_telegram_inbound(handle, bus, session_id, origin_str)
        elif handle.kind == "slack":
            _phase2_slack_inbound(handle, bus, session_id, origin_str)
        elif handle.kind == "wechat":
            _phase2_wechat_inbound(handle, bus, session_id, origin_str)
        else:
            handle.last_error = f"phase2 not implemented for {handle.kind}"
    except Exception as exc:
        handle.last_error = f"phase2 inbound: {type(exc).__name__}: {exc}"[:512]
    finally:
        # The outbound loop only exits on stop_event; set it now so this
        # worker tears down cleanly.
        handle.stop_event.set()
        out_t.join(timeout=2.0)


def _phase2_telegram_inbound(handle, bus, session_id, origin_str):
    from bridges import telegram as _tg
    token   = handle.config.get("telegram_token", "")
    chat_id = int(handle.config.get("telegram_chat_id", 0) or 0)
    if not token or not chat_id:
        handle.last_error = "phase2 telegram: missing token/chat_id"
        return

    # Flush old messages so a restart doesn't reply to everything from
    # the last week.
    flush = _tg._tg_api(token, "getUpdates", {"offset": -1, "timeout": 0})
    offset = 0
    if flush and flush.get("ok") and flush.get("result"):
        offset = flush["result"][-1]["update_id"] + 1

    # Long-poll timeout of 25 s would mean ``stop()`` waits up to that
    # long for the HTTP call to return before observing the stop_event.
    # Keep it short (~5 s) so a SIGTERM or `bridge.stop` lands in
    # operator-tolerable time. The legacy supervisor uses 30 s because
    # it has full control of stop ordering; the daemon-side worker
    # benefits more from snappy shutdown than from squeezing latency.
    POLL_TIMEOUT_S = 5

    while not handle.stop_event.is_set():
        result = _tg._tg_api(token, "getUpdates", {
            "offset": offset, "timeout": POLL_TIMEOUT_S,
            "allowed_updates": ["message"],
        })
        if not result or not result.get("ok"):
            handle.stop_event.wait(5)
            continue
        for upd in result.get("result", []):
            offset = upd["update_id"] + 1
            msg = upd.get("message") or {}
            if (msg.get("chat") or {}).get("id") != chat_id:
                continue
            text = msg.get("text") or ""
            if not text:
                continue
            bus.publish("session_inbound", {
                "session_id": session_id,
                "text":       text,
                "origin":     origin_str,
                "message_id": f"tg_{upd['update_id']}",
                "ts":         time.time(),
            })
            handle.last_poll_at = time.time()


def _phase2_slack_inbound(handle, bus, session_id, origin_str):
    """Slack Phase 2 inbound — uses the existing ``_slack_api`` helper
    to call ``conversations.history`` with an ``oldest`` cursor so each
    iteration only pulls genuinely-new messages.

    Matches the cursor-initialisation discipline of the legacy
    ``_slack_poll_loop`` (``bridges/slack.py``):
      * Initialise ``cursor`` to the current wall-clock time so the
        bridge does **not** replay backlog on first poll (otherwise a
        Phase 2 restart would re-announce every recent message).
      * Update ``cursor`` to each message's ``ts`` as we process it.
    """
    from bridges import slack as _sk
    token   = handle.config.get("slack_token", "")
    channel = handle.config.get("slack_channel", "")
    if not token or not channel:
        handle.last_error = "phase2 slack: missing token/channel"
        return

    # Seed the cursor at "now" so the first poll skips backlog. The
    # legacy supervisor does the same — see ``bridges/slack.py:_slack_poll_loop``.
    cursor = str(time.time())
    while not handle.stop_event.is_set():
        params: dict = {
            "channel": channel,
            "oldest":  cursor,
            "limit":   20,
        }
        result = _sk._slack_api(token, "conversations.history", params)
        if not result or not result.get("ok"):
            handle.stop_event.wait(5)
            continue
        # Slack returns messages newest-first; reverse so we publish in
        # chronological order.
        for m in list(reversed(result.get("messages") or [])):
            ts_str = m.get("ts", "")
            if not ts_str or ts_str <= cursor:
                continue
            cursor = ts_str
            text = m.get("text") or ""
            if not text or m.get("bot_id"):  # skip our own messages
                continue
            bus.publish("session_inbound", {
                "session_id": session_id,
                "text":       text,
                "origin":     origin_str,
                "message_id": f"sl_{ts_str}",
                "ts":         time.time(),
            })
            handle.last_poll_at = time.time()
        # Wait honours stop_event so shutdown is observed within ~3 s.
        handle.stop_event.wait(3)


def _phase2_wechat_inbound(handle, bus, session_id, origin_str):
    """WeChat Phase 2 inbound — long-poll via ``_wx_get_updates``.

    The iLink protocol's response shape (verified against
    ``bridges/wechat.py:411``) is:

      ``{"ret": int, "errcode": int, "msgs": [...], "get_updates_buf": str}``

    Each message in ``msgs`` has ``from_user_id``, ``content`` (falls
    back to ``text``), and an identifier in one of ``message_id`` /
    ``seq`` / ``client_id``. We use ``get_updates_buf`` from the
    response as the next iteration's sync token so the long-poll only
    returns new messages.
    """
    from bridges import wechat as _wc
    token    = handle.config.get("wechat_token", "")
    base_url = handle.config.get("wechat_base_url", "")
    user_id  = handle.config.get("wechat_user_id", "")
    if not token or not base_url or not user_id:
        handle.last_error = "phase2 wechat: missing token/base_url/user_id"
        return

    sync_buf = ""
    while not handle.stop_event.is_set():
        result = _wc._wx_get_updates(base_url, token, sync_buf)
        if not result:
            handle.stop_event.wait(5)
            continue
        # iLink auth/ratelimit failures come back as `ret != 0`; the
        # legacy poll loop logs and retries — we do the same and wait
        # honours stop_event for fast shutdown.
        if int(result.get("ret", 0) or 0) != 0:
            handle.last_error = (
                f"wx ret={result.get('ret')} errcode={result.get('errcode')}"
            )[:256]
            handle.stop_event.wait(5)
            continue
        new_buf = result.get("get_updates_buf")
        if new_buf:
            sync_buf = str(new_buf)
        for m in result.get("msgs") or []:
            if str(m.get("from_user_id") or "").strip() != user_id:
                continue
            text = str(m.get("content") or m.get("text") or "").strip()
            if not text:
                continue
            mid = (m.get("message_id") or m.get("seq")
                   or m.get("client_id") or time.time())
            bus.publish("session_inbound", {
                "session_id": session_id,
                "text":       text,
                "origin":     origin_str,
                "message_id": f"wc_{mid}",
                "ts":         time.time(),
            })
            handle.last_poll_at = time.time()


__all__ = [
    "BridgeHandle",
    "SUPPORTED_KINDS",
    "enabled",
    "get",
    "list_all",
    "list_persisted",
    "notify",
    "start",
    "stop",
    "stop_all",
]
