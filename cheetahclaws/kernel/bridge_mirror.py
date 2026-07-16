"""bridge_mirror.py — bridges ↔ kernel.mbox glue (RFC 0018).

Helpers that map the Telegram / WeChat / Slack / … bridge surfaces
onto canonical kernel.mbox topics. The existing ``bridges/`` code is
untouched; a future patch (or external code) can adopt this mirror
with two function calls.

Topics:
    bridge.<kind>.inbound       messages received from the world
    bridge.<kind>.outbound      messages queued for the world

Where ``<kind>`` is a lowercase identifier (telegram, wechat, slack,
discord, or any custom string matching ``[a-z][a-z0-9_-]*``).

Subscribers see ``BridgeMessage`` JSON in the kernel mailbox payload
and can pull bridge-specific extras from ``payload['metadata']``.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .api import Kernel


_KIND_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
MESSAGE_KIND = "bridge.message"


# ── Bridge kind constants ─────────────────────────────────────────────────


class BridgeKind:
    TELEGRAM = "telegram"
    WECHAT   = "wechat"
    SLACK    = "slack"
    DISCORD  = "discord"
    KNOWN    = (TELEGRAM, WECHAT, SLACK, DISCORD)


# ── Topic helpers ─────────────────────────────────────────────────────────


def _validate_kind(kind: str) -> None:
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"kind must be non-empty str, got {kind!r}")
    if not _KIND_RE.match(kind):
        raise ValueError(
            f"kind must match {_KIND_RE.pattern} (lower-case alnum + _ -), "
            f"got {kind!r}"
        )


def inbound_topic(kind: str) -> str:
    _validate_kind(kind)
    return f"bridge.{kind}.inbound"


def outbound_topic(kind: str) -> str:
    _validate_kind(kind)
    return f"bridge.{kind}.outbound"


# ── BridgeMessage ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BridgeMessage:
    kind:      str            # bridge identifier
    sender:    str            # opaque sender id (chat_id, user_id, …)
    text:      str
    direction: str            # 'inbound' | 'outbound'
    metadata:  dict = field(default_factory=dict)
    ts:        float = 0.0

    def to_dict(self) -> dict:
        return {
            "kind":      self.kind,
            "sender":    self.sender,
            "text":      self.text,
            "direction": self.direction,
            "metadata":  self.metadata,
            "ts":        self.ts,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "BridgeMessage":
        return cls(
            kind      = str(payload.get("kind", "")),
            sender    = str(payload.get("sender", "")),
            text      = str(payload.get("text", "")),
            direction = str(payload.get("direction", "")),
            metadata  = dict(payload.get("metadata") or {}),
            ts        = float(payload.get("ts", 0.0)),
        )


# ── BridgeMirror ─────────────────────────────────────────────────────────


class BridgeMirror:
    """Thin wrapper around kernel.mbox publish + subscribe with the
    canonical topic naming + BridgeMessage shape."""

    def __init__(
        self,
        kernel: "Kernel",
        *,
        sender_pid: Optional[int] = None,
    ) -> None:
        self._kernel = kernel
        self._sender_pid = sender_pid

    # ── inbound (world → kernel) ─────────────────────────────────────

    def mirror_inbound(
        self,
        *,
        kind: str,
        sender: str,
        text: str,
        metadata: Optional[dict] = None,
        ts: Optional[float] = None,
    ) -> dict:
        """Publish an inbound bridge event to ``bridge.<kind>.inbound``.

        Returns the dict from ``kernel.mbox.publish``: ``{delivered,
        rejected, msg_ids}``. ``delivered=0`` when nobody's
        subscribed, which is fine — the publisher's job is to emit;
        the subscriber's job is to listen.
        """
        msg = BridgeMessage(
            kind=kind, sender=str(sender), text=str(text),
            direction="inbound",
            metadata=metadata or {},
            ts=ts if ts is not None else time.time(),
        )
        return self._kernel.mailbox.publish(
            sender_pid=self._sender_pid,
            topic=inbound_topic(kind),
            kind=MESSAGE_KIND,
            payload=msg.to_dict(),
        )

    # ── outbound (kernel → world) ────────────────────────────────────

    def queue_outbound(
        self,
        *,
        kind: str,
        recipient: str,
        text: str,
        metadata: Optional[dict] = None,
        ts: Optional[float] = None,
    ) -> dict:
        """Publish an outbound message for the bridge worker to pick
        up. The bridge's OutboundReceiver thread will pull this and
        call its send_fn."""
        msg = BridgeMessage(
            kind=kind, sender=str(recipient), text=str(text),
            direction="outbound",
            metadata=metadata or {},
            ts=ts if ts is not None else time.time(),
        )
        return self._kernel.mailbox.publish(
            sender_pid=self._sender_pid,
            topic=outbound_topic(kind),
            kind=MESSAGE_KIND,
            payload=msg.to_dict(),
        )

    # ── subscribe sugar ──────────────────────────────────────────────

    def subscribe_inbound(self, agent_pid: int, kind: str) -> None:
        self._kernel.mailbox.subscribe(agent_pid, inbound_topic(kind))

    def subscribe_outbound(self, agent_pid: int, kind: str) -> None:
        self._kernel.mailbox.subscribe(agent_pid, outbound_topic(kind))


# ── OutboundReceiver ─────────────────────────────────────────────────────


class OutboundReceiver:
    """Background drainer for outbound bridge messages.

    Wraps a worker agent's mailbox: on each tick, pulls pending
    BridgeMessages, decodes them, and calls ``send_fn`` for each.
    The bridge wrapper provides ``send_fn`` to talk to the actual
    bridge API (Telegram bot.send_message, Slack chat.postMessage,
    etc.).

    Lifecycle:
        rx = OutboundReceiver(kernel, agent_pid=42, kind="telegram",
                              send_fn=lambda m: bot.send(...))
        rx.start()       # spawns daemon thread
        ...
        rx.stop()        # graceful halt; thread joined

    Or use ``drain_once()`` for one-shot ticks (useful in tests).
    """

    def __init__(
        self,
        kernel: "Kernel",
        *,
        agent_pid: int,
        kind: str,
        send_fn: Callable[[BridgeMessage], None],
        poll_interval_s: float = 1.0,
        batch_size: int = 32,
        on_error: Optional[Callable[[BridgeMessage, BaseException], None]] = None,
    ) -> None:
        _validate_kind(kind)
        if not isinstance(agent_pid, int):
            raise ValueError(f"agent_pid must be int, got {type(agent_pid).__name__}")
        if not callable(send_fn):
            raise ValueError("send_fn must be callable")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0")
        if batch_size < 1:
            batch_size = 1

        self._kernel = kernel
        self._agent_pid = agent_pid
        self._kind = kind
        self._send_fn = send_fn
        self._poll_interval = float(poll_interval_s)
        self._batch_size = int(batch_size)
        self._on_error = on_error
        self._cursor = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── operation ────────────────────────────────────────────────────

    def drain_once(self) -> int:
        """Pull pending messages and call send_fn for each. Returns
        count drained. Idempotent — repeated calls advance the
        cursor and won't re-deliver."""
        msgs = self._kernel.mailbox.recv(
            pid=self._agent_pid,
            since_msg_id=self._cursor,
            limit=self._batch_size,
            mark_delivered=True,
        )
        count = 0
        for m in msgs:
            self._cursor = max(self._cursor, m.msg_id)
            if m.kind != MESSAGE_KIND:
                continue
            try:
                bridge_msg = BridgeMessage.from_payload(m.payload)
            except Exception as e:
                if self._on_error is not None:
                    try:
                        self._on_error(BridgeMessage(
                            kind=self._kind, sender="?", text="?",
                            direction="outbound", metadata={}, ts=0,
                        ), e)
                    except Exception:
                        pass
                continue
            if bridge_msg.direction != "outbound":
                continue
            try:
                self._send_fn(bridge_msg)
                count += 1
            except BaseException as e:
                if self._on_error is not None:
                    try:
                        self._on_error(bridge_msg, e)
                    except Exception:
                        pass
                # Don't propagate — one bad send shouldn't kill the
                # drainer.
        return count

    # ── background mode ──────────────────────────────────────────────

    def start(self) -> None:
        """Start the background drain loop. Idempotent — calling
        twice on a running receiver is a no-op."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"bridge-rx-{self._kind}-{self._agent_pid}",
        )
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Halt the background loop and wait for it to exit.
        Idempotent."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                drained = self.drain_once()
            except Exception:
                # Keep the loop alive on transient errors; on_error
                # is for per-message handling, not for drain-level
                # failures.
                drained = 0
            if drained == 0:
                self._stop_event.wait(timeout=self._poll_interval)


__all__ = [
    "BridgeKind",
    "BridgeMessage",
    "BridgeMirror",
    "OutboundReceiver",
    "MESSAGE_KIND",
    "inbound_topic",
    "outbound_topic",
]
