"""mailbox.py — AgentMailbox (RFC 0009).

Direct messaging + topic pub/sub between agents. Push fan-out for
pub/sub (one row per subscriber per publish) — simple, correct,
fine for v1's expected fan-out scale. Pull/cursor model can land in a
follow-up RFC if a high-fanout scenario emerges.

Strictly additive — nothing else in the codebase imports this yet.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .errors import (
    MailboxAlreadyExists,
    MailboxFull,
    MailboxInvalidPayload,
    MailboxNotFound,
    MailboxSubscriptionMissing,
    UnknownPid,
)

if TYPE_CHECKING:
    from cheetahclaws.daemon.rpc import CallContext, RpcRegistry


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Mailbox:
    pid:          int
    queue_size:   int
    retention_s:  float
    created_at:   float

    def to_dict(self) -> dict:
        return {
            "pid":         self.pid,
            "queue_size":  self.queue_size,
            "retention_s": self.retention_s,
            "created_at":  self.created_at,
        }


@dataclass(frozen=True)
class Message:
    msg_id:        int
    sender_pid:    Optional[int]
    recipient_pid: int
    topic:         Optional[str]
    kind:          str
    payload:       dict
    posted_at:     float
    delivered_at:  Optional[float]
    expires_at:    Optional[float]

    def to_dict(self) -> dict:
        return {
            "msg_id":        self.msg_id,
            "sender_pid":    self.sender_pid,
            "recipient_pid": self.recipient_pid,
            "topic":         self.topic,
            "kind":          self.kind,
            "payload":       self.payload,
            "posted_at":     self.posted_at,
            "delivered_at":  self.delivered_at,
            "expires_at":    self.expires_at,
        }


@dataclass(frozen=True)
class Subscription:
    pid:        int
    topic:      str
    created_at: float


def _row_to_msg(row: sqlite3.Row) -> Message:
    payload_raw = row["payload"]
    try:
        payload = json.loads(payload_raw) if payload_raw else {}
    except json.JSONDecodeError:
        payload = {"_raw": payload_raw}
    return Message(
        msg_id        = row["msg_id"],
        sender_pid    = row["sender_pid"],
        recipient_pid = row["recipient_pid"],
        topic         = row["topic"],
        kind          = row["kind"],
        payload       = payload,
        posted_at     = row["posted_at"],
        delivered_at  = row["delivered_at"],
        expires_at    = row["expires_at"],
    )


# ── Store ──────────────────────────────────────────────────────────────────


class MailboxStore:
    """SQLite-backed mailbox sharing kernel.db conn + write lock."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        write_lock: Optional[threading.Lock] = None,
    ) -> None:
        self._conn = conn
        self._lock = write_lock or threading.Lock()

    # ── Mailbox CRUD ──────────────────────────────────────────────────

    def create(
        self,
        *,
        pid: int,
        queue_size: int = 1024,
        retention_s: float = 3600.0,
    ) -> Mailbox:
        if not isinstance(pid, int):
            raise MailboxInvalidPayload("pid must be int", field="pid")
        if not isinstance(queue_size, int) or queue_size < 1:
            raise MailboxInvalidPayload(
                "queue_size must be a positive int",
                field="queue_size",
            )
        if not isinstance(retention_s, (int, float)) or retention_s < 0:
            raise MailboxInvalidPayload(
                "retention_s must be >= 0", field="retention_s",
            )
        now = time.time()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT pid FROM agent_processes WHERE pid = ?", (pid,),
                ).fetchone()
                if row is None:
                    raise UnknownPid(pid)
                exists = self._conn.execute(
                    "SELECT 1 FROM agent_mailboxes WHERE pid = ?", (pid,),
                ).fetchone()
                if exists:
                    raise MailboxAlreadyExists(pid)
                self._conn.execute(
                    "INSERT INTO agent_mailboxes "
                    "(pid, queue_size, retention_s, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (pid, queue_size, float(retention_s), now),
                )
        return Mailbox(pid=pid, queue_size=queue_size,
                       retention_s=float(retention_s), created_at=now)

    def get(self, pid: int) -> Mailbox:
        row = self._conn.execute(
            "SELECT * FROM agent_mailboxes WHERE pid = ?", (pid,),
        ).fetchone()
        if row is None:
            raise MailboxNotFound(pid)
        return Mailbox(
            pid=row["pid"], queue_size=int(row["queue_size"]),
            retention_s=float(row["retention_s"]),
            created_at=float(row["created_at"]),
        )

    def delete(self, pid: int) -> int:
        """Drop mailbox + subscriptions + messages for pid. Returns count
        of messages purged."""
        if not isinstance(pid, int):
            raise MailboxInvalidPayload("pid must be int", field="pid")
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT 1 FROM agent_mailboxes WHERE pid = ?", (pid,),
                ).fetchone()
                if row is None:
                    raise MailboxNotFound(pid)
                cnt_row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM agent_messages "
                    "WHERE recipient_pid = ?", (pid,),
                ).fetchone()
                purged = int(cnt_row["n"])
                self._conn.execute(
                    "DELETE FROM agent_messages WHERE recipient_pid = ?",
                    (pid,),
                )
                self._conn.execute(
                    "DELETE FROM agent_subscriptions WHERE pid = ?", (pid,),
                )
                self._conn.execute(
                    "DELETE FROM agent_mailboxes WHERE pid = ?", (pid,),
                )
        return purged

    # ── Subscriptions ─────────────────────────────────────────────────

    def subscribe(self, pid: int, topic: str) -> None:
        if not isinstance(pid, int):
            raise MailboxInvalidPayload("pid must be int", field="pid")
        if not isinstance(topic, str) or not topic:
            raise MailboxInvalidPayload("topic must be non-empty str",
                                        field="topic")
        now = time.time()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT 1 FROM agent_mailboxes WHERE pid = ?", (pid,),
                ).fetchone()
                if row is None:
                    raise MailboxNotFound(pid)
                # Idempotent: INSERT OR IGNORE.
                self._conn.execute(
                    "INSERT OR IGNORE INTO agent_subscriptions "
                    "(pid, topic, created_at) VALUES (?, ?, ?)",
                    (pid, topic, now),
                )

    def unsubscribe(self, pid: int, topic: str) -> None:
        if not isinstance(pid, int):
            raise MailboxInvalidPayload("pid must be int", field="pid")
        if not isinstance(topic, str) or not topic:
            raise MailboxInvalidPayload("topic must be non-empty str",
                                        field="topic")
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    "DELETE FROM agent_subscriptions "
                    "WHERE pid = ? AND topic = ?",
                    (pid, topic),
                )
                if cur.rowcount == 0:
                    raise MailboxSubscriptionMissing(pid, topic)

    def list_subscriptions(self, pid: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT topic FROM agent_subscriptions WHERE pid = ? "
            "ORDER BY topic ASC",
            (pid,),
        ).fetchall()
        return [r["topic"] for r in rows]

    def list_subscribers(self, topic: str) -> list[int]:
        rows = self._conn.execute(
            "SELECT pid FROM agent_subscriptions WHERE topic = ? "
            "ORDER BY pid ASC",
            (topic,),
        ).fetchall()
        return [r["pid"] for r in rows]

    # ── Send (direct) ─────────────────────────────────────────────────

    def send(
        self,
        *,
        sender_pid: Optional[int],
        recipient_pid: int,
        kind: str,
        payload: dict,
        expires_at: Optional[float] = None,
    ) -> int:
        if not isinstance(recipient_pid, int):
            raise MailboxInvalidPayload("recipient_pid must be int",
                                        field="recipient_pid")
        if sender_pid is not None and not isinstance(sender_pid, int):
            raise MailboxInvalidPayload("sender_pid must be int or null",
                                        field="sender_pid")
        if not isinstance(kind, str) or not kind:
            raise MailboxInvalidPayload("kind must be non-empty str",
                                        field="kind")
        if not isinstance(payload, dict):
            raise MailboxInvalidPayload("payload must be an object",
                                        field="payload")
        if expires_at is not None and (
            not isinstance(expires_at, (int, float)) or expires_at <= 0
        ):
            raise MailboxInvalidPayload("expires_at must be > 0 or null",
                                        field="expires_at")
        now = time.time()
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self._conn:
                mb = self._conn.execute(
                    "SELECT queue_size FROM agent_mailboxes WHERE pid = ?",
                    (recipient_pid,),
                ).fetchone()
                if mb is None:
                    raise MailboxNotFound(recipient_pid)
                queue_size = int(mb["queue_size"])
                pending = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM agent_messages "
                    "WHERE recipient_pid = ? AND delivered_at IS NULL",
                    (recipient_pid,),
                ).fetchone()
                if int(pending["n"]) >= queue_size:
                    raise MailboxFull(recipient_pid, queue_size)
                cur = self._conn.execute(
                    "INSERT INTO agent_messages "
                    "(sender_pid, recipient_pid, topic, kind, payload, "
                    " posted_at, expires_at) "
                    "VALUES (?, ?, NULL, ?, ?, ?, ?)",
                    (sender_pid, recipient_pid, kind, payload_json,
                     now, expires_at),
                )
                msg_id = int(cur.lastrowid)
        return msg_id

    # ── Publish (pub/sub fan-out) ─────────────────────────────────────

    def publish(
        self,
        *,
        sender_pid: Optional[int],
        topic: str,
        kind: str,
        payload: dict,
        expires_at: Optional[float] = None,
        fail_on_full: bool = False,
    ) -> dict:
        if not isinstance(topic, str) or not topic:
            raise MailboxInvalidPayload("topic must be non-empty str",
                                        field="topic")
        if not isinstance(kind, str) or not kind:
            raise MailboxInvalidPayload("kind must be non-empty str",
                                        field="kind")
        if not isinstance(payload, dict):
            raise MailboxInvalidPayload("payload must be an object",
                                        field="payload")
        if expires_at is not None and (
            not isinstance(expires_at, (int, float)) or expires_at <= 0
        ):
            raise MailboxInvalidPayload("expires_at must be > 0 or null",
                                        field="expires_at")
        now = time.time()
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        delivered: list[int] = []
        rejected = 0

        with self._lock:
            with self._conn:
                # Find subscribers.
                rows = self._conn.execute(
                    "SELECT s.pid AS pid, m.queue_size AS queue_size "
                    "FROM agent_subscriptions s "
                    "JOIN agent_mailboxes m ON m.pid = s.pid "
                    "WHERE s.topic = ?",
                    (topic,),
                ).fetchall()
                # Pre-check capacities under fail_on_full.
                if fail_on_full:
                    for r in rows:
                        pending = self._conn.execute(
                            "SELECT COUNT(*) AS n FROM agent_messages "
                            "WHERE recipient_pid = ? AND delivered_at IS NULL",
                            (r["pid"],),
                        ).fetchone()
                        if int(pending["n"]) >= int(r["queue_size"]):
                            raise MailboxFull(r["pid"], int(r["queue_size"]))
                # Insert per-subscriber.
                for r in rows:
                    pid = int(r["pid"])
                    qs = int(r["queue_size"])
                    pending = self._conn.execute(
                        "SELECT COUNT(*) AS n FROM agent_messages "
                        "WHERE recipient_pid = ? AND delivered_at IS NULL",
                        (pid,),
                    ).fetchone()
                    if int(pending["n"]) >= qs:
                        rejected += 1
                        continue
                    cur = self._conn.execute(
                        "INSERT INTO agent_messages "
                        "(sender_pid, recipient_pid, topic, kind, "
                        " payload, posted_at, expires_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (sender_pid, pid, topic, kind, payload_json,
                         now, expires_at),
                    )
                    delivered.append(int(cur.lastrowid))

        return {
            "delivered": len(delivered),
            "rejected":  rejected,
            "msg_ids":   delivered,
        }

    # ── Receive ───────────────────────────────────────────────────────

    def recv(
        self,
        *,
        pid: int,
        since_msg_id: int = 0,
        limit: int = 100,
        mark_delivered: bool = True,
        now: Optional[float] = None,
    ) -> list[Message]:
        if not isinstance(pid, int):
            raise MailboxInvalidPayload("pid must be int", field="pid")
        if not isinstance(since_msg_id, int) or since_msg_id < 0:
            since_msg_id = 0
        if not isinstance(limit, int) or limit < 1:
            limit = 100
        elif limit > 10_000:
            limit = 10_000
        if now is None:
            now = time.time()

        with self._lock:
            with self._conn:
                # Verify mailbox exists.
                mb_row = self._conn.execute(
                    "SELECT 1 FROM agent_mailboxes WHERE pid = ?", (pid,),
                ).fetchone()
                if mb_row is None:
                    raise MailboxNotFound(pid)
                rows = self._conn.execute(
                    "SELECT * FROM agent_messages "
                    "WHERE recipient_pid = ? "
                    "  AND msg_id > ? "
                    "  AND (expires_at IS NULL OR expires_at > ?) "
                    "ORDER BY msg_id ASC LIMIT ?",
                    (pid, since_msg_id, now, limit),
                ).fetchall()
                if mark_delivered and rows:
                    # Mark only those that aren't already delivered.
                    ids = [int(r["msg_id"]) for r in rows
                           if r["delivered_at"] is None]
                    if ids:
                        placeholders = ",".join("?" * len(ids))
                        self._conn.execute(
                            "UPDATE agent_messages SET delivered_at = ? "
                            f"WHERE msg_id IN ({placeholders})",
                            (now, *ids),
                        )
                # Re-read to capture post-update delivered_at.
                if mark_delivered and rows:
                    ids = [int(r["msg_id"]) for r in rows]
                    placeholders = ",".join("?" * len(ids))
                    rows = self._conn.execute(
                        f"SELECT * FROM agent_messages "
                        f"WHERE msg_id IN ({placeholders}) "
                        "ORDER BY msg_id ASC",
                        ids,
                    ).fetchall()
        return [_row_to_msg(r) for r in rows]

    def peek(
        self,
        *,
        pid: int,
        since_msg_id: int = 0,
        limit: int = 100,
        now: Optional[float] = None,
    ) -> list[Message]:
        return self.recv(
            pid=pid, since_msg_id=since_msg_id, limit=limit,
            mark_delivered=False, now=now,
        )

    # ── Garbage collection ────────────────────────────────────────────

    def gc_expired(self, now: Optional[float] = None) -> int:
        """Purge expired + delivered-past-retention messages. Returns
        count purged."""
        if now is None:
            now = time.time()
        purged = 0
        with self._lock:
            with self._conn:
                cur1 = self._conn.execute(
                    "DELETE FROM agent_messages "
                    "WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now,),
                )
                purged += int(cur1.rowcount or 0)
                # Per-mailbox retention purge — messages whose
                # delivered_at + retention_s < now.
                cur2 = self._conn.execute(
                    "DELETE FROM agent_messages "
                    "WHERE delivered_at IS NOT NULL "
                    "  AND EXISTS ("
                    "      SELECT 1 FROM agent_mailboxes m "
                    "      WHERE m.pid = agent_messages.recipient_pid "
                    "        AND agent_messages.delivered_at + m.retention_s < ?"
                    "  )",
                    (now,),
                )
                purged += int(cur2.rowcount or 0)
        return purged


# ── RPC registration ───────────────────────────────────────────────────────


def register(registry: "RpcRegistry", store: MailboxStore) -> None:
    from .errors import KernelError

    def _translate(fn):
        def wrapper(params, ctx):
            try:
                return fn(params, ctx)
            except MailboxInvalidPayload as e:
                raise TypeError(str(e))
            except KernelError as e:
                raise RuntimeError(f"{type(e).__name__}: {e}")
        wrapper.__name__ = fn.__name__
        return wrapper

    @_translate
    def mbox_create(params, ctx):
        mb = store.create(
            pid=_req_int(params, "pid"),
            queue_size=int(params.get("queue_size", 1024)),
            retention_s=float(params.get("retention_s", 3600.0)),
        )
        return {"pid": mb.pid}

    @_translate
    def mbox_delete(params, ctx):
        pid = _req_int(params, "pid")
        purged = store.delete(pid)
        return {"pid": pid, "removed_messages": purged}

    @_translate
    def mbox_subscribe(params, ctx):
        pid = _req_int(params, "pid")
        topic = _req_str(params, "topic")
        store.subscribe(pid, topic)
        return {"pid": pid, "topic": topic}

    @_translate
    def mbox_unsubscribe(params, ctx):
        pid = _req_int(params, "pid")
        topic = _req_str(params, "topic")
        store.unsubscribe(pid, topic)
        return {"pid": pid, "topic": topic}

    @_translate
    def mbox_list_subscriptions(params, ctx):
        return {"topics": store.list_subscriptions(_req_int(params, "pid"))}

    @_translate
    def mbox_send(params, ctx):
        sender = params.get("sender_pid")
        if sender is not None and not isinstance(sender, int):
            raise MailboxInvalidPayload("sender_pid must be int or null",
                                        field="sender_pid")
        msg_id = store.send(
            sender_pid=sender,
            recipient_pid=_req_int(params, "recipient_pid"),
            kind=_req_str(params, "kind"),
            payload=params.get("payload", {}) or {},
            expires_at=(None if params.get("expires_at") is None
                        else float(params["expires_at"])),
        )
        return {"msg_id": msg_id}

    @_translate
    def mbox_publish(params, ctx):
        sender = params.get("sender_pid")
        if sender is not None and not isinstance(sender, int):
            raise MailboxInvalidPayload("sender_pid must be int or null",
                                        field="sender_pid")
        return store.publish(
            sender_pid=sender,
            topic=_req_str(params, "topic"),
            kind=_req_str(params, "kind"),
            payload=params.get("payload", {}) or {},
            expires_at=(None if params.get("expires_at") is None
                        else float(params["expires_at"])),
            fail_on_full=bool(params.get("fail_on_full", False)),
        )

    @_translate
    def mbox_recv(params, ctx):
        msgs = store.recv(
            pid=_req_int(params, "pid"),
            since_msg_id=int(params.get("since_msg_id", 0)),
            limit=int(params.get("limit", 100)),
            mark_delivered=bool(params.get("mark_delivered", True)),
        )
        next_cursor = msgs[-1].msg_id if msgs else int(params.get("since_msg_id", 0))
        return {"messages": [m.to_dict() for m in msgs],
                "next_cursor": next_cursor}

    @_translate
    def mbox_peek(params, ctx):
        msgs = store.peek(
            pid=_req_int(params, "pid"),
            since_msg_id=int(params.get("since_msg_id", 0)),
            limit=int(params.get("limit", 100)),
        )
        return {"messages": [m.to_dict() for m in msgs]}

    @_translate
    def mbox_gc_expired(params, ctx):
        now = params.get("now")
        if now is not None:
            now = float(now)
        return {"purged": store.gc_expired(now=now)}

    registry.register("kernel.mbox.create",             mbox_create)
    registry.register("kernel.mbox.delete",             mbox_delete)
    registry.register("kernel.mbox.subscribe",          mbox_subscribe)
    registry.register("kernel.mbox.unsubscribe",        mbox_unsubscribe)
    registry.register("kernel.mbox.list_subscriptions", mbox_list_subscriptions)
    registry.register("kernel.mbox.send",               mbox_send)
    registry.register("kernel.mbox.publish",            mbox_publish)
    registry.register("kernel.mbox.recv",               mbox_recv)
    registry.register("kernel.mbox.peek",               mbox_peek)
    registry.register("kernel.mbox.gc_expired",         mbox_gc_expired)


def _req_int(params: dict, key: str) -> int:
    if key not in params:
        raise MailboxInvalidPayload(f"missing {key!r}", field=key)
    v = params[key]
    if not isinstance(v, int):
        raise MailboxInvalidPayload(f"{key!r} must be int", field=key)
    return v


def _req_str(params: dict, key: str) -> str:
    if key not in params:
        raise MailboxInvalidPayload(f"missing {key!r}", field=key)
    v = params[key]
    if not isinstance(v, str):
        raise MailboxInvalidPayload(f"{key!r} must be str", field=key)
    return v
