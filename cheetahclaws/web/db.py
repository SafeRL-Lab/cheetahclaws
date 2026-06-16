"""SQLite-backed persistence for the CheetahClaws web UI.

Single source of truth for the SQLAlchemy engine + session factory + a tiny
repository layer. Keeping CRUD here means the rest of the web package only
imports `db.repo`, never SQLAlchemy directly.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

try:
    from sqlalchemy import create_engine, desc, select, func
    from sqlalchemy.orm import Session, sessionmaker
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SQLAlchemy is required for the web UI. "
        "Install it with: pip install 'cheetahclaws[web]'"
    ) from exc

from cheetahclaws.web.models import (
    ApiCredential, Base, ChatSessionRow, Folder, Message, User,
)


# ── Engine / session factory ─────────────────────────────────────────────

DEFAULT_DB_PATH = Path.home() / ".cheetahclaws" / "web.db"

_engine = None
_SessionLocal: Optional[sessionmaker] = None
_init_lock = threading.Lock()


def _ensure_db_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def init_db(db_path: Optional[Path] = None) -> None:
    """Create the SQLite file and tables if missing. Idempotent."""
    global _engine, _SessionLocal
    with _init_lock:
        if _engine is not None:
            return
        path = _ensure_db_path(Path(db_path or
                                    os.environ.get("CHEETAHCLAWS_WEB_DB",
                                                   str(DEFAULT_DB_PATH))))
        # check_same_thread=False — we use SQLAlchemy's pool which serializes
        # access; many threads need to share the connection.
        _engine = create_engine(
            f"sqlite:///{path}",
            connect_args={"check_same_thread": False},
            future=True,
        )
        Base.metadata.create_all(_engine)
        # Light-touch migration for existing DBs that predate folders:
        # add chat_sessions.folder_id if missing. SQLite ALTER TABLE is
        # limited but ADD COLUMN with NULL default works fine.
        from sqlalchemy import text
        with _engine.begin() as conn:
            cols = {row[1] for row in conn.exec_driver_sql(
                "PRAGMA table_info(chat_sessions)"
            ).fetchall()}
            if "folder_id" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE chat_sessions ADD COLUMN folder_id INTEGER"
                    " REFERENCES folders(id) ON DELETE SET NULL"
                )
                conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS ix_chat_sessions_folder_id"
                    " ON chat_sessions(folder_id)"
                )
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False,
                                     expire_on_commit=False, future=True)
        # Tighten file permissions — the DB now holds password hashes & API keys.
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


@contextmanager
def session_scope() -> Iterator[Session]:
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    db: Session = _SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Repository ───────────────────────────────────────────────────────────
# Thin functions returning plain dicts so callers don't hold ORM objects
# across session boundaries (avoids DetachedInstanceError).

class repo:
    """Namespace for CRUD helpers."""

    # ── Users ──────────────────────────────────────────────────────────

    @staticmethod
    def user_count() -> int:
        with session_scope() as db:
            return db.scalar(select(func.count(User.id))) or 0

    @staticmethod
    def get_user_by_username(username: str) -> Optional[dict]:
        with session_scope() as db:
            u = db.scalar(select(User).where(User.username == username))
            if not u:
                return None
            return {"id": u.id, "username": u.username,
                    "password_hash": u.password_hash, "is_admin": u.is_admin,
                    "created_at": u.created_at}

    @staticmethod
    def get_user(user_id: int) -> Optional[dict]:
        with session_scope() as db:
            u = db.get(User, user_id)
            if not u:
                return None
            return {"id": u.id, "username": u.username,
                    "is_admin": u.is_admin, "created_at": u.created_at}

    @staticmethod
    def create_user(username: str, password_hash: str,
                    is_admin: bool = False) -> dict:
        with session_scope() as db:
            u = User(username=username, password_hash=password_hash,
                     is_admin=is_admin)
            db.add(u)
            db.flush()
            return {"id": u.id, "username": u.username,
                    "is_admin": u.is_admin, "created_at": u.created_at}

    # ── Chat sessions ──────────────────────────────────────────────────

    @staticmethod
    def upsert_session(session_id: str, user_id: int, *,
                       title: Optional[str] = None,
                       config: Optional[dict] = None) -> None:
        """Create or update a chat session row. Updates last_active."""
        with session_scope() as db:
            row = db.get(ChatSessionRow, session_id)
            if row is None:
                row = ChatSessionRow(
                    id=session_id, user_id=user_id,
                    title=title or "New chat",
                    config_json=json.dumps(config or {}),
                )
                db.add(row)
            else:
                if title is not None:
                    row.title = title
                if config is not None:
                    row.config_json = json.dumps(config)
                row.last_active = time.time()

    @staticmethod
    def list_sessions(user_id: int) -> list[dict]:
        with session_scope() as db:
            rows = db.execute(
                select(
                    ChatSessionRow,
                    func.count(Message.id).label("msg_count"),
                )
                .outerjoin(Message, Message.session_id == ChatSessionRow.id)
                .where(ChatSessionRow.user_id == user_id)
                .group_by(ChatSessionRow.id)
                .order_by(desc(ChatSessionRow.last_active))
            ).all()
            return [
                {
                    "id": r.ChatSessionRow.id,
                    "title": r.ChatSessionRow.title,
                    "created_at": r.ChatSessionRow.created_at,
                    "last_active": r.ChatSessionRow.last_active,
                    "message_count": int(r.msg_count or 0),
                    "folder_id": r.ChatSessionRow.folder_id,
                }
                for r in rows
            ]

    @staticmethod
    def get_session(session_id: str, user_id: int) -> Optional[dict]:
        with session_scope() as db:
            row = db.get(ChatSessionRow, session_id)
            if not row or row.user_id != user_id:
                return None
            try:
                cfg = json.loads(row.config_json or "{}")
            except json.JSONDecodeError:
                cfg = {}
            return {
                "id": row.id, "title": row.title,
                "user_id": row.user_id,
                "created_at": row.created_at,
                "last_active": row.last_active,
                "config": cfg,
            }

    @staticmethod
    def rename_session(session_id: str, user_id: int, title: str) -> bool:
        with session_scope() as db:
            row = db.get(ChatSessionRow, session_id)
            if not row or row.user_id != user_id:
                return False
            row.title = title.strip()[:200] or "Untitled"
            return True

    @staticmethod
    def delete_session(session_id: str, user_id: int) -> bool:
        with session_scope() as db:
            row = db.get(ChatSessionRow, session_id)
            if not row or row.user_id != user_id:
                return False
            db.delete(row)
            return True

    @staticmethod
    def touch_session(session_id: str) -> None:
        with session_scope() as db:
            row = db.get(ChatSessionRow, session_id)
            if row:
                row.last_active = time.time()

    @staticmethod
    def move_session_to_folder(session_id: str, user_id: int,
                                folder_id: Optional[int]) -> bool:
        """Set or clear a session's folder. None means ungrouped.

        Verifies the folder (when given) belongs to the same user — silently
        rejects cross-user moves the same way other ownership checks do.
        """
        with session_scope() as db:
            row = db.get(ChatSessionRow, session_id)
            if not row or row.user_id != user_id:
                return False
            if folder_id is not None:
                fld = db.get(Folder, folder_id)
                if not fld or fld.user_id != user_id:
                    return False
            row.folder_id = folder_id
            return True

    # ── Folders ────────────────────────────────────────────────────────

    @staticmethod
    def list_folders(user_id: int) -> list[dict]:
        with session_scope() as db:
            rows = db.execute(
                select(
                    Folder,
                    func.count(ChatSessionRow.id).label("sess_count"),
                )
                .outerjoin(ChatSessionRow,
                            ChatSessionRow.folder_id == Folder.id)
                .where(Folder.user_id == user_id)
                .group_by(Folder.id)
                .order_by(Folder.created_at)
            ).all()
            return [
                {
                    "id": r.Folder.id,
                    "name": r.Folder.name,
                    "created_at": r.Folder.created_at,
                    "session_count": int(r.sess_count or 0),
                }
                for r in rows
            ]

    @staticmethod
    def create_folder(user_id: int, name: str) -> Optional[dict]:
        """Create a folder. Returns None if the name already exists for
        this user (UniqueConstraint violation)."""
        from sqlalchemy.exc import IntegrityError
        name = (name or "").strip()[:120]
        if not name:
            return None
        with session_scope() as db:
            f = Folder(user_id=user_id, name=name)
            db.add(f)
            try:
                db.flush()
            except IntegrityError:
                db.rollback()
                return None
            return {"id": f.id, "name": f.name,
                    "created_at": f.created_at, "session_count": 0}

    @staticmethod
    def rename_folder(folder_id: int, user_id: int, name: str) -> bool:
        from sqlalchemy.exc import IntegrityError
        name = (name or "").strip()[:120]
        if not name:
            return False
        with session_scope() as db:
            f = db.get(Folder, folder_id)
            if not f or f.user_id != user_id:
                return False
            f.name = name
            try:
                db.flush()
            except IntegrityError:
                db.rollback()
                return False
            return True

    @staticmethod
    def delete_folder(folder_id: int, user_id: int) -> bool:
        """Delete a folder. Sessions inside it are preserved and become
        ungrouped. We NULL out folder_id explicitly because SQLite's
        PRAGMA foreign_keys is off in this engine, so the ON DELETE SET NULL
        wouldn't fire on its own."""
        from sqlalchemy import update
        with session_scope() as db:
            f = db.get(Folder, folder_id)
            if not f or f.user_id != user_id:
                return False
            db.execute(
                update(ChatSessionRow)
                .where(ChatSessionRow.folder_id == folder_id,
                       ChatSessionRow.user_id == user_id)
                .values(folder_id=None)
            )
            db.delete(f)
            return True

    # ── Messages ───────────────────────────────────────────────────────

    @staticmethod
    def append_message(session_id: str, role: str, content: str,
                       tool_calls: Optional[list] = None) -> int:
        with session_scope() as db:
            m = Message(
                session_id=session_id,
                role=role,
                content=content,
                tool_calls_json=json.dumps(tool_calls) if tool_calls else None,
            )
            db.add(m)
            row = db.get(ChatSessionRow, session_id)
            if row:
                row.last_active = time.time()
                # Auto-title from first user message
                if (role == "user" and row.title in ("New chat", "Untitled")
                        and content and not content.startswith("/")):
                    row.title = content.strip().splitlines()[0][:60]
            db.flush()
            return m.id

    @staticmethod
    def get_messages(session_id: str) -> list[dict]:
        with session_scope() as db:
            rows = db.scalars(
                select(Message).where(Message.session_id == session_id)
                .order_by(Message.id)
            ).all()
            out: list[dict] = []
            for m in rows:
                d = {"role": m.role, "content": m.content,
                     "created_at": m.created_at}
                if m.tool_calls_json:
                    try:
                        d["tool_calls"] = json.loads(m.tool_calls_json)
                    except json.JSONDecodeError:
                        pass
                out.append(d)
            return out

    # ── API credentials ────────────────────────────────────────────────

    @staticmethod
    def set_credential(user_id: int, provider: str, api_key: str) -> None:
        with session_scope() as db:
            existing = db.scalar(
                select(ApiCredential)
                .where(ApiCredential.user_id == user_id,
                       ApiCredential.provider == provider)
            )
            if existing:
                existing.api_key = api_key
            else:
                db.add(ApiCredential(user_id=user_id, provider=provider,
                                     api_key=api_key))

    @staticmethod
    def get_credentials(user_id: int) -> dict[str, str]:
        with session_scope() as db:
            rows = db.scalars(
                select(ApiCredential).where(ApiCredential.user_id == user_id)
            ).all()
            return {r.provider: r.api_key for r in rows}
