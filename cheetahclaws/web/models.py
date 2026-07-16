"""SQLAlchemy ORM models for the CheetahClaws web UI.

Schema:
    users           — login accounts (bcrypt password hash)
    chat_sessions   — agent conversations (owned by a user)
    messages        — turn-by-turn message history within a session
    api_credentials — per-user provider API keys (stored opaquely; see notes)

All timestamps are stored as Unix epoch seconds (Float) so the DB layer stays
free of timezone gotchas; the UI converts to local time client-side.
"""
from __future__ import annotations

import time

try:
    from sqlalchemy import (
        Boolean, Float, ForeignKey, Integer, String, Text,
        UniqueConstraint,
    )
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SQLAlchemy is required for the web UI. "
        "Install it with: pip install 'cheetahclaws[web]'"
    ) from exc


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True,
                                          nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[float] = mapped_column(Float, default=time.time,
                                              nullable=False)

    sessions: Mapped[list["ChatSessionRow"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    credentials: Mapped[list["ApiCredential"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Folder(Base):
    """User-scoped folder for grouping chat sessions (flat hierarchy)."""
    __tablename__ = "folders"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_user_folder_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True, nullable=False,
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[float] = mapped_column(Float, default=time.time,
                                              nullable=False)

    sessions: Mapped[list["ChatSessionRow"]] = relationship(
        back_populates="folder"
    )


class ChatSessionRow(Base):
    """Persistent metadata for a chat session.

    The runtime ChatSession (api.py) hydrates from this on demand and
    writes back when messages or config change.
    """
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"),
                                          index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(200), default="New chat", nullable=False)
    created_at: Mapped[float] = mapped_column(Float, default=time.time,
                                              nullable=False)
    last_active: Mapped[float] = mapped_column(Float, default=time.time,
                                               nullable=False, index=True)
    config_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    folder_id: Mapped[int | None] = mapped_column(
        ForeignKey("folders.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )

    user: Mapped[User] = relationship(back_populates="sessions")
    folder: Mapped["Folder | None"] = relationship(back_populates="sessions")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        index=True, nullable=False,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    tool_calls_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time,
                                              nullable=False)

    session: Mapped[ChatSessionRow] = relationship(back_populates="messages")


class ApiCredential(Base):
    """Per-user provider API key. Stored opaquely (not encrypted at rest yet).

    Treat the SQLite file as sensitive — restrict file mode to 0600 in db.py.
    A future iteration should add AES-GCM encryption with a key derived from
    a user-supplied passphrase.
    """
    __tablename__ = "api_credentials"
    __table_args__ = (
        UniqueConstraint("user_id", "provider", name="uq_user_provider"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"),
                                          nullable=False)
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    api_key: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[float] = mapped_column(Float, default=time.time,
                                              nullable=False)

    user: Mapped[User] = relationship(back_populates="credentials")
