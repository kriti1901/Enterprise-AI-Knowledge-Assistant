"""SQLAlchemy models and session management."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from app.config import get_settings


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(20), default="user")  # "admin" | "user"
    departments: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    filename: Mapped[str] = mapped_column(String(300))
    department: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|indexed|failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_by: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    chunks: Mapped[list[Chunk]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    department: Mapped[str] = mapped_column(String(100), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    page: Mapped[int] = mapped_column(Integer, default=1)
    text: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.id"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    metrics: Mapped[dict] = mapped_column(JSON)


@lru_cache
def get_engine() -> Engine:
    url = get_settings().database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def new_session() -> Session:
    """Session for code outside request handling (startup, scripts, evaluation)."""
    return _session_factory()()


def get_session() -> Iterator[Session]:
    session = new_session()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    Base.metadata.create_all(get_engine())
