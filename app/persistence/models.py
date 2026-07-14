"""Relational models: sessions and documents (Phase 2).

These give per-user isolation a durable home: which files belong to which session.
The ``messages`` table and the LangGraph checkpoint tables arrive in Phase 5 with
the checkpointer (design.md section 7).
"""

from datetime import datetime, timezone

from sqlalchemy import String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.persistence.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SessionRow(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    documents: Mapped[list["DocumentRow"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("sessions.session_id"), index=True
    )
    file_name: Mapped[str] = mapped_column(String)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String, default="ingested")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped["SessionRow"] = relationship(back_populates="documents")
