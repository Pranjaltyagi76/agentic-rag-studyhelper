"""Relational database engine + session factory.

SQLAlchemy against ``settings.DATABASE_URL``. Local dev/test defaults to SQLite;
deploy points it at Neon Postgres (deployment.md). ``init_db`` creates tables on
startup (Alembic migrations are a Phase 9 concern).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

# SQLite + FastAPI's threadpool needs check_same_thread=False; harmless elsewhere.
_connect_args = (
    {"check_same_thread": False}
    if settings.DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


def init_db() -> None:
    """Create tables if they do not exist. Called on app startup."""
    # Import models so they register on Base.metadata before create_all.
    from app.persistence import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
