"""Relational database engine + session factory.

SQLAlchemy against ``settings.DATABASE_URL``. Local dev/test defaults to SQLite;
deploy points it at Neon Postgres (deployment.md).

Schema management: ``init_db`` (create_all) is the zero-setup bootstrap for local dev
and tests — it only ever CREATES missing tables, never alters existing ones. Evolving
the schema (adding/altering columns on the live DB) is Alembic's job: the migrations in
``alembic/`` are the source of truth for changes. See ``alembic/README.md``. A CI test
(``tests/test_migrations.py``) fails if the models drift from the migrations.
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
