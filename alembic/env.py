"""Alembic environment for the Study Helper.

Wired to the app so migrations never drift from the code or the deployment:
- the URL comes from ``app.config.settings.DATABASE_URL`` (SQLite local / Neon prod),
- ``target_metadata`` is the app's ``Base.metadata`` (autogenerate sees the models),
- migrations are SCOPED to the tables SQLAlchemy owns. The same Postgres database also
  holds LangGraph checkpoint tables and pgvector tables, created by those libraries —
  Alembic must never manage or (on autogenerate) propose dropping them.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import settings
from app.persistence.db import Base

# Import models so every table registers on Base.metadata before autogenerate compares.
from app.persistence import models  # noqa: F401

config = context.config

# Feed the app's DB URL to Alembic (kept out of alembic.ini so no secret is committed).
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Apply the ini's logging config for CLI use, but let a programmatic caller opt out
# (cfg.attributes["configure_logging"] = False) so running a migration in-process can't
# hijack the app's / pytest's logging.
if config.config_file_name is not None and config.attributes.get("configure_logging", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# The only tables Alembic is allowed to manage. Guards against autogenerate proposing to
# drop the checkpoint / pgvector / mlflow tables that share the same database.
_MANAGED_TABLES = set(target_metadata.tables.keys())


def _include_name(name, type_, parent_names):
    if type_ == "table":
        # Keep None (the alembic_version bookkeeping table) and our own tables.
        return name is None or name in _MANAGED_TABLES
    return True


def _configure(connection=None, url=None):
    context.configure(
        connection=connection,
        url=url,
        target_metadata=target_metadata,
        include_name=_include_name,
        compare_type=True,
        # SQLite can't ALTER most columns in place; batch mode rebuilds the table so
        # future column changes work on the local backend too.
        render_as_batch=settings.DATABASE_URL.startswith("sqlite"),
    )


def run_migrations_offline() -> None:
    _configure(url=settings.DATABASE_URL)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
