"""Alembic migrations — schema correctness + a drift guard.

Hermetic: upgrades a throwaway SQLite DB, so no API keys and no touching real data.
The drift check fails if the models in app/persistence/models.py have diverged from the
migrations (e.g. a column was added without a migration).
"""

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

import app.config as config_mod


def _cfg():
    cfg = Config("alembic.ini")
    cfg.attributes["configure_logging"] = False  # don't hijack pytest's logging
    return cfg


def test_upgrade_builds_expected_schema(monkeypatch, tmp_path):
    url = f"sqlite:///{(tmp_path / 'mig.db').as_posix()}"
    monkeypatch.setattr(config_mod.settings, "DATABASE_URL", url)

    command.upgrade(_cfg(), "head")

    insp = inspect(create_engine(url))
    tables = set(insp.get_table_names())
    assert {"sessions", "documents", "alembic_version"} <= tables

    doc_cols = {c["name"] for c in insp.get_columns("documents")}
    assert {"id", "session_id", "file_name", "chunk_count", "status", "created_at"} == doc_cols
    assert any(ix["column_names"] == ["session_id"] for ix in insp.get_indexes("documents"))


def test_no_model_migration_drift(monkeypatch, tmp_path):
    url = f"sqlite:///{(tmp_path / 'drift.db').as_posix()}"
    monkeypatch.setattr(config_mod.settings, "DATABASE_URL", url)

    cfg = _cfg()
    command.upgrade(cfg, "head")
    # Raises if Base.metadata (the models) and the migration head disagree.
    command.check(cfg)


def test_downgrade_reverts(monkeypatch, tmp_path):
    url = f"sqlite:///{(tmp_path / 'down.db').as_posix()}"
    monkeypatch.setattr(config_mod.settings, "DATABASE_URL", url)

    cfg = _cfg()
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    tables = set(inspect(create_engine(url)).get_table_names())
    assert "sessions" not in tables and "documents" not in tables
