# Database migrations (Alembic)

Alembic manages **changes** to the relational schema (the `sessions` and `documents`
tables). It is wired to the app in [`env.py`](env.py): the database URL comes from
`app.config.settings.DATABASE_URL` (SQLite locally / Neon Postgres in prod), and
autogenerate compares against the app's `Base.metadata`.

## How this coexists with `init_db()`

`app/persistence/db.py:init_db()` still runs `create_all` at startup. That's the
zero-setup bootstrap for local dev and tests — it only ever **creates missing tables**,
it never alters existing ones. So any change to an existing table (a new column, a type
change) MUST go through a migration; `create_all` won't apply it.

Only the two SQLAlchemy-owned tables are managed. The same Neon database also holds
LangGraph checkpoint tables and pgvector tables, created by those libraries — `env.py`'s
`include_name` filter keeps Alembic from ever touching or dropping them.

## One-time adoption on an existing database

The live Neon DB and any local `studyhelper.db` were created by `create_all` and have no
`alembic_version` row yet. Adopt the baseline **without recreating anything**:

```bash
alembic stamp head
```

A brand-new/empty database instead gets the schema built by migrations:

```bash
alembic upgrade head
```

## Making a schema change

1. Edit the models in `app/persistence/models.py`.
2. Autogenerate a migration and review it (autogenerate is a draft, not gospel):
   ```bash
   alembic revision --autogenerate -m "add documents.page_count"
   ```
3. Apply it locally, then on prod (point `DATABASE_URL` at Neon):
   ```bash
   alembic upgrade head
   ```

`alembic downgrade -1` reverts the last migration. `render_as_batch` is on for SQLite so
column changes work on the local backend too.

## Drift guard

`tests/test_migrations.py` upgrades a throwaway SQLite DB to `head` and runs
`alembic check`; it fails if the models and migrations have diverged (e.g. a model was
changed without a migration). It needs no API keys and runs in CI.
