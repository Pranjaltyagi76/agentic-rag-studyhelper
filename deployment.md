# Deployment — Advanced Agentic RAG Study Helper

> How the system is containerized, configured, and shipped. Realizes NFR-4
> (portability) and NFR-5 (configurability), plus roadmap Phases 7 & 9.
>
> **Guiding constraint: zero budget — everything runs on free tiers.** See the
> [strategy.md](strategy.md) decision log. Where a "better" paid option exists, the
> free option wins unless the user says otherwise.

---

## 0. Free-tier stack (locked)

| Layer | Service | Free? | Docker? | Trade-off accepted |
|---|---|---|---|---|
| **App** | **Render** (free web service, Docker) | ✅ no card | ✅ builds our Dockerfile | Spins down after 15 min idle (~1 min cold start); 750 hrs/mo |
| **Postgres + vectors** | **Neon** (Postgres + `pgvector`) | ✅ non-expiring | ❌ managed | ~0.5 GB storage cap |
| **Observability** | **LangSmith** | ✅ free tier | — | Trace quota limits |
| **Evaluation** | **MLflow** (run locally) | ✅ | ✅ local | Not hosted; run on demand |

Rejected: **Render** — its free Postgres is **deleted after 30 days** and persistent
disks are **paid-only**, so a durable free deploy isn't possible there.

---

## 1. Runtime topology

```
┌──────────────────────────────────────────────────────┐
│ Hugging Face Space (Docker, free)                     │
│  ┌───────────┐                                        │
│  │  app      │  FastAPI + LangGraph agent (uvicorn)   │
│  │ (Docker)  │                                        │
│  └─────┬─────┘                                         │
└────────┼──────────────────────────────────────────────┘
         │ (egress)
         ├───────────────► Neon (managed Postgres + pgvector)
         │                  · sessions, messages, documents
         │                  · LangGraph checkpoints
         │                  · embeddings (pgvector)
         │
         └───────────────► Groq · Google · Tavily · LangSmith APIs
```

- **app**: FastAPI + LangGraph agent, served by uvicorn. Built from our Dockerfile
  by HF Spaces.
- **Neon**: single managed datastore for **both** relational state **and** vectors
  (via `pgvector`). No persistent disk needed → free tier works end-to-end.
- **vectors**: `pgvector` is canonical. Chroma-on-disk is kept only as a *local dev*
  convenience, never the deployed backend (a persistent disk costs money).

---

## 2. Configuration (env only — never commit secrets)

`.env.example` (committed):
```
# LLMs / tools
GROQ_API_KEY=
GOOGLE_API_KEY=
TAVILY_API_KEY=
HUGGINGFACEHUB_API_TOKEN=

# persistence (Neon free Postgres + pgvector)
DATABASE_URL=postgresql://<user>:<pass>@<neon-host>/<db>?sslmode=require
VECTOR_BACKEND=pgvector          # pgvector (deploy) | chroma (local dev only)
CHROMA_DIR=./Chromadb            # local dev only; unused when VECTOR_BACKEND=pgvector

# observability (LangSmith)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=studyhelper
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# evaluation (MLflow — local)
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=studyhelper-rag-eval

# app
APP_ENV=production
LOG_LEVEL=info
PORT=7860                        # HF Spaces expects the app on 7860
```

Rules:
- Real `.env` is git-ignored. On HF Spaces, secrets are set as **Space secrets**, not
  committed.
- No key/URL is hard-coded (current code loads via `dotenv` — keep that pattern).

---

## 3. Docker — where it's actually used

Docker is **not** wasted on the free plan. It has two jobs, both free:

1. **Local development** — `docker-compose up` runs app + Postgres/pgvector on your
   machine for a reproducible dev loop.
2. **The deploy image** — HF Spaces (Docker SDK) **builds and runs this exact
   Dockerfile**. Docker *is* the deploy artifact.

Docker is **not** used for the production database — Neon is managed. You only run a
Postgres container locally (for dev), never in the cloud.

### Dockerfile (multi-stage, sketch — port 7860 for HF Spaces)
```dockerfile
FROM python:3.11-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
EXPOSE 7860
HEALTHCHECK CMD curl -f http://localhost:7860/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### docker-compose.yml — LOCAL DEV ONLY (uses pgvector image, mirrors Neon)
```yaml
services:
  app:
    build: .
    env_file: .env
    ports: ["7860:7860"]
    depends_on: [postgres]
  postgres:
    image: pgvector/pgvector:pg16     # matches Neon's pgvector in prod
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: app
      POSTGRES_DB: studyhelper
    volumes: ["pgdata:/var/lib/postgresql/data"]
volumes: { pgdata: {} }
```

Using `pgvector/pgvector:pg16` locally means dev and prod (Neon) share the **same**
vector backend — no "works locally, breaks in prod" gap.

---

## 4. Database migrations
- Enable the extension once: `CREATE EXTENSION IF NOT EXISTS vector;` (Neon supports it).
- **Relational schema (`sessions`, `documents`) is managed by Alembic** — see
  [`alembic/README.md`](alembic/README.md). `init_db()`'s `create_all` still bootstraps
  missing tables on boot (idempotent), but column/table *changes* ship as migrations.
  - **One-time on the existing Neon DB** (created by `create_all`, not yet stamped):
    `alembic stamp head` — adopts the baseline without recreating anything.
  - Thereafter apply changes with `alembic upgrade head` (point `DATABASE_URL` at Neon).
- Checkpoint tables are created by LangGraph's `.setup()`; pgvector tables by PGVector.
  Alembic is scoped away from both, so it never touches them.

---

## 5. Deploy target (locked: Hugging Face Spaces + Neon)

**Primary (free):** Hugging Face Spaces (Docker) for the app + Neon for Postgres/pgvector.

Fallback free options (only if HF Spaces limits bite):

| Platform | Free? | Note |
|---|---|---|
| **HF Spaces (Docker)** ⭐ | ✅ | Chosen. Idle sleep, no expiring DB. |
| Fly.io | ⚠️ | Pay-as-you-go; tiny usage can be ~free but needs a card. |
| Render free web service | ⚠️ | App free but pair with Neon (Render's own free DB expires in 30d). |
| Koyeb free | ⚠️ | Small free instance; verify current limits. |

Neon stays the database in every case (free, non-expiring, pgvector).

---

## 6. CI/CD (target, free — GitHub Actions)
1. Lint + unit tests (nodes/subgraphs).
2. Build image (sanity).
3. Push to the HF Space (git push to the Space remote triggers its build).
4. Smoke-test `/health`, `/upload`, `/chat`, `/evaluate`.
5. Confirm traces land in LangSmith.

---

## 7. Ops checklist
- [ ] `/health` returns db + vectorstore status.
- [ ] Neon holds state + vectors (survives redeploy — no disk needed).
- [ ] Secrets set as HF Space secrets, not committed.
- [ ] Log level + structured logs.
- [ ] Handle cold-start/wake gracefully (first request after idle is slow — expected).
- [ ] First boot downloads the fastembed ONNX model (~83 MB) into the model cache;
      pre-warm in the Docker build or accept a one-time delay on first embed.
- [ ] Rate limiting / auth (post-v1, tracked in requirements §5).

---

## 8. Rollback
- HF Spaces keeps build history; revert the Space to a previous commit.
- Migrations are additive/backward-compatible where possible; keep a down-path.
