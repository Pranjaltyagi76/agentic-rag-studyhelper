# Architecture — Advanced Agentic RAG Study Helper

> **Status:** Design document. No implementation yet. This describes the *target*
> architecture and how it differs from the current codebase.

---

## 1. Where we are today

The current graph (`Agent.py`) is **lightly agentic RAG**. It already has:

- An **LLM planner** that decomposes a query into ordered tasks and routes them
  (`PlannerNode`, `Agent.py:66`).
- **Conditional retrieval** — teacher/quiz nodes decide whether to retrieve
  (`use_rag`) instead of always retrieving (`teacher_node`, `Agent.py:379`).
- **Conditional web fallback** — a second planner decides `use_web`
  (`web_research_system`, `Agent.py:264`).
- **Tool routing** across `teacher` / `quiz_generator` / `quiz_eval`.

### What makes it *not yet* "advanced"

| Missing capability | Why it matters | Where it lands in the redesign |
|---|---|---|
| **Retrieval grading** | Retrieved chunks are used without checking relevance (CRAG). | Retrieval subgraph → `grade_docs` |
| **Self-correcting re-retrieval** | Retrieval is single-shot; no "bad docs → rewrite → retry" loop. | Retrieval subgraph loop |
| **Groundedness / hallucination check** | The lesson is never verified against its sources (Self-RAG). | Teacher generate+verify loop |
| **Adaptive re-planning** | The loop back to `planner` only replays a *static* plan (`executor`, `Agent.py:874`). | Adaptive planner |
| **Real memory** | `messages` is passed in empty on every call (`app.py:63`). | Checkpointer + session store |
| **Multi-user safety** | `current_agent_state` / `uploaded_files` are global module vars (`app.py:36`). | Session-scoped state |

**Conclusion:** "agentic RAG" is defensible; "advanced" is not — the two loops that
define advanced agentic RAG (grade → correct → retry, and generate → verify →
regenerate) are absent, and there is no durable per-user state.

---

## 2. Target architecture

Core idea: turn each retrieval-using node from a straight line into a
**self-correcting loop**, add **session-scoped persistence**, and make the
**planner adaptive**.

### 2.1 High-level node graph

```
START
  │
  ▼
planner  ⇄  (adaptive: re-invoked with results-so-far; may add/skip/reorder tasks)
  ├──► teacher      → [retrieval subgraph] → [generate + verify loop] ─┐
  ├──► quiz_gen     → [retrieval subgraph] → generate ─────────────────┤
  └──► quiz_eval ──────────────────────────────────────────────────────┤
                              (all nodes loop back to planner) ─────────┘
                                                                        │
                                                                        ▼
                                                                       END

  + PostgresSaver checkpointer wraps the whole graph (per thread_id)
```

### 2.2 Retrieval subgraph (shared by teacher & quiz_generator)

Replaces the linear RAG currently inside each node.

```
query
  │
  ▼
rewrite_query ──► retrieve ──► grade_docs
     ▲                             │
     │              ┌──────────────┴───────────────┐
     │          relevant &                     irrelevant /
     │          sufficient?                    insufficient?
     │              │                              │
     │              ▼                       (retry up to N)
     │           to generate         ┌────────────┴─────────┐
     │                               │                      │
     └───────────────────────────────┘              web_fallback
                (rewrite & retry)                          │
                                              docs still insufficient?
                                                           │
                                                    flag "insufficient"
                                                           ▼
                                                      to generate
```

- **`grade_docs`**: LLM scores each chunk relevant/irrelevant, drops the junk.
- **Loop**: too few relevant chunks → rewrite query → re-retrieve (cap **N**
  attempts) → web fallback → otherwise admit insufficiency to the generator.
- Implemented once as a reusable subgraph; both teacher and quiz call it.

### 2.3 Generation + verification loop (teacher)

```
generate_lesson ──► groundedness_check (lesson vs. sources)
                          │
                    grounded?
                    ├── yes ─► return lesson
                    └── no  ─► regenerate (cap M) or return with uncertainty flag
```

### 2.4 Adaptive planner

- The planner emits an initial plan **and** is re-invoked with the results so far,
  allowed to add / skip / reorder tasks (e.g. "teacher couldn't answer → insert a
  research task"). Today `executor` can only replay a fixed list (`Agent.py:874`).

---

## 3. Persistence & memory

- **Graph checkpointer**: `PostgresSaver` (LangGraph) → durable graph state,
  resumable runs, and real `messages` memory keyed by `thread_id`.
- **Session model**: replace the global `current_agent_state` / `uploaded_files`
  with a `session_id` → per-user state row in Postgres.
- **Documents**: file metadata in a Postgres table; vectors in **pgvector** (same
  Neon Postgres) filtered by `session_id` **and** `file_name` (today the filter is
  `file_name` only — see `RAG_Tool`, `Agent.py:226`). Chroma is local-dev only.

### Proposed data model (sketch)

```
sessions(          session_id PK, user_id, created_at, updated_at )
messages(          id PK, session_id FK, role, content, created_at )
documents(         id PK, session_id FK, file_name, status, chunk_count, created_at )
langgraph_checkpoints( ... managed by PostgresSaver ... )
vectors → pgvector (same Neon Postgres), metadata: { session_id, file_name, chunk_id }
```

---

## 4. Cross-cutting infrastructure (the five asks)

| Concern | Plan |
|---|---|
| **FastAPI + API** | `/chat` and `/evaluate` become `thread_id`/`session_id`-scoped. Add SSE or WebSocket streaming of node-by-node progress. |
| **Persistence** | **Neon** Postgres for checkpoints + sessions + docs **and** embeddings (`pgvector`) — one free datastore. Chroma local-dev only. |
| **Docker** | Multi-stage image (port 7860 for HF Spaces). `docker-compose` for local dev uses `pgvector/pgvector:pg16` to mirror Neon. Secrets via env, never baked in. |
| **Deployment** | **Free tier: Hugging Face Spaces** (Docker) for the app + **Neon** for Postgres/pgvector. Env-based config, healthcheck, migrations on boot. Render rejected (free DB expires 30d, disk paid). |
| **Observability** | **LangSmith** tracing on every node + retrieval-quality metrics (grade pass rate, retry count, groundedness rate). OpenTelemetry on the FastAPI layer for request traces/latency. |
| **Evaluation** | **MLflow** for offline RAG-quality eval (retrieval relevance, groundedness, answer quality) + experiment tracking across prompt/retrieval changes. Post-M2. |

---

## 5. Target repo layout (proposed, not yet created)

```
app/
  main.py            # FastAPI app, routers, middleware
  api/
    chat.py          # /chat, streaming
    evaluate.py      # /evaluate
    upload.py        # /upload
  agent/
    graph.py         # top-level StateGraph assembly
    planner.py       # adaptive planner node
    teacher.py       # teacher node + generate/verify loop
    quiz.py          # quiz_generator, quiz_eval
    retrieval.py     # shared retrieval subgraph (rewrite→retrieve→grade→loop)
    state.py         # AgentState, pydantic schemas
  ingest/
    ingest.py        # PDF load + chunk (from current ingest.py)
  persistence/
    db.py            # Postgres engine/session
    models.py        # sessions, messages, documents
    checkpointer.py  # PostgresSaver wiring
    vectorstore.py   # pgvector (deploy) / Chroma (local dev) — from current database.py
  observability/
    tracing.py       # LangSmith/OTel setup
  config.py          # env-driven settings
docker/
  Dockerfile
  docker-compose.yml
tests/
ARCHITECTURE.md      # this file
```

---

## 6. Suggested build order (phased, for when we implement)

1. **Refactor** current single-file agent into the `app/` layout above (behavior-preserving).
2. **Session state** — kill global vars, add `session_id`, Postgres sessions/docs, scoped vector filter.
3. **Retrieval subgraph** — rewrite → retrieve → grade → retry → web fallback.
4. **Generate+verify loop** in teacher; **adaptive planner**.
5. **Checkpointer** (PostgresSaver) + real `messages` memory.
6. **Streaming API** (SSE/WebSocket).
7. **Docker Compose** (app + pgvector Postgres) for local dev; Dockerfile for HF Spaces.
8. **Observability** (LangSmith + OTel); **Evaluation** (MLflow, local).
9. **Deploy**.

Each phase is independently shippable and testable.
