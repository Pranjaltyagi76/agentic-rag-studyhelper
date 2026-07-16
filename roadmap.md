# Roadmap — Advanced Agentic RAG Study Helper

> Execution plan. Each phase is independently shippable & testable. Check items off
> as we go. Ties to [requirements.md](requirements.md) FR/NFR ids and
> [design.md](design.md) sections.

---

## Legend
`[ ]` todo · `[~]` in progress · `[x]` done · **DoD** = definition of done

---

## Phase 0 — Planning docs ✅
- [x] ARCHITECTURE.md
- [x] requirements.md, design.md, roadmap.md, deployment.md, review.md, strategy.md
- **DoD:** shared understanding of what/how/when before code changes.

## Phase 1 — Behavior-preserving refactor ✅
- [x] Split `Agent.py` / `app.py` into the `app/` package (ARCHITECTURE §5).
- [x] No logic change; endpoints unchanged (`/upload`, `/chat`, `/evaluate` + new `/health`).
- [x] Resolve embedding issue (A2: removed dead Google embeddings; HF canonical) and dead HF model (A5: dropped).
- [x] Curated `requirements.txt` + `requirements.lock.txt` (A12); embeddings run on
      fastembed/ONNX (torch-free, A13).
- **DoD:** ✅ verified — `.venv` (Python 3.13) installs clean, app boots, all endpoints
  route, and the flow was exercised end-to-end against live keys.

## Phase 2 — Sessions & multi-user isolation (FR-6, NFR-1) ✅
- [x] Remove global `current_agent_state` / `uploaded_files` → per-`session_id` store (A1).
- [x] Add `session_id` to all endpoints + `AgentState` + the frontend (localStorage).
- [x] Relational `sessions` + `documents` tables via SQLAlchemy (SQLite local / Neon Postgres prod).
      (`messages` table deferred to Phase 5 with the checkpointer.)
- [x] Vector filter includes `session_id` (fixed `RAG_Tool`, A3) + `session_id` in chunk metadata.
- **DoD:** ✅ **verified live** — two sessions seeded with different notes; session A asked
  for a codename that only existed in session B's notes and could NOT see it ("retrieved
  notes section is empty"), while owner B retrieved it fine. No cross-session leak.
- **Note:** the in-memory per-session store was later retired in Phase 5 — transient state
  (the quiz for `/evaluate`) is now durable in the checkpointer.

## Phase 3 — Self-corrective retrieval subgraph (FR-2.2, FR-2.3) ✅
- [x] `plan_retrieval` → `retrieve` → `grade` → `plan_web` → `web_fetch` subgraph
      (`app/agent/retrieval.py`), with a `grade → rewrite → retrieve` retry loop.
- [x] Retry cap `RETRIEVAL_MAX_ATTEMPTS` (default 2); LLM query rewrite on weak retrieval.
- [x] Shared by teacher (`allow_web=True`) & quiz_generator (`allow_web=False`); their
      old inline planners are gone.
- **DoD:** ✅ verified live — grader dropped an irrelevant chunk (Eiffel Tower) while
  keeping relevant ones (photosynthesis); retry loop wired + capped. Traces will show
  in LangSmith once Phase 8 enables it.

## Phase 4 — Generate+verify loop + adaptive planner (FR-2.5, FR-5.3) ✅
- [x] Teacher `generate → check_groundedness → regenerate` loop, cap `GENERATION_MAX_ATTEMPTS`;
      flags unsupported claims if still ungrounded; skips check when there are no sources.
- [x] Planner re-invoked after each task; on a failure signal (ungrounded lesson) it asks
      an LLM whether to insert a corrective task or finish early; cap `REPLAN_MAX`.
- **DoD:** ✅ verified live — groundedness check runs (grounded=True on grounded input,
  lesson clean; regenerate/flag path wired + capped); planner stays cheap on success
  and adapts only on failure (replans=0 when grounded).

## Phase 5 — Persistence & memory (FR-6.2, FR-6.4, NFR-2) ✅
- [x] Checkpointer wraps the graph, keyed by `thread_id == session_id`
      (`app/persistence/checkpointer.py`): SqliteSaver local / PostgresSaver on Neon.
- [x] Real conversational `messages` memory: nodes append Human/AI messages; teacher
      uses recent history for continuity; persists across requests.
- [x] Retired the in-memory `session_store`; `/evaluate` now resumes the quiz from the
      durable checkpoint (survives restart).
- **DoD:** ✅ verified across a genuine process restart — a fresh process loaded the
  session's lesson + message history from the checkpoint and continued the conversation.

## Phase 6 — Streaming API (FR-7.2) ✅
- [x] SSE `/chat/stream` emits node-by-node progress then a final-state event; `/chat`
      (non-streaming) kept for compatibility. Frontend consumes the stream + shows progress.
- [x] Consistent error envelope `{ "error": { code, message, detail? } }` via exception
      handlers (HTTP + validation + unhandled); `/health` (from Phase 1).
- **DoD:** ✅ verified — TestClient saw progress events + final lesson over
  text/event-stream; 400 and 422 both returned the structured envelope.

## Phase 7 — Docker (NFR-4, NFR-5) ✅ (code+files done; `docker compose up` = user verify)
- [x] Multi-stage `Dockerfile` at repo root (port 7860 for HF Spaces; pre-downloads the
      fastembed model into the image); `.dockerignore` keeps secrets/data out.
- [x] `docker-compose.yml` mirrors prod: app on `pgvector/pgvector:pg16`
      (`VECTOR_BACKEND=pgvector`, Postgres for state+checkpoints+vectors), app exposed on host `8000`.
- [x] Implemented the `pgvector` vector backend (`VECTOR_BACKEND` switch in
      `vectorstore.py`) so local Docker uses the same backend as Neon; portable `$eq`
      retrieval filter works on both Chroma and pgvector (verified).
- **DoD:** `docker compose up --build` → working stack on pgvector (needs Docker; not
  runnable in this dev sandbox — code + compose validated, user runs to confirm).

## Phase 8 — Observability (NFR-3) ✅
- [x] **LangSmith** tracing via `app/observability/tracing.py` — auto-enables when
      `LANGCHAIN_API_KEY` is set (no-op otherwise); `/chat` runs tagged (`run_name`,
      `tags`, `metadata`) for filterable traces.
- [x] Retrieval-quality metrics (`app/observability/metrics.py`): grade pass rate +
      attempt from the grader, grounded + attempts from the teacher (structured logs).
- [x] OpenTelemetry instruments the FastAPI layer; OTLP export if endpoint configured.
- **DoD:** ✅ **verified live** — key added and authenticated; real traces land in the
  `studyhelper` project (node tree visible: planner → executor → ChatGroq → parsers).
  Metrics emit (`pass_rate`, `grounded`); OTel instruments the app.

## Phase 8.5 — Evaluation (NFR-9, post-M2) ✅
- [x] Curated eval set (`evaluation/dataset.py`) — note chunks + question + reference,
      with distractors to exercise grading.
- [x] **MLflow** eval (`evaluation/run_eval.py`): retrieval relevance, faithfulness,
      answer correctness via LLM judges (`evaluation/judges.py`).
- [x] Runs log params (model, retrieval/generation knobs) + averaged metrics + a
      per-example artifact to MLflow (SQLite backend) → comparable across changes.
- **DoD:** ✅ baseline run logged (3 examples, avg 5.0 across all metrics). Re-running
  with a changed knob/prompt under a new `--label` produces a comparable run in `mlflow ui`.

## Phase 9 — Deployment (free tier: **Render** + Neon) ✅ **LIVE**
> Host changed from HF Spaces → Render: HF locked the Docker SDK behind a paid plan
> (see strategy.md decision log). Neon is unchanged.
- [x] Provision **Neon** free Postgres; `CREATE EXTENSION vector;` (pgvector 0.8.0).
- [x] Pre-flight verified against live Neon: psycopg3, SQLAlchemy/psycopg2 (PG 17.10),
      `init_db` tables, PGVector store, PostgresSaver checkpointer — all OK.
- [x] Code pushed to a GitHub repo (`Pranjaltyagi76/agentic-rag-studyhelper`, now public);
      secret scan across all commits: 0 hits.
- [x] Render **free web service** (Docker runtime), Virginia region — same as Neon.
- [x] Env vars set: `DATABASE_URL` (Neon), `VECTOR_BACKEND=pgvector`, `APP_ENV=production`,
      Groq/Google/Tavily/LangSmith keys.
- [x] Frontend served at `/` by the app itself, so the deployed URL **is** the app
      (previously a 404 — a visitor had no way to use it).
- **DoD:** ✅ **LIVE — https://agentic-rag-studyhelper.onrender.com**
  Verified in production: `/` → 200 text/html serving the app; `/health` → 200;
  live `/chat` teaches correctly (the general-knowledge regression is gone);
  traces flow to LangSmith; **$0 spent**.
- **Known trade-off:** free tier spins down after 15 min idle → ~50s cold start.

---

## Milestones
- **M1 — Solid foundation:** Phases 1–2 (structure + isolation). ✅ **DONE**
- **M2 — Advanced RAG:** Phases 3–4 (the self-correcting loops = "advanced"). ✅ **DONE**
- **M3 — Durable & observable:** Phases 5–6, 8, 8.5. ✅ **DONE**
- **M4 — Shipped:** Phases 7, 9. ✅ **DONE** — live at
  https://agentic-rag-studyhelper.onrender.com

## Risks (see review.md for the live log)
- Embedding mismatch corrupts retrieval → address in Phase 1.
- LLM structured-output flakiness on planners → validation + retries.
- Cost/latency from nested loops → the N/M/replan caps in design.md §3–5.
