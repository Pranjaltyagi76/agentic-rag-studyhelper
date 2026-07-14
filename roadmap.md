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

## Phase 1 — Behavior-preserving refactor ✅ (code complete; runtime boot pending deps+keys)
- [x] Split `Agent.py` / `app.py` into the `app/` package (ARCHITECTURE §5).
- [x] No logic change; endpoints unchanged (`/upload`, `/chat`, `/evaluate` + new `/health`).
- [x] Resolve embedding issue (A2: removed dead Google embeddings; HF canonical) and dead HF model (A5: dropped).
- [x] Added missing runtime deps to `requirements.txt` (A12).
- **DoD:** same behavior, new structure. ⚠️ **Boot not yet verified** — this env lacks
  installed deps + API keys. Verify with: `pip install -r requirements.txt` then
  `uvicorn app.main:app --port 8000` and exercise the flow. Byte-compile + import
  graph checks passed.

## Phase 2 — Sessions & multi-user isolation (FR-6, NFR-1) ✅ (code complete; boot pending deps+keys)
- [x] Remove global `current_agent_state` / `uploaded_files` → per-`session_id` store (A1).
- [x] Add `session_id` to all endpoints + `AgentState` + the frontend (localStorage).
- [x] Relational `sessions` + `documents` tables via SQLAlchemy (SQLite local / Neon Postgres prod).
      (`messages` table deferred to Phase 5 with the checkpointer.)
- [x] Vector filter includes `session_id` (fixed `RAG_Tool`, A3) + `session_id` in chunk metadata.
- **DoD:** two concurrent sessions are isolated — different `session_id` → separate
  uploads, separate retrieval, separate quiz state. ⚠️ Verify at runtime (user testing).
- **Note:** transient agent state (quiz for `/evaluate`) is in-memory per session; it
  is isolated but not yet durable across restart (durability = Phase 5).

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

## Phase 8 — Observability (NFR-3) ✅ (wired; live LangSmith needs the user's key)
- [x] **LangSmith** tracing via `app/observability/tracing.py` — auto-enables when
      `LANGCHAIN_API_KEY` is set (no-op otherwise); `/chat` runs tagged (`run_name`,
      `tags`, `metadata`) for filterable traces.
- [x] Retrieval-quality metrics (`app/observability/metrics.py`): grade pass rate +
      attempt from the grader, grounded + attempts from the teacher (structured logs).
- [x] OpenTelemetry instruments the FastAPI layer; OTLP export if endpoint configured.
- **DoD:** wiring verified (tracing correctly OFF without key, OTel instruments, metrics
  emit, app serves). End-to-end LangSmith trace confirmed once the user adds their key.

## Phase 8.5 — Evaluation (NFR-9, post-M2)
- [ ] Curate a small RAG eval set (questions + expected/reference answers per doc).
- [ ] **MLflow** eval: retrieval relevance, groundedness/faithfulness, answer quality.
- [ ] Log runs to MLflow so prompt/retrieval changes are compared over time.
- **DoD:** a config/prompt change shows a measurable delta in MLflow.
- *Note:* needs the self-correcting loops (M2) to exist before it's meaningful.

## Phase 9 — Deployment (free tier: HF Spaces + Neon)
- [ ] Provision **Neon** free Postgres; `CREATE EXTENSION vector;`.
- [ ] Deploy app to **Hugging Face Spaces** (Docker SDK) from our Dockerfile.
- [ ] Secrets set as HF Space secrets; healthcheck; migrations on boot.
- [ ] `VECTOR_BACKEND=pgvector` in prod (Chroma stays local-dev only).
- **DoD:** public HF Space URL runs the same flow as local; traces flow in prod; $0 spent.

---

## Milestones
- **M1 — Solid foundation:** Phases 1–2 (structure + isolation).
- **M2 — Advanced RAG:** Phases 3–4 (the self-correcting loops = "advanced"). ✅ **DONE**
- **M3 — Durable & observable:** Phases 5–6, 8 (+ 8.5 evaluation). *(Phase 5 ✅)*
- **M4 — Shipped:** Phases 7, 9.

## Risks (see review.md for the live log)
- Embedding mismatch corrupts retrieval → address in Phase 1.
- LLM structured-output flakiness on planners → validation + retries.
- Cost/latency from nested loops → the N/M/replan caps in design.md §3–5.
