# Strategy — Advanced Agentic RAG Study Helper

> The "why" behind the build. Principles and trade-offs that should guide every
> decision when the detailed docs don't have an answer. Read this when in doubt.

---

## 1. North star

Ship a **genuinely advanced agentic RAG** study assistant that is **multi-user,
durable, observable, and deployed** — not a demo. "Advanced" is earned by the two
self-correcting loops (retrieval grading + groundedness verification) and adaptive
planning, not by adding models.

---

## 2. Guiding principles

1. **Correctness before features.** The multi-user isolation and embedding-mismatch
   bugs (review.md A1–A3) outrank any new capability. A leaky or wrong-retrieval
   system is worse than a simple one.
2. **Every loop must terminate.** Self-correction is powerful and dangerous; caps
   (N/M/replan) are non-negotiable — they bound both infinite loops and cost.
3. **One place per concern.** Retrieval logic lives in a single shared subgraph;
   prompts in one module; config in env. Duplication is how the two planners in the
   current code silently drift.
4. **Observable by construction.** If we can't see a node's decision in a trace, we
   can't tune or trust it. Instrument as we build, not after.
5. **Stateless app, stateful store.** The FastAPI process holds nothing per-user;
   Postgres holds all session state. This is what makes it deployable and scalable.
6. **Ship in thin vertical slices.** Each roadmap phase is independently deployable
   and testable. No big-bang rewrite.
7. **Docs are the contract.** requirements → design → code. If code must diverge,
   update the doc in the same change.
8. **Free-first, always.** Zero budget is a hard constraint, not a preference. Every
   tool/host/service must have a viable free tier; accept free-tier trade-offs (idle
   sleep, cold starts, storage caps) rather than paying. Reject options with expiring
   free databases or paid-only requirements. Lead with the free option when presenting
   choices.

---

## 3. Key trade-offs (and our call)

| Trade-off | Options | Decision | Why |
|---|---|---|---|
| RAG sophistication vs. cost/latency | naive ↔ deep self-correction | Bounded self-correction (caps) | Advanced quality without runaway cost |
| Vector store | Chroma ↔ pgvector | **pgvector** (canonical); Chroma local-dev only | Free deploy needs no persistent disk; one datastore (Neon) for state + vectors |
| Deploy host | Render ↔ HF Spaces ↔ Fly | **Hugging Face Spaces** (Docker) | Truly free, runs our Dockerfile, no expiring DB / paid disk |
| Managed DB | Render PG ↔ Neon ↔ Supabase | **Neon** (Postgres + pgvector) | Free, non-expiring, holds state + vectors in one place |
| Streaming | none ↔ SSE ↔ WebSocket | SSE | HTTP-native, simplest progress UX |
| Memory | in-memory ↔ Postgres checkpointer | Postgres | Durability + resumability (NFR-2) |
| Auth | now ↔ later | Later (session_id v1) | Don't block core value; isolation via session_id first |
| Observability vendor | Langfuse ↔ LangSmith | **LangSmith** | Native LangChain/LangGraph tracing, least wiring |
| Evaluation tooling | none ↔ MLflow ↔ Langfuse evals | **MLflow** | Offline RAG quality eval + experiment tracking across prompt/retrieval changes |

---

## 4. What "advanced agentic RAG" means here (so we don't over/under-build)

- **Agentic** = a planner orchestrates multiple tools (teacher/quiz/eval) and
  **adapts** the plan to results.
- **Self-corrective** = CRAG-style doc grading + Self-RAG-style groundedness checks
  nested inside the agent.
- We are **not** building: multi-agent debate, autonomous tool discovery, or
  fine-tuning. Those are scope creep for v1.

---

## 5. Definition of success

- A user uploads notes, gets a **grounded** lesson (or an honest "insufficient"),
  is quizzed, and evaluated — with the system visibly retrieving, grading, and
  self-correcting in traces.
- Two users never collide.
- `git pull` + `docker-compose up` reproduces the whole system.
- The team can debug any request from its trace alone.

---

## 6. Decision log (append as we make calls)

| Date | Decision | Rationale |
|---|---|---|
| 2026-07-14 | Adopt phased roadmap; docs-first before code. | Shared contract, no big-bang rewrite. |
| 2026-07-14 | Canonical embedding = HF `all-MiniLM-L6-v2` (both ingest + query). | Local, free, no API key, no network dependency on the core retrieval path. Resolves A2. `ingest.py` must switch off Google embeddings. |
| 2026-07-14 | Observability = **LangSmith**; Evaluation = **MLflow**. | LangSmith for live per-node/LLM tracing (native to LangGraph); MLflow for offline RAG-quality eval + experiment tracking. No tool overlap. |
| 2026-07-14 | **Free-first is a hard constraint** (user has no budget). | All infra must run on free tiers; accept free-tier trade-offs over paying. |
| 2026-07-14 | Deploy = **HF Spaces** (Docker) + **Neon** (Postgres + pgvector); vector backend = **pgvector** (canonical), Chroma local-dev only. | Fully free & durable. Render rejected: free Postgres expires in 30 days, persistent disk is paid-only. pgvector removes the paid-disk need. |
| 2026-07-14 | Relational store (Phase 2) = **SQLite locally / Neon Postgres in prod** via one `DATABASE_URL`. | Zero-setup local testing (free), same code in prod. Mirrors the Chroma/pgvector split. |
