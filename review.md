# Review — Advanced Agentic RAG Study Helper

> Living audit + quality-gate log. Two purposes:
> 1. **Current-state audit** — issues found in today's code.
> 2. **Review checklist** — the bar every phase/PR must clear.
> Update this file as issues are opened/closed.

---

## 1. Current-state audit (as of Phase 0)

Severity: 🔴 blocker · 🟠 correctness · 🟡 quality · 🔵 nice-to-have

| # | Sev | Issue | Location | Fix in phase |
|---|-----|-------|----------|--------------|
| A1 | ✅ 🔴 | ~~Global mutable `current_agent_state` & `uploaded_files`.~~ **RESOLVED (Phase 2):** per-`session_id` in-memory store (`app/api/session_store.py`) for transient state; per-session uploads from the DB. No cross-user overwrite. | `app/api/session_store.py` | Phase 2 ✅ |
| A2 | ✅ 🔴 | ~~Embedding mismatch.~~ **RESOLVED (Phase 1):** the Google embeddings in the old `ingest.py` were *unused dead code* — the real pipeline already embedded with HF `all-MiniLM-L6-v2` via `vectordb.add_documents`. So there was no live mismatch, only misleading code. Removed the dead Google embeddings; HF is the single embedder in `app/persistence/vectorstore.py`. No re-ingestion needed. | `app/persistence/vectorstore.py`, `app/ingest/ingest.py` | Phase 1 ✅ |
| A3 | ✅ 🟠 | ~~Vector retrieval filters by `file_name` only.~~ **RESOLVED (Phase 2):** `RAG_Tool` filters by `session_id` (always) + `file_name`; chunks carry `session_id` metadata. A session can only retrieve its own docs. | `app/agent/retrieval.py`, `app/ingest/ingest.py` | Phase 2 ✅ |
| A4 | 🟠 | `messages` is always passed empty → no memory despite `add_messages` reducer. | `app/api/chat.py` | Phase 5 |
| A5 | ✅ 🟡 | ~~Unused HF `DeepSeek-V4-Flash` model.~~ **RESOLVED (Phase 1):** dropped during the refactor; not carried into `app/`. | (was `app.py:19-25`) | Phase 1 ✅ |
| A6 | 🟡 | No relevance grading — retrieved chunks used as-is (naive, not advanced). | `teacher_node`, `quiz_generator_node` | Phase 3 |
| A7 | 🟡 | No groundedness/hallucination check on generated lessons. | `teacher_node` | Phase 4 |
| A8 | 🟡 | "Adaptive" planner is actually static replay of a fixed task list. | `executor` `Agent.py:874` | Phase 4 |
| A9 | 🟡 | CORS `allow_origins=["*"]` with credentials — tighten before deploy. | `app.py:29` | Phase 7 |
| A10 | 🔵 | No `/health`, no structured error envelope, no streaming. | `app.py` | Phase 6 |
| A11 | 🔵 | No tests. | repo | ongoing |
| A12 | 🟠 | `requirements.txt` (frozen) is **missing packages the code imports**: `langchain-groq`, `langchain-tavily`, `sentence-transformers`, `pypdf`. Implies the original code was never fully run in that env. **Partly fixed (Phase 1):** added them (unpinned) at the bottom of `requirements.txt`; must `pip install -r requirements.txt` and re-freeze to pin. | `requirements.txt` | Phase 1 (verify on install) |
| A13 | 🔵 | Local HF embeddings pull `sentence-transformers` + `torch` (large download). Fine for free tier but slows first boot; note for HF Spaces build. | `app/persistence/vectorstore.py` | Phase 7/9 |
| A14 | 🔵 | Any vectors ingested BEFORE Phase 2 lack `session_id` metadata, so they won't match the session filter. Not a code bug — just re-upload docs after Phase 2 (or wipe `Chromadb/`). | `Chromadb/` | note only |

---

## 2. Per-PR review checklist

### Correctness
- [ ] No global mutable request state; everything scoped to `session_id`.
- [ ] Every vector query filters on `session_id`.
- [ ] Ingest and query embeddings use the **same** model.
- [ ] All self-correction loops have a hard cap (N/M/replan) — provably terminate.
- [ ] Structured-output LLM calls handle validation failure (retry/fallback).

### Design fidelity
- [ ] Matches node contracts in [design.md](design.md) §2.
- [ ] Retrieval logic lives in the shared subgraph, not duplicated.
- [ ] New env vars added to `.env.example` + [deployment.md](deployment.md).

### Quality
- [ ] Node is unit-testable in isolation (pure `state -> partial_state`).
- [ ] No secrets/keys hard-coded.
- [ ] Prompts kept in one place; not silently divergent copies.

### Observability
- [ ] New node/LLM call is traced.
- [ ] Failure paths log with context (session_id, node, attempt).

### Security / privacy
- [ ] No user data in URLs/query strings.
- [ ] CORS scoped for deploy.
- [ ] Upload validates file type/size.

---

## 3. Acceptance-test log (fill as phases complete)

| Test | Phase | Status |
|---|---|---|
| Two sessions isolated (no cross-read) | 2 | 🧪 ready to test (code complete) |
| Restart resumes in-flight session | 5 | ⬜ |
| Irrelevant chunks dropped + re-retrieval fires | 3 | ⬜ |
| Ungrounded generation → regenerate/flag | 4 | ⬜ |
| `docker-compose up` full stack works | 7 | ⬜ |
| Every request traced | 8 | ⬜ |
| Deployed URL == local flow | 9 | ⬜ |

---

## 4. How to run a review
- Use `/code-review` on the working diff for correctness + cleanup before each merge.
- Cross-check the diff against §2 checklist and [design.md](design.md) contracts.
- Close audit items above with the PR/commit that fixes them.
