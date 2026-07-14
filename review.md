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
| A4 | ✅ 🟠 | ~~`messages` always empty → no memory.~~ **RESOLVED (Phase 5):** nodes append Human/AI messages, persisted by the checkpointer (SqliteSaver/PostgresSaver) keyed by `session_id`; teacher uses recent history. | `app/agent/*`, `app/persistence/checkpointer.py` | Phase 5 ✅ |
| A15 | 🔵 | Follow-up/meta queries ("summarize what you taught me") still run strict retrieval+grounding and may hedge; history is available but the teacher grounds against notes. Consider a lightweight "conversational" path that answers from history without retrieval. | `app/agent/teacher.py` | future |
| A16 | ✅ 🟠 | ~~Structured-output calls had no retry/fallback (Groq `tool_use_failed` on apostrophes → 500).~~ **RESOLVED:** `app/agent/structured.py` `structured_invoke()` retries, **salvages the JSON from Groq's rejected `failed_generation`** (fixes the invalid `\'` escape), and falls back to a per-call default. All structured calls (planner, retrieval subgraph, groundedness, quiz, eval) routed through it. Verified: unit salvage + live apostrophe query. | `app/agent/structured.py`, `app/agent/*` | ✅ |
| A5 | ✅ 🟡 | ~~Unused HF `DeepSeek-V4-Flash` model.~~ **RESOLVED (Phase 1):** dropped during the refactor; not carried into `app/`. | (was `app.py:19-25`) | Phase 1 ✅ |
| A6 | ✅ 🟡 | ~~No relevance grading — chunks used as-is.~~ **RESOLVED (Phase 3):** shared retrieval subgraph grades chunks (drops irrelevant) + rewrites/retries on weak retrieval, capped by `RETRIEVAL_MAX_ATTEMPTS`. | `app/agent/retrieval.py` | Phase 3 ✅ |
| A7 | ✅ 🟡 | ~~No groundedness/hallucination check.~~ **RESOLVED (Phase 4):** teacher verifies the lesson against its sources and regenerates (cap `GENERATION_MAX_ATTEMPTS`), flagging unsupported claims if still ungrounded. | `app/agent/teacher.py` | Phase 4 ✅ |
| A8 | ✅ 🟡 | ~~"Adaptive" planner is static replay.~~ **RESOLVED (Phase 4):** planner adapts on a failure signal (ungrounded lesson) — inserts a corrective task or finishes early, capped by `REPLAN_MAX`; `executor` honors an early-finish flag. | `app/agent/planner.py`, `app/agent/graph.py` | Phase 4 ✅ |
| A9 | 🟡 | CORS `allow_origins=["*"]` with credentials — tighten before deploy. | `app.py:29` | Phase 7 |
| A10 | ✅ 🔵 | ~~No `/health`, error envelope, or streaming.~~ **RESOLVED:** `/health` (Phase 1); SSE `/chat/stream` + `{error:{code,message,detail?}}` envelope (Phase 6). | `app/main.py`, `app/api/chat.py` | Phase 6 ✅ |
| A11 | 🔵 | No tests. | repo | ongoing |
| A12 | ✅ 🟠 | ~~`requirements.txt` missing imported packages / conflicting pins.~~ **RESOLVED:** replaced the broken freeze with a curated `requirements.txt`; venv built cleanly on Python 3.13; exact versions captured in `requirements.lock.txt`. | `requirements.txt`, `requirements.lock.txt` | ✅ |
| A13 | ✅ 🔵 | ~~Local HF embeddings pull `sentence-transformers` + `torch` (~2.5 GB).~~ **RESOLVED:** switched embedding runtime to **fastembed (ONNX)** — same model, ~15 MB + one-time ~83 MB model download, no torch. Much lighter for free HF Spaces. | `app/persistence/vectorstore.py` | ✅ |
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
| Restart resumes in-flight session | 5 | ✅ verified — fresh process loaded lesson + messages from checkpoint |
| Irrelevant chunks dropped + re-retrieval fires | 3 | ✅ drop verified live; retry wired+capped |
| Ungrounded generation → regenerate/flag | 4 | ✅ check runs live (grounded=True→clean); regen/flag wired+capped |
| `docker-compose up` full stack works | 7 | ⬜ |
| Every request traced | 8 | ⬜ |
| Deployed URL == local flow | 9 | ⬜ |

---

## 4. How to run a review
- Use `/code-review` on the working diff for correctness + cleanup before each merge.
- Cross-check the diff against §2 checklist and [design.md](design.md) contracts.
- Close audit items above with the PR/commit that fixes them.
