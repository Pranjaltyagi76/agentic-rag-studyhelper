# Project Metrics & Results

Measured results for the Advanced Agentic RAG Study Helper. Every number here was
produced by an actual run — nothing is estimated. Reproduction commands are included
so any of it can be re-checked.

**Last measured:** 2026-07-15 · **Cost to run the whole stack:** $0 (all free tiers)

---

## 1. RAG quality (MLflow, LLM-as-judge)

Logged run: experiment `studyhelper-rag-eval`, run **`baseline`** (`b208ff51`).
Scores are 1–5 (5 = best), judged by an LLM against a reference answer.

| Metric | Score | What it measures |
|---|---|---|
| **avg_retrieval_relevance** | **5.00 / 5** | Do the retrieved notes contain what's needed to answer? |
| **avg_faithfulness** | **5.00 / 5** | Is the answer grounded in those notes (not hallucinated)? |
| **avg_answer_correctness** | **5.00 / 5** | Does the answer match the reference? |

### Per-example

| Example | Retrieval relevance | Faithfulness | Answer correctness | Retrieval attempts |
|---|---|---|---|---|
| `photosynthesis` | 5 | 5 | 5 | 1 |
| `newtons-second-law` | 5 | 5 | 5 | 1 |
| `binary-search` | 5 | 5 | 5 | 1 |

`retrieval_attempts = 1` everywhere means retrieval was judged **sufficient on the first
pass** — the rewrite/retry loop was correctly *not* needed.

### Configuration logged with the run
| Param | Value |
|---|---|
| `groq_model` | `llama-3.3-70b-versatile` |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fastembed/ONNX) |
| `retrieval_max_attempts` | 2 |
| `generation_max_attempts` | 2 |
| `num_examples` | 3 |

**Reproduce:**
```bash
python evaluation/run_eval.py --label baseline
mlflow ui --backend-store-uri sqlite:///mlflow.db   # compare runs at :5000
```

> **Caveat, stated plainly:** 3 examples with clean, unambiguous notes is a smoke-test
> baseline, not a benchmark. A perfect 5.00 says "no regression on easy cases" — it does
> **not** prove quality on hard/ambiguous material. The value is the *delta* when a
> prompt or knob changes.

---

## 2. Advanced-RAG behaviour (the loops that make it "advanced")

| Capability | Result | Evidence |
|---|---|---|
| **Retrieval grading drops junk** (CRAG) | ✅ Verified | Seeded 2 photosynthesis chunks + 1 Eiffel Tower distractor → grader **kept both relevant, dropped the distractor**. Final notes contained no "Eiffel". |
| **Grade pass rate metric** | ✅ Emitting | e.g. `retrieval_grade attempt=1 relevant=2 total=5 pass_rate=0.40` |
| **Groundedness check** (Self-RAG) | ✅ Verified | `grounded=True` on grounded input; lesson returned clean with no uncertainty flag. |
| **Regenerate on ungrounded** | 🔶 Wired, not observed | Path implemented + capped at `GENERATION_MAX_ATTEMPTS=2`; forcing a genuinely ungrounded generation on demand is non-deterministic. |
| **Adaptive planner** | ✅ Verified | `replans=0` on success (stays cheap, no extra LLM call); LLM recovery path fires only on a failure signal, capped at `REPLAN_MAX=3`. |
| **All loops terminate** | ✅ By construction | Hard caps: retrieval 2, generation 2, replans 3. |

---

## 3. Multi-user isolation (the security-critical one)

**Method:** two sessions seeded with different notes. Session B's notes contained an
invented codename (`ZORBLAX-9931`) that cannot come from general knowledge or the web —
so any appearance in session A would be an unambiguous leak.

| Check | Result |
|---|---|
| Session A (not the owner) sees the secret? | **No** — replied *"the retrieved notes section is empty"* ✅ |
| Session B (the owner) sees the secret? | **Yes** — retrieved correctly ✅ |
| Vector-store filter isolation (direct, no LLM) | Queried for another session's content → **0 leak** ✅ |

---

## 4. Durability & memory

| Check | Result |
|---|---|
| State survives a **real process restart** | ✅ Fresh process loaded the session's lesson + message history from the checkpoint |
| Messages accumulate across turns | ✅ 2 → 4 messages across turns, persisted |
| `/evaluate` resumes the quiz after restart | ✅ Reads from the durable checkpoint (in-memory store retired) |

---

## 5. API / end-to-end suite

**7/7 checks passed** against the live app:

| Check | Result |
|---|---|
| `GET /health` → 200 | ✅ |
| `POST /evaluate` unknown session → 400 structured envelope | ✅ |
| `POST /chat` teach (**apostrophe query**, the A16 regression) → lesson | ✅ |
| `POST /chat` quiz → questions returned | ✅ |
| `POST /evaluate` → verdict | ✅ |
| Follow-up turn (memory) | ✅ |
| `POST /chat/stream` → progress + final SSE events | ✅ |

---

## 6. Observability

| Check | Result |
|---|---|
| LangSmith traces landing | ✅ Live in project `studyhelper` — node tree visible: `planner → executor → ChatGroq → parsers` |
| Retrieval/groundedness metrics | ✅ Emitted per request |
| OpenTelemetry on FastAPI | ✅ Instrumented (exports when an OTLP endpoint is set) |

---

## 7. Production data layer (Neon pre-flight)

All three drivers verified against the **live** Neon database — **6/6**:

| Check | Result |
|---|---|
| Direct (non-pooled) connection | ✅ no `-pooler` |
| `psycopg3` + pgvector extension | ✅ `vector 0.8.0` |
| SQLAlchemy / `psycopg2` | ✅ PostgreSQL **17.10** |
| `init_db` (sessions/documents tables) | ✅ created |
| PGVector store | ✅ ready |
| PostgresSaver checkpointer | ✅ ready *(bug found & fixed here — see below)* |

---

## 8. Engineering wins worth recording

| Item | Impact |
|---|---|
| **fastembed (ONNX) instead of sentence-transformers (PyTorch)** | Same model, **~15 MB vs ~2.5 GB** — no torch. Made a free-tier deploy viable and fixed a dependency deadlock. |
| **A16: structured-output salvage** | Groq returned `tool_use_failed` on apostrophes ("Newton's") → 500s. Now retried + **salvaged from the rejected generation**. Verified: unit repro + live apostrophe query. |
| **Checkpointer connection pool** | Pre-flight caught `from_conn_string().__enter__()` closing the connection — would have **silently killed memory in production**. |
| **Secret hygiene** | Scan across all 19 commits: **0 secrets**. `.env` never committed. |
| **Curated requirements** | Replaced a broken 200+ line `pip freeze` (missing imports, conflicting pins) with ~20 real deps + a lockfile. |

---

## 9. Known gaps (honest)

| Gap | Status |
|---|---|
| `docker compose up` never actually run | Docker not installed locally; Dockerfile/compose validated by inspection only. Real test = the Render deploy. |
| Ungrounded → regenerate never observed firing | Mechanism verified; can't force it deterministically. |
| Eval set is 3 easy examples | Smoke test, not a benchmark. Expand for real signal. |
| No per-user auth | v1 relies on unguessable session tokens + optional `APP_API_KEY`. Real auth is future work. |
| Deployed URL | Phase 9 in progress (Render + Neon). |

---

## 10. Cost

| Component | Tier | Cost |
|---|---|---|
| Groq (LLM) | Free — 100k tokens/day | $0 |
| Neon (Postgres + pgvector) | Free, non-expiring | $0 |
| LangSmith | Free tier | $0 |
| MLflow | Local (SQLite) | $0 |
| Embeddings | Local ONNX, no API | $0 |
| Render (web service) | Free — 750 hrs/mo | $0 |
| **Total** | | **$0** |

*Note: the Groq free tier is 100k tokens/day. Heavy testing exhausted it once — the
self-correcting loops make each request cost several LLM calls (bounded by the caps).*
