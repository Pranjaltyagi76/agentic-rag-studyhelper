# Project Metrics & Results

Measured results for the Advanced Agentic RAG Study Helper. Every number was produced
by an actual run and logged to MLflow — nothing is estimated. Reproduction commands are
included so any of it can be re-checked.

**Last measured:** 2026-07-15 · **Cost to run the whole stack:** $0 (all free tiers)

---

## 1. Headline: ablation — does the "advanced" machinery actually earn its keep?

The central question for any agentic-RAG system. Both variants run over **identical
seeded data**; the only difference is the pipeline.

- **naive** — top-k (k=4) similarity retrieval, no grading, no rewrite/retry *(classic RAG)*
- **advanced** — `plan → retrieve → grade → rewrite/retry` subgraph *(this project)*

Metrics are **deterministic**, computed against the dataset's ground-truth `relevant`
labels — no LLM judge can flatter them.

| Metric | Naive | **Advanced** | Delta |
|---|---|---|---|
| **Retrieval precision** | 0.396 | **1.000** | **+153%** |
| **Retrieval recall** | 1.000 | **1.000** | no loss |
| **Retrieval F1** | 0.558 | **1.000** | **+79%** |
| **Distractor rejection** | 0.042 | **1.000** | **4% → 100%** |
| **Hallucination rate** (unanswerable) | 0.500 | **0.000** | **−100%** |
| Chunks fed to generator | 3.17 | 1.00 | −68% context noise |
| Retrieval latency | 0.02 s | 1.10 s | **+1.08 s** (the cost) |

### What this actually says
- **Grading eliminated 100% of distractors and raised precision ~2.5× with zero recall
  loss.** Naive RAG passed ~3 chunks per query of which ~60% were irrelevant noise.
- **The cost is real and measured:** ~1 extra second per query and 2 extra LLM calls.
  That is the honest price of self-correction.
- The hardest case (`mitosis-vs-meiosis-near-miss`) is the decisive one: embeddings rank
  the wrong-but-similar chunk highly, so **only grading rejects it**. Naive scored
  precision 0.33 there; advanced scored 1.00.

**Reproduce:**
```bash
python evaluation/run_eval.py --variant naive
python evaluation/run_eval.py --variant advanced
mlflow ui --backend-store-uri sqlite:///mlflow.db     # compare runs at :5000
```

### Dataset (6 cases, adversarial by design)
| Type | Cases | Purpose |
|---|---|---|
| `distractor` | 2 | Off-topic chunks that must be dropped |
| `near_miss` | 1 | Closely-related *wrong* topic — embeddings fail, only grading catches it |
| `multi_hop` | 1 | Answer requires combining two chunks (recall probe) |
| `unanswerable` | 2 | Answer genuinely absent — **hallucination probe** |

> **Caveat, stated plainly:** 6 curated cases is a focused benchmark, not a large-scale
> one. It's designed to isolate *specific failure modes*, and it did exactly that (§2).
> Broader claims need a bigger set.

---

## 2. A defect this benchmark found — and the fix

The eval's value proved itself immediately: it **caught a real bug that all previous
testing missed.**

**Finding:** on unanswerable questions the advanced pipeline correctly retrieved
**zero** chunks (right call) — but the teacher, handed empty context, then answered from
**general knowledge anyway**, ignoring the user's *"according to my notes"* scoping.

```
hallucination_rate:   naive 0.50   →   advanced 1.00     ← advanced was WORSE
```
Naive scored better only by luck: it passed irrelevant chunks, so the model had
something to point at and said "not in these notes."

**Root cause:** [`teacher.py`](../../app/agent/teacher.py) skipped the groundedness check
entirely when `sources` was empty, so nothing prevented general-knowledge substitution.

**Fix:** a source-scoping guard — when `use_rag` was true but grading returned no
relevant notes, the generator is explicitly instructed to say the notes don't cover it
and *not* to answer from its own knowledge.

**Re-measured after the fix:**

| Metric | Before | After |
|---|---|---|
| **hallucination_rate** | 1.000 | **0.000** |
| **abstention_rate** | 0.000 | **1.000** |
| retrieval F1 | 1.000 | 1.000 *(no regression)* |
| distractor rejection | 1.000 | 1.000 *(no regression)* |

---

## 3. Advanced-RAG behaviour

| Capability | Result | Evidence |
|---|---|---|
| **Retrieval grading drops junk** (CRAG) | ✅ | 100% distractor rejection vs 4% naive |
| **Near-miss rejection** | ✅ | Meiosis chunk rejected on a mitosis question |
| **Multi-hop recall** | ✅ | Both required chunks retained (recall 1.00) |
| **Source scoping / abstention** | ✅ | 100% abstention on unanswerable (after §2 fix) |
| **Groundedness check** (Self-RAG) | ✅ | `grounded=True` on grounded input; lesson clean |
| **Regenerate on ungrounded** | 🔶 Wired, not observed firing | Capped at `GENERATION_MAX_ATTEMPTS=2`; not deterministically forceable |
| **Adaptive planner** | ✅ | `replans=0` on success (no wasted LLM call); recovery only on failure, capped at `REPLAN_MAX=3` |
| **All loops terminate** | ✅ | Hard caps: retrieval 2, generation 2, replans 3 |
| **Degradation is never silent** | ✅ | Grader fallback marks rows `degraded` and excludes them from metrics |

---

## 4. Multi-user isolation

**Method:** two sessions seeded with different notes; session B's contained an invented
codename (`ZORBLAX-9931`) that cannot come from general knowledge or the web — so any
appearance in session A is an unambiguous leak.

| Check | Result |
|---|---|
| Session A (not owner) sees the secret? | **No** — *"the retrieved notes section is empty"* ✅ |
| Session B (owner) sees it? | **Yes** ✅ |
| Vector-store filter isolation (direct, no LLM) | 0 leak ✅ |

---

## 5. Durability & memory

| Check | Result |
|---|---|
| State survives a **real process restart** | ✅ Fresh process loaded lesson + message history from the checkpoint |
| Messages accumulate across turns | ✅ 2 → 4, persisted |
| `/evaluate` resumes the quiz after restart | ✅ From the durable checkpoint |

---

## 6. API / end-to-end suite — **7/7**

`/health` 200 · unknown-session → structured 400 · teach (apostrophe/A16 regression) ·
quiz → questions · evaluate → verdict · follow-up turn (memory) · SSE stream
(progress + final events).

---

## 7. Observability

| Check | Result |
|---|---|
| LangSmith traces | ✅ Live in project `studyhelper`; node tree `planner → executor → ChatGroq → parsers` |
| Retrieval/groundedness metrics | ✅ Emitted per request (`pass_rate`, `grounded`) |
| OpenTelemetry on FastAPI | ✅ Instrumented (exports when OTLP endpoint set) |

---

## 8. Production data layer (Neon pre-flight) — **6/6**

Direct (non-pooled) connection ✅ · `psycopg3` + **pgvector 0.8.0** ✅ ·
SQLAlchemy/`psycopg2` + **PostgreSQL 17.10** ✅ · `init_db` tables ✅ ·
PGVector store ✅ · PostgresSaver checkpointer ✅ *(bug found & fixed here — §9)*

---

## 9. Engineering wins

| Item | Impact |
|---|---|
| **Ablation benchmark** | Turned "we built loops" into **+153% precision, 4%→100% distractor rejection**, and surfaced a defect no other test caught |
| **fastembed (ONNX) over sentence-transformers (PyTorch)** | Same model, **~15 MB vs ~2.5 GB** — made a free-tier deploy viable and broke a dependency deadlock |
| **A16: structured-output salvage** | Groq `tool_use_failed` on apostrophes ("Newton's") → 500s. Now retried + **salvaged from the rejected generation**. Unit repro + live verification |
| **Checkpointer connection pool** | Pre-flight caught `from_conn_string().__enter__()` closing the connection — would have **silently killed memory in production** |
| **Degradation vs. quality** | A rate-limited grader used to fall back to "keep everything" and be scored as if it were a real grading decision. Now flagged and excluded |
| **Secret hygiene** | Scan across all commits: **0 secrets**; `.env` never committed |
| **Curated requirements** | Replaced a broken 200+ line `pip freeze` (missing imports, conflicting pins) with ~20 real deps + lockfile |

---

## 10. Known gaps (honest)

| Gap | Status |
|---|---|
| `docker compose up` never actually run | Docker not installed locally; Dockerfile/compose validated by inspection. Real test = the Render deploy |
| Ungrounded → regenerate never observed firing | Mechanism verified + capped; can't force it deterministically |
| Eval is 6 curated cases | Isolates failure modes well; not a large-scale benchmark |
| LLM answer-quality judges not in the headline run | Available via `--judge-answers`; deterministic metrics preferred for the ablation |
| No per-user auth | v1 relies on unguessable session tokens + optional `APP_API_KEY` |
| Deployed URL | Phase 9 in progress (Render + Neon) |

---

## 11. Cost

| Component | Tier | Cost |
|---|---|---|
| Groq (LLM) | Free — 100k tokens/day | $0 |
| Neon (Postgres + pgvector) | Free, non-expiring | $0 |
| LangSmith | Free tier | $0 |
| MLflow | Local (SQLite) | $0 |
| Embeddings | Local ONNX, no API | $0 |
| Render (web service) | Free — 750 hrs/mo | $0 |
| **Total** | | **$0** |

*The self-correcting loops cost several LLM calls per request (bounded by the caps).
Heavy testing exhausted the 100k/day Groq tier once — a real constraint worth knowing.*
