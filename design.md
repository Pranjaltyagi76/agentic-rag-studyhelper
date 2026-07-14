# Technical Design — Advanced Agentic RAG Study Helper

> Companion to [ARCHITECTURE.md](ARCHITECTURE.md) (the map) and
> [requirements.md](requirements.md) (the what). This file is the **detailed
> contracts**: state shape, node I/O, subgraph logic, API schemas.

---

## 1. Agent state

```python
class AgentState(TypedDict):
    # identity / memory
    session_id: str
    thread_id: str
    messages: Annotated[list[BaseMessage], add_messages]

    # planning
    query: str
    plan: PlannerState | None
    current_task_index: int
    replans: int                    # guard against infinite re-planning

    # documents
    uploaded_files: list[str]

    # teaching / retrieval outputs
    lesson: str | None
    retrieval_attempts: int         # cap = N
    generation_attempts: int        # cap = M
    grounded: bool | None

    # quiz
    quiz: Quiz | None
    current_question_id: int | None
    user_answer: str | None
    quiz_evaluation: QuizEval | None
```

Schemas `task`, `PlannerState`, `Quiz`, `QuizQuestion`, `QuizEval` carry over from
`Agent.py` unchanged except where noted below.

---

## 2. Node contracts

Each node is a pure `(state) -> partial_state` function. Contracts:

| Node | Reads | Writes | Notes |
|---|---|---|---|
| `planner` | query, plan, results-so-far, replans | plan, replans | Adaptive; may add/skip/reorder. Cap replans. |
| `teacher` | query, uploaded_files, retrieved docs | lesson, grounded, attempts, current_task_index | Wraps retrieval subgraph + verify loop. |
| `quiz_generator` | query, uploaded_files, docs | quiz, current_task_index | Wraps retrieval subgraph. |
| `quiz_eval` | quiz, current_question_id, user_answer | quiz_evaluation, current_task_index | No retrieval. |

**Routing** (`executor`): if `current_task_index >= len(plan.tasks)` → END, else
route to `plan.tasks[current_task_index].tool`. Add a **replan hook**: before END,
let the planner decide whether the goal is actually met.

---

## 3. Retrieval subgraph (shared)

Reusable subgraph consumed by `teacher` and `quiz_generator`.

### Nodes
1. **`plan_retrieval`** — LLM fills a `RetrievalPlan`:
   ```python
   class RetrievalPlan(BaseModel):
       use_rag: bool
       filename: str | None
       rag_query: str | None
       number_chunks: int | None
       use_web: bool | None
       web_queries: list[str] | None
   ```
2. **`retrieve`** — `RAG_Tool` with filter `{ session_id, file_name }`
   (⚠️ today it filters `file_name` only — `Agent.py:226`).
3. **`grade_docs`** — LLM scores each chunk `relevant | irrelevant`; keep relevant.
   ```python
   class DocGrade(BaseModel):
       chunk_id: str
       verdict: Literal["relevant", "irrelevant"]
   ```
4. **`decide_next`** — router:
   - enough relevant chunks → `done`
   - too few & `retrieval_attempts < N` → `rewrite` (regenerate `rag_query`) → `retrieve`
   - too few & attempts exhausted → `web_fallback`
   - web done or unavailable → `done` (may mark `insufficient`)
5. **`web_fallback`** — `research_tool(web_queries)`.

### Termination
- Hard cap `N` retrieval loops (default 2) + 1 web fallback. Always terminates.

---

## 4. Generate + verify loop (teacher)

1. **`generate_lesson`** — synthesize lesson from docs + web (existing prompt).
2. **`check_groundedness`** — LLM verdict:
   ```python
   class Groundedness(BaseModel):
       grounded: bool
       unsupported_claims: list[str]
   ```
3. **`decide`**:
   - grounded → return
   - not grounded & `generation_attempts < M` → regenerate with the unsupported
     claims fed back
   - attempts exhausted → return lesson **with an uncertainty note**

---

## 5. Adaptive planner detail

- Initial call: same as today (`PlannerNode`, `Agent.py:66`).
- After each task node, re-enter planner with a summary of results. Planner may:
  - mark the plan complete (→ END),
  - append a task (e.g. `teacher` produced "insufficient" → add a `teacher` retry
    with web forced, or a research task),
  - skip a now-redundant task.
- Guard: `replans` cap (default 3) → force END to avoid loops.

---

## 6. API contracts (FastAPI)

### `POST /upload`
```
multipart: file, session_id
→ 200 { file_name, status, chunk_count }
```

### `POST /chat`
```
json: { session_id, query }
→ 200 stream (SSE): events { node, partial_state } ... final { state }
```

### `POST /evaluate`
```
json: { session_id, question_id, user_answer }
→ 200 { quiz_evaluation }
```

### `GET /health`
```
→ 200 { status: "ok", db: "up", vectorstore: "up" }
```

### Error envelope
```
→ 4xx/5xx { error: { code, message, detail? } }
```

---

## 7. Persistence design

- **Checkpointer**: `PostgresSaver(thread_id=session_id)` wraps the compiled graph.
  Enables FR-6.2 (memory) and FR-6.4 (resumability).
- **Tables**: `sessions`, `messages`, `documents` (see ARCHITECTURE §3).
- **Vector metadata** must include `session_id`; all retrieval filters on it.

---

## 8. Key design decisions & rationale

| Decision | Why | Alternative rejected |
|---|---|---|
| Retrieval as a **shared subgraph** | DRY across teacher & quiz; single place to tune | Duplicated inline logic (current state) |
| **Caps** on every loop (N/M/replans) | Guarantees termination, bounds cost | Unbounded self-correction |
| **PostgresSaver** for memory | One store for state + resumability | In-memory (loses NFR-2) |
| **session_id in vector filter** | Multi-user isolation (NFR-1) | file_name-only (data leak) |
| **pgvector on Neon (not Chroma)** | Free deploy needs no paid disk; one datastore for state + vectors | Chroma-on-disk (needs paid persistent disk) |
| **HF Spaces + Neon host** | Truly free & durable | Render (free DB expires 30d, disk paid) |
| **SSE streaming** | Simple, HTTP-native progress | WebSocket (more moving parts for v1) |

---

## 9. Open questions

- ~~Which embedding model is canonical?~~ **RESOLVED (2026-07-14): HF
  `all-MiniLM-L6-v2` for both ingest and query.** `ingest.py` must stop using
  Google `text-embedding-004` and share the embedder from `database.py` (384-dim).
  Any docs embedded under the old model must be re-ingested.
- Validate the HF chat model id in `app.py:22` (`DeepSeek-V4-Flash`) — appears
  unused/possibly invalid; likely remove.
- ~~Chroma vs pgvector for the deployed target.~~ **RESOLVED (2026-07-14): pgvector
  on Neon is canonical for deploy (free, no paid disk, one datastore for state +
  vectors). Chroma is local-dev only.**
