# Requirements — Advanced Agentic RAG Study Helper

> Companion to [ARCHITECTURE.md](ARCHITECTURE.md). This file is the source of truth
> for **what** we must build. Architecture/design describe **how**.

---

## 1. Product goal

An AI study assistant that can **teach** a topic, **quiz** the user, and **evaluate**
answers — grounded in the user's uploaded notes, falling back to web research, and
self-correcting when retrieval or generation is weak. Delivered as a deployable,
observable, multi-user API.

---

## 2. Functional requirements

### FR-1 Ingestion
- FR-1.1 Accept PDF upload via API.
- FR-1.2 Extract text; if the PDF is scanned/empty-text, OCR via vision model
  (already done in `ingest.py`).
- FR-1.3 Chunk (size 1000 / overlap 200) with metadata `{ session_id, file_name, chunk_id }`.
- FR-1.4 Embed and store in the vector DB.
- FR-1.5 Return ingestion status (queued / processing / done / failed) + chunk count.

### FR-2 Teaching
- FR-2.1 Decide whether notes retrieval is needed (`use_rag`).
- FR-2.2 Retrieve, **grade relevance**, and retry with a rewritten query if weak.
- FR-2.3 Fall back to web research when notes are insufficient.
- FR-2.4 Synthesize one coherent lesson.
- FR-2.5 **Verify groundedness** of the lesson against sources; regenerate or flag
  uncertainty if ungrounded.

### FR-3 Quiz generation
- FR-3.1 Decide notes vs. general knowledge.
- FR-3.2 Generate exactly the requested number of questions (mcq / true_false / short_answer).
- FR-3.3 Each question carries id, correct answer, type, explanation, options (mcq).

### FR-4 Quiz evaluation
- FR-4.1 Evaluate a user's answer for a given `question_id`.
- FR-4.2 mcq/true_false → Correct/Wrong; short_answer → 0–5 rating with justification.

### FR-5 Planning / orchestration
- FR-5.1 Decompose a request into ordered atomic tasks.
- FR-5.2 Route each task to teacher / quiz_generator / quiz_eval.
- FR-5.3 **Adaptive re-planning**: revise the plan based on results (e.g. insert a
  research task when teaching fails).

### FR-6 Sessions & memory
- FR-6.1 Every request is scoped to a `session_id` / `thread_id`.
- FR-6.2 Conversation `messages` persist across requests within a session.
- FR-6.3 A user's documents and retrieval are isolated to their session.
- FR-6.4 Runs are resumable after a crash/restart (checkpointing).

### FR-7 API
- FR-7.1 `POST /upload`, `POST /chat`, `POST /evaluate`, `GET /health`.
- FR-7.2 Streaming responses (node-by-node progress) via SSE or WebSocket.
- FR-7.3 Consistent error envelope + status codes.

---

## 3. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | **Multi-user isolation** | No global mutable state; two users never see each other's data. |
| NFR-2 | **Persistence** | State survives process restart (Postgres). |
| NFR-3 | **Observability** | Every node + LLM call traced in **LangSmith**; retrieval-quality metrics captured. |
| NFR-9 | **Evaluability** | RAG quality (retrieval relevance, groundedness) measurable offline via **MLflow** against a test set; results tracked across changes. |
| NFR-4 | **Portability** | Runs via `docker-compose up` locally; deploys as a Docker image to Hugging Face Spaces. |
| NFR-10 | **Zero cost** | Entire stack runs on free tiers (HF Spaces + Neon + LangSmith + local MLflow). No paid dependency. |
| NFR-5 | **Configurability** | All secrets/URLs via env; nothing hard-coded or baked into images. |
| NFR-6 | **Latency** | Streaming first-token < 3s for cached/simple paths (best-effort). |
| NFR-7 | **Resilience** | Retrieval retry cap N; generation regenerate cap M; graceful web-search failure. |
| NFR-8 | **Testability** | Nodes and subgraphs unit-testable in isolation. |

---

## 4. Constraints & assumptions

- **Models**: Groq `llama-3.3-70b-versatile` (reasoning), HF `all-MiniLM-L6-v2`
  (embeddings — canonical, both ingest + query), Gemini for PDF OCR, Tavily for web search.
  *(Note: `app.py:22` references a HF model that must be validated/removed before use.)*
- **Observability**: LangSmith. **Evaluation**: MLflow.
- **Vector store**: `pgvector` (canonical, on Neon) for deploy; Chroma for local dev only.
- **Auth**: out of scope for v1 (session_id is client-supplied); add later.
- **Single language**: English UI/prompts for v1.
- **Budget = $0 (hard constraint)**: every tool/host/service must run on a free tier.
  Deploy stack: Hugging Face Spaces (app, Docker) + Neon (Postgres + pgvector) +
  LangSmith (free) + MLflow (local). Free-tier trade-offs (idle sleep, cold starts,
  storage caps) are acceptable; paid-only requirements are not.

---

## 5. Out of scope (v1)

- User authentication / accounts.
- Non-PDF ingestion (docx, web pages, images-as-notes).
- Multi-tenant billing / rate limiting.
- Fine-tuning any model.

---

## 6. Acceptance criteria (definition of done)

- [ ] Two concurrent sessions cannot read each other's notes or state.
- [ ] Killing and restarting the app resumes an in-flight session's memory.
- [ ] Retrieval grading demonstrably drops irrelevant chunks (traced).
- [ ] A deliberately ungrounded generation triggers regenerate/flag.
- [ ] `docker-compose up` yields a working `/health`, `/upload`, `/chat`, `/evaluate`.
- [ ] Every request produces a trace in LangSmith.
- [ ] Deployed URL serves the same flow as local.
