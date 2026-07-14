# RAG Evaluation (Phase 8.5)

Offline quality scoring for the agentic-RAG pipeline, logged to MLflow so prompt /
retrieval changes can be compared over time.

## What it measures

For each example in `dataset.py` it runs the self-correcting retrieval + teacher
generation, then scores three LLM-judge metrics (1–5):

- **retrieval_relevance** — do the retrieved notes contain what's needed to answer?
- **faithfulness** — is the generated answer grounded in those notes (not hallucinated)?
- **answer_correctness** — how well does the answer match the reference?

## Run it

```bash
python evaluation/run_eval.py --label baseline
```

Change a prompt or a knob (e.g. `RETRIEVAL_MAX_ATTEMPTS`), then run again with a new
label to get a comparable run:

```bash
python evaluation/run_eval.py --label more-retries
```

## Compare runs

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000  → experiment "studyhelper-rag-eval"
```

Each run logs the config as **params** and the averaged scores as **metrics**, plus
per-example scores as a `per_example.json` artifact — so a change's delta is visible
side by side.

## Files

- `dataset.py` — curated eval set (note chunks + question + reference answer).
- `judges.py` — LLM-as-judge scorers (reuse the app's robust structured-output helper).
- `run_eval.py` — harness: seed → retrieve → generate → judge → log to MLflow.
