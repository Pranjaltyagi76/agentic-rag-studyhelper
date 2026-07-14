"""Offline RAG evaluation harness (Phase 8.5).

For each example: seed its note chunks (in an isolated per-run session), run the
self-correcting retrieval + teacher generation, then score three judges
(retrieval relevance, faithfulness, answer correctness). Aggregate metrics + params
are logged to MLflow so runs are comparable over time.

Run:  python evaluation/run_eval.py [--label my-variant]

Compare runs / open the UI:  mlflow ui   (then visit http://127.0.0.1:5000)
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())  # so `import app` works when run from the repo root
from dotenv import load_dotenv

load_dotenv(os.path.join(os.getcwd(), ".env"))

import mlflow
from langchain_core.documents import Document

from app.config import settings
from app.persistence.vectorstore import vectordb
from app.agent.retrieval import run_retrieval
from app.agent.teacher import _generate_lesson
from evaluation.dataset import EVAL_SET
from evaluation.judges import retrieval_relevance, faithfulness, answer_correctness


def _seed(session_id: str, example: dict) -> None:
    docs = [
        Document(page_content=c, metadata={"session_id": session_id, "file_name": example["file_name"]})
        for c in example["chunks"]
    ]
    vectordb.add_documents(docs)


def evaluate(label: str) -> dict:
    run_id = str(int(time.time()))
    per_example = []

    for ex in EVAL_SET:
        session_id = f"eval-{run_id}-{ex['id']}"
        _seed(session_id, ex)

        ctx = run_retrieval(
            session_id=session_id,
            query=ex["question"],
            uploaded_files=[ex["file_name"]],
            purpose="teach",
            allow_web=False,
        )
        notes = ctx.get("notes_text", "")
        answer = _generate_lesson(ex["question"], notes, "", "", "")

        scores = {
            "retrieval_relevance": retrieval_relevance(ex["question"], ex["reference"], notes),
            "faithfulness": faithfulness(answer, notes),
            "answer_correctness": answer_correctness(ex["question"], ex["reference"], answer),
        }
        per_example.append({"id": ex["id"], **scores, "retrieval_attempts": ctx.get("attempts", 0)})
        print(f"  {ex['id']:<20} {scores}")

    metrics = {
        f"avg_{k}": statistics.mean(e[k] for e in per_example)
        for k in ("retrieval_relevance", "faithfulness", "answer_correctness")
    }
    return {"label": label, "per_example": per_example, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="default", help="Variant label for this run.")
    args = parser.parse_args()

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=args.label):
        # Params: the knobs a change would alter — so runs are comparable.
        mlflow.log_params({
            "label": args.label,
            "groq_model": settings.GROQ_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "retrieval_max_attempts": settings.RETRIEVAL_MAX_ATTEMPTS,
            "generation_max_attempts": settings.GENERATION_MAX_ATTEMPTS,
            "num_examples": len(EVAL_SET),
        })

        print(f"Running eval '{args.label}' over {len(EVAL_SET)} examples...")
        result = evaluate(args.label)

        mlflow.log_metrics(result["metrics"])
        mlflow.log_dict({"per_example": result["per_example"]}, "per_example.json")

        print("\nAggregate metrics:")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v:.2f}")
        print(f"\nLogged to MLflow experiment '{settings.MLFLOW_EXPERIMENT_NAME}' "
              f"(tracking: {settings.MLFLOW_TRACKING_URI}).")
        print(f"Compare runs:  mlflow ui --backend-store-uri {settings.MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
