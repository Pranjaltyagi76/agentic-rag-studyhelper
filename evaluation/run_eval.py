"""RAG evaluation + ablation harness (Phase 8.5).

Answers the question that actually matters: **does the self-correcting pipeline beat
naive RAG, and by how much?**

Two variants are run over identical seeded data:

  naive     : top-k similarity retrieval, no grading, no rewrite/retry  (classic RAG)
  advanced  : the full subgraph — plan -> retrieve -> grade -> rewrite/retry -> web

Headline metrics are **deterministic**, computed from the dataset's ground-truth
`relevant` labels (no LLM judge can flatter them):

  precision / recall / f1     of the chunk set handed to the generator
  distractor_rejection        share of irrelevant chunks correctly kept out
  hallucination_rate          on unanswerable questions, how often it invented an answer
  retrieval_latency_s         wall-clock cost of the extra machinery

Run:
    python evaluation/run_eval.py --variant advanced
    python evaluation/run_eval.py --variant naive
    python evaluation/run_eval.py --variant advanced --judge-answers   # + LLM quality judges

Compare:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import os
import sys
import time
import uuid

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv

load_dotenv(os.path.join(os.getcwd(), ".env"))

import mlflow
from langchain_core.documents import Document

from app.config import settings
from app.persistence.vectorstore import vectordb
from app.agent.retrieval import run_retrieval, RAG_Tool
from app.agent.teacher import _generate_lesson
from evaluation.dataset import EVAL_SET, relevant_texts, all_texts
from evaluation.retrieval_metrics import prf1, distractor_rejection, mean
from evaluation import judges

# A realistic naive baseline: fixed top-k similarity, no grading, no retry.
NAIVE_K = 4


def _seed(session_id: str, example: dict) -> None:
    vectordb.add_documents([
        Document(page_content=c["text"],
                 metadata={"session_id": session_id, "file_name": example["file_name"]})
        for c in example["chunks"]
    ])


def _retrieve(variant: str, session_id: str, example: dict) -> tuple[list[str], int, bool]:
    """Return (chunk texts handed to the generator, retrieval attempts, degraded).

    `degraded=True` means the grader's LLM call never succeeded (rate limit/outage) so
    chunks were kept ungraded. Such rows are NOT valid quality measurements.
    """
    if variant == "naive":
        docs = RAG_Tool(
            query=example["question"],          # raw question, no LLM rewrite
            filename=example["file_name"],
            k=NAIVE_K,
            session_id=session_id,
        )
        return [d.page_content for d in docs], 1, False

    ctx = run_retrieval(
        session_id=session_id,
        query=example["question"],
        uploaded_files=[example["file_name"]],
        purpose="teach",
        allow_web=False,                        # keep the comparison to the notes only
    )
    return (
        [d.page_content for d in (ctx.get("relevant_docs") or [])],
        ctx.get("attempts", 0),
        bool(ctx.get("grade_degraded")),
    )


def evaluate(variant: str, judge_answers: bool) -> dict:
    run_id = uuid.uuid4().hex[:8]
    rows = []

    for ex in EVAL_SET:
        session_id = f"eval-{run_id}-{ex['id']}"
        _seed(session_id, ex)

        t0 = time.perf_counter()
        returned, attempts, degraded = _retrieve(variant, session_id, ex)
        latency = time.perf_counter() - t0

        gold = relevant_texts(ex)
        row = {
            "id": ex["id"],
            "type": ex["type"],
            **prf1(returned, gold),
            "distractor_rejection": distractor_rejection(returned, all_texts(ex), gold),
            "retrieval_attempts": attempts,
            "retrieval_latency_s": round(latency, 2),
            "chunks_kept": len(returned),
            "degraded": degraded,
        }

        # Hallucination probe: unanswerable questions must NOT get an invented answer.
        if not ex["answerable"]:
            notes = "\n\n".join(returned)
            answer = _generate_lesson(ex["question"], notes, "", "", "")
            row["abstained"] = judges.abstained(ex["question"], answer)

        # Optional LLM quality judges (costs tokens; off by default).
        elif judge_answers:
            notes = "\n\n".join(returned)
            answer = _generate_lesson(ex["question"], notes, "", "", "")
            row["faithfulness"] = judges.faithfulness(answer, notes)
            row["answer_correctness"] = judges.answer_correctness(ex["question"], ex["reference"], answer)

        rows.append(row)
        print(f"  {ex['id']:<34} P={row['precision']:.2f} R={row['recall']:.2f} "
              f"F1={row['f1']:.2f} rej={row['distractor_rejection']:.2f}"
              + (" [DEGRADED]" if degraded else "")
              + (f" abstained={row['abstained']}" if 'abstained' in row else ""))

    unanswerable_ids = {e["id"] for e in EVAL_SET if not e["answerable"]}
    # Degraded rows measure an outage, not retrieval quality — exclude them from the
    # headline numbers and report the count instead of quietly skewing the average.
    valid = [r for r in rows if not r["degraded"]]
    answerable = [r for r in valid if r["id"] not in unanswerable_ids]
    unanswerable = [r for r in valid if "abstained" in r]

    metrics = {
        "avg_precision": mean([r["precision"] for r in answerable]),
        "avg_recall": mean([r["recall"] for r in answerable]),
        "avg_f1": mean([r["f1"] for r in answerable]),
        "avg_distractor_rejection": mean([r["distractor_rejection"] for r in valid]),
        "avg_retrieval_latency_s": mean([r["retrieval_latency_s"] for r in valid]),
        "avg_chunks_kept": mean([float(r["chunks_kept"]) for r in valid]),
        "degraded_examples": float(sum(1 for r in rows if r["degraded"])),
        "valid_examples": float(len(valid)),
    }
    if unanswerable:
        metrics["hallucination_rate"] = mean([0.0 if r["abstained"] else 1.0 for r in unanswerable])
        metrics["abstention_rate"] = mean([1.0 if r["abstained"] else 0.0 for r in unanswerable])
    if judge_answers:
        fa = [r["faithfulness"] for r in rows if "faithfulness" in r]
        ac = [r["answer_correctness"] for r in rows if "answer_correctness" in r]
        if fa:
            metrics["avg_faithfulness"] = mean(fa)
            metrics["avg_answer_correctness"] = mean(ac)

    return {"per_example": rows, "metrics": metrics}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["advanced", "naive"], default="advanced")
    p.add_argument("--judge-answers", action="store_true", help="Also run LLM quality judges (costs tokens).")
    args = p.parse_args()

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=args.variant):
        mlflow.log_params({
            "variant": args.variant,
            "groq_model": settings.GROQ_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "retrieval_max_attempts": settings.RETRIEVAL_MAX_ATTEMPTS if args.variant == "advanced" else 0,
            "grading_enabled": args.variant == "advanced",
            "naive_k": NAIVE_K if args.variant == "naive" else None,
            "num_examples": len(EVAL_SET),
        })

        print(f"\nRunning '{args.variant}' over {len(EVAL_SET)} examples...")
        result = evaluate(args.variant, args.judge_answers)

        mlflow.log_metrics(result["metrics"])
        mlflow.log_dict({"per_example": result["per_example"]}, "per_example.json")

        print(f"\n=== {args.variant.upper()} ===")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v:.3f}")
        print(f"\nLogged to MLflow ({settings.MLFLOW_TRACKING_URI}).")
        print(f"Compare:  mlflow ui --backend-store-uri {settings.MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
