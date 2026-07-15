"""Deterministic retrieval metrics (Phase 8.5).

These are computed from the dataset's ground-truth `relevant` labels — no LLM judge,
so the numbers are reproducible and can't be flattered by a generous grader.

Given the chunk set a pipeline actually passed to the generator:
  precision = of what we kept, how much was actually relevant  (↑ = less noise/hallucination fuel)
  recall    = of what was relevant, how much did we keep       (↑ = less missing context)
  f1        = harmonic mean
  distractor_rejection = of the irrelevant chunks present, how many did we correctly drop
"""


def prf1(returned: list[str], relevant: set[str]) -> dict:
    ret = set(returned)
    if not ret:
        # Returning nothing: precision is undefined -> 1.0 by convention (no noise),
        # recall is 0 unless there was nothing to find.
        return {
            "precision": 1.0,
            "recall": 1.0 if not relevant else 0.0,
            "f1": 1.0 if not relevant else 0.0,
        }
    tp = len(ret & relevant)
    precision = tp / len(ret)
    recall = (tp / len(relevant)) if relevant else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def distractor_rejection(returned: list[str], all_chunks: list[str], relevant: set[str]) -> float:
    """Fraction of the irrelevant chunks that were correctly kept OUT of the context."""
    irrelevant = [c for c in all_chunks if c not in relevant]
    if not irrelevant:
        return 1.0
    kept = set(returned)
    rejected = sum(1 for c in irrelevant if c not in kept)
    return rejected / len(irrelevant)


def mean(xs: list[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0
