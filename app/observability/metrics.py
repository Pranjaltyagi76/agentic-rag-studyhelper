"""Retrieval-quality metrics (Phase 8).

Lightweight structured logging of the numbers that tell us whether the advanced-RAG
loops are actually helping: grade pass rate, retrieval attempts, groundedness result.
These lines are easy to scrape/aggregate and complement the LangSmith traces.
"""

import logging

logger = logging.getLogger("studyhelper.metrics")


def log_retrieval_grade(relevant: int, total: int, attempt: int) -> None:
    rate = (relevant / total) if total else 0.0
    logger.info(
        "retrieval_grade attempt=%d relevant=%d total=%d pass_rate=%.2f",
        attempt, relevant, total, rate,
    )


def log_groundedness(grounded, attempts: int) -> None:
    logger.info("groundedness grounded=%s attempts=%d", grounded, attempts)


def log_grade_degraded(error) -> None:
    """The grader fell back to 'keep everything' because the LLM call never succeeded.

    This is a DEGRADATION (rate limit, outage), not a grading decision — surface it so
    it is never mistaken for retrieval quality.
    """
    logger.warning(
        "retrieval_grade DEGRADED: grader unavailable, kept all chunks ungraded (%s: %s)",
        type(error).__name__, str(error)[:120],
    )
