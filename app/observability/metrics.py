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
