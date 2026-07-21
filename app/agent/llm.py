"""Shared LLM instances (Groq) + explicit 429 rate-limit backoff.

Single source of the reasoning model so every node uses the same configured client
(strategy.md: "one place per concern").

Rate limiting: the Groq SDK will, by default, retry a 429 internally (twice) with an
opaque backoff. We turn that off (``max_retries=0``) and own a single, *visible*
backoff layer here — ``invoke_with_backoff`` — so a rate limit is logged (the project's
"degradation is never silent" principle), honors the server's ``Retry-After``, and is
bounded and configurable. Free-tier Groq (100k tokens/day) makes 429s a normal event
under load, not an exception, so this path matters.
"""

import logging
import random
import re
import time

from langchain_groq import ChatGroq

from app.config import settings

logger = logging.getLogger("studyhelper.llm")

# max_retries=0: we handle 429 backoff ourselves (below); disabling the SDK's own retry
# keeps it to one counted, logged layer instead of two nested ones.
model = ChatGroq(model=settings.GROQ_MODEL, max_retries=0)

# Cheap classifier for the grading step (model cascading). Separate Groq quota bucket,
# so relevance-grading no longer competes with generation for the daily token budget.
grader_model = ChatGroq(model=settings.GRADER_MODEL, max_retries=0)


def _status_code(err: BaseException) -> int | None:
    """HTTP status carried by err or any error in its cause/context chain."""
    e: BaseException | None = err
    seen: set[int] = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        code = getattr(e, "status_code", None)
        if isinstance(code, int):
            return code
        e = e.__cause__ or e.__context__
    return None


def _is_rate_limit(err: BaseException) -> bool:
    """True if err (or a cause in its chain) is a 429 rate-limit error."""
    if _status_code(err) == 429:
        return True
    e: BaseException | None = err
    seen: set[int] = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if type(e).__name__ == "RateLimitError":
            return True
        e = e.__cause__ or e.__context__
    s = str(err).lower()
    return "rate limit" in s or "429" in s


def _is_retryable(err: BaseException) -> bool:
    """Retry a 429 (rate limit) or a transient 5xx (the SDK used to retry these)."""
    if _is_rate_limit(err):
        return True
    code = _status_code(err)
    return code is not None and code >= 500


def _retry_after_seconds(err: BaseException) -> float | None:
    """Delay the server asked for — Retry-After header, else the body message."""
    e: BaseException | None = err
    seen: set[int] = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        resp = getattr(e, "response", None)
        headers = getattr(resp, "headers", None)
        if headers is not None:
            try:
                ra = headers.get("retry-after")
            except Exception:
                ra = None
            if ra:
                try:
                    return float(ra)  # Groq sends seconds; ignore HTTP-date form
                except (TypeError, ValueError):
                    pass
        e = e.__cause__ or e.__context__
    # Groq's body usually reads: "Please try again in 7.29s"
    m = re.search(r"try again in\s*([0-9.]+)\s*s", str(err), re.IGNORECASE)
    return float(m.group(1)) if m else None


def invoke_with_backoff(runnable, messages):
    """Invoke a Groq runnable, backing off on 429 (and transient 5xx) errors.

    Waits the server's Retry-After when present, otherwise capped exponential backoff
    with jitter. Non-retryable errors propagate immediately, so callers' own handling
    (e.g. structured_invoke's salvage) still runs. Bounded by
    ``settings.RATE_LIMIT_MAX_RETRIES`` so it always terminates; the final failure is
    re-raised.
    """
    attempts = settings.RATE_LIMIT_MAX_RETRIES
    for i in range(attempts + 1):
        try:
            return runnable.invoke(messages)
        except Exception as e:  # noqa: BLE001 — classify, then re-raise if not retryable
            if not _is_retryable(e) or i >= attempts:
                raise
            server = _retry_after_seconds(e)
            backoff = min(settings.RATE_LIMIT_BASE_DELAY * (2 ** i), settings.RATE_LIMIT_MAX_DELAY)
            delay = (server if server is not None else backoff) + random.uniform(0, 0.5)
            logger.warning(
                "groq throttled (status=%s); backing off %.2fs before retry %d/%d (retry_after=%s)",
                _status_code(e), delay, i + 1, attempts, server,
            )
            time.sleep(delay)
