"""Robust structured-output calls (audit A16).

Groq's function-calling parser intermittently returns ``tool_use_failed`` — often on
otherwise-valid generations whose JSON contains awkward escaping (e.g. ``\\'`` from a
word like "Newton's"). The model's actual output is usually fine; only Groq's tool
wrapper rejects it.

``structured_invoke`` wraps ``model.with_structured_output(schema).invoke(...)`` with:
  1. retries (for genuinely transient failures),
  2. salvage — pull the JSON out of the rejected ``failed_generation`` and validate it,
  3. a caller-supplied ``default`` so a node degrades gracefully instead of 500-ing.
"""

import json
import re

from app.config import settings
from app.agent.llm import model


def _failed_generation(err: Exception) -> str | None:
    """Extract Groq's ``failed_generation`` payload from an error, if present."""
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        fg = (body.get("error") or {}).get("failed_generation")
        if fg:
            return fg
    # Fallback: dig it out of the string representation.
    m = re.search(r"failed_generation'?\s*:\s*'(.+?)'\s*\}\s*\}?\s*$", str(err), re.S)
    return m.group(1) if m else None


def _salvage(err: Exception, schema):
    """Recover a schema instance from a rejected generation's embedded JSON."""
    fg = _failed_generation(err)
    if not fg:
        return None
    start, end = fg.find("{"), fg.rfind("}")
    if start == -1 or end <= start:
        return None
    raw = fg[start : end + 1].replace("\\'", "'")  # fix the invalid \' escape
    try:
        data = json.loads(raw)
        return schema.model_validate(data)
    except Exception:
        return None


def structured_invoke(schema, messages, *, default=None, on_fallback=None, llm=None):
    """Invoke an LLM for a structured `schema`, with retries + salvage + default.

    `llm` overrides the default reasoning model — pass the cheap grader model for
    classification-only steps (model cascading).

    `on_fallback(error)` is called if we end up returning `default` — i.e. the model
    never produced a usable result. Callers should use it to surface *degradation*
    (rate limits, outages) rather than let a fallback masquerade as a real answer.

    Raises the last error only if salvage fails on every attempt and no `default`
    was supplied.
    """
    runnable = (llm or model).with_structured_output(schema)
    last_err: Exception | None = None

    for _ in range(settings.STRUCTURED_MAX_RETRIES + 1):
        try:
            return runnable.invoke(messages)
        except Exception as e:  # noqa: BLE001 — we deliberately handle any LLM/tool error
            last_err = e
            salvaged = _salvage(e, schema)
            if salvaged is not None:
                return salvaged

    if default is not None:
        if on_fallback is not None:
            on_fallback(last_err)
        return default
    raise last_err
