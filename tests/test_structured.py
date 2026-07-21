"""Robust structured-output calls (app/agent/structured.py, audit A16).

Covers the three behaviours that keep a node from 500-ing on Groq's flaky tool parser:
a normal result, salvaging JSON out of a rejected ``failed_generation``, and falling
back to a caller-supplied default (with the ``on_fallback`` degradation hook).
"""

import pytest
from pydantic import BaseModel

from app.agent.structured import structured_invoke


class _Schema(BaseModel):
    value: int


class _Runnable:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, messages):
        return self._fn(messages)


class _LLM:
    """Stand-in for a chat model: with_structured_output(...).invoke(...)."""

    def __init__(self, fn):
        self._fn = fn

    def with_structured_output(self, schema):
        return _Runnable(self._fn)


class _GroqToolError(Exception):
    """Mimics Groq's tool_use_failed error carrying the raw generation in .body."""

    def __init__(self, failed_generation):
        super().__init__("tool_use_failed")
        self.body = {"error": {"failed_generation": failed_generation}}


def test_returns_structured_result():
    out = structured_invoke(_Schema, [], llm=_LLM(lambda m: _Schema(value=7)))
    assert out.value == 7


def test_salvages_json_from_failed_generation():
    def boom(_):
        # Note the invalid \' escape that structured._salvage is meant to repair.
        raise _GroqToolError('{"value": 5, "note": "Newton\\\'s law"}')

    out = structured_invoke(_Schema, [], llm=_LLM(boom))
    assert out.value == 5


def test_falls_back_to_default_and_fires_on_fallback():
    seen = []

    def boom(_):
        raise RuntimeError("transient outage")

    out = structured_invoke(
        _Schema,
        [],
        llm=_LLM(boom),
        default=_Schema(value=-1),
        on_fallback=lambda e: seen.append(e),
    )
    assert out.value == -1
    assert len(seen) == 1 and isinstance(seen[0], RuntimeError)


def test_raises_when_unrecoverable_and_no_default():
    def boom(_):
        raise RuntimeError("transient outage")

    with pytest.raises(RuntimeError):
        structured_invoke(_Schema, [], llm=_LLM(boom))
