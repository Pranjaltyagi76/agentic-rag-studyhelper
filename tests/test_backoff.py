"""Groq 429 rate-limit backoff (app/agent/llm.py:invoke_with_backoff).

All sleeps are monkeypatched to a recorder, so these run instantly while still asserting
the delay decisions (retry-after honored, exponential otherwise, bounded retries).
"""

import pytest
from pydantic import BaseModel

import app.agent.llm as llm_mod
from app.agent.llm import invoke_with_backoff, _is_rate_limit, _retry_after_seconds


class _Resp:
    def __init__(self, headers):
        self.headers = headers


class _Err429(Exception):
    """Stand-in for groq.RateLimitError (status_code=429 + an httpx-like response)."""

    status_code = 429

    def __init__(self, retry_after=None):
        super().__init__("rate limit reached")
        self.response = _Resp({"retry-after": retry_after} if retry_after else {})


class _Err500(Exception):
    status_code = 503

    def __init__(self):
        super().__init__("service unavailable")


class _Runnable:
    """Fails its first `fail_times` invokes with `err_factory`, then returns `value`."""

    def __init__(self, fail_times, err_factory, value="ok"):
        self.fail_times = fail_times
        self.err_factory = err_factory
        self.value = value
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.err_factory()
        return self.value


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: slept.append(s))
    return slept


def test_detection_helpers():
    assert _is_rate_limit(_Err429()) is True
    assert _is_rate_limit(ValueError("nope")) is False
    assert _retry_after_seconds(_Err429(retry_after="4")) == 4.0
    # Falls back to parsing the body message when there's no header.
    assert _retry_after_seconds(Exception("Please try again in 7.5s")) == 7.5


def test_retries_then_succeeds(no_sleep):
    r = _Runnable(fail_times=2, err_factory=_Err429)
    assert invoke_with_backoff(r, []) == "ok"
    assert r.calls == 3          # 2 failures + 1 success
    assert len(no_sleep) == 2    # slept before each retry


def test_honors_retry_after(no_sleep):
    r = _Runnable(fail_times=1, err_factory=lambda: _Err429(retry_after="3"))
    assert invoke_with_backoff(r, []) == "ok"
    # Delay is the server's 3s (plus <=0.5s jitter), not the exponential default.
    assert 3.0 <= no_sleep[0] <= 3.5


def test_non_rate_limit_propagates_immediately(no_sleep):
    r = _Runnable(fail_times=1, err_factory=lambda: ValueError("bad request"))
    with pytest.raises(ValueError):
        invoke_with_backoff(r, [])
    assert r.calls == 1          # no retry
    assert no_sleep == []        # no backoff


def test_transient_5xx_is_retried(no_sleep):
    r = _Runnable(fail_times=1, err_factory=_Err500)
    assert invoke_with_backoff(r, []) == "ok"
    assert r.calls == 2


def test_bounded_and_reraises(no_sleep, monkeypatch):
    monkeypatch.setattr(llm_mod.settings, "RATE_LIMIT_MAX_RETRIES", 3)
    r = _Runnable(fail_times=99, err_factory=_Err429)  # never succeeds
    with pytest.raises(_Err429):
        invoke_with_backoff(r, [])
    assert r.calls == 4          # initial + 3 retries
    assert len(no_sleep) == 3


def test_structured_invoke_defaults_on_persistent_rate_limit(no_sleep, monkeypatch):
    """A 429 that outlasts backoff falls through to the caller's default (no salvage spin)."""
    monkeypatch.setattr(llm_mod.settings, "RATE_LIMIT_MAX_RETRIES", 1)

    from app.agent.structured import structured_invoke

    class _Schema(BaseModel):
        value: int

    class _AlwaysThrottled:
        def with_structured_output(self, schema):
            return _Runnable(fail_times=99, err_factory=_Err429)

    seen = []
    out = structured_invoke(
        _Schema, [], llm=_AlwaysThrottled(),
        default=_Schema(value=-1), on_fallback=lambda e: seen.append(e),
    )
    assert out.value == -1
    assert len(seen) == 1 and _is_rate_limit(seen[0])
