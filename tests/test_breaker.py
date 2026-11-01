"""Circuit breaker logic and its effect on touchpoint resolution (no Django)."""

from llm_kit.core.observability.breaker import CircuitBreaker, Policy
from llm_kit.core.touchpoints import touchpoint
from tests.test_touchpoints_core import FakeEnv


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make():
    clock = Clock()
    return CircuitBreaker(policy=Policy(failures=3, window=60, cooldown=30, quota_cooldown=600), clock=clock), clock


def test_opens_after_repeated_retryable_failures_within_window():
    b, clock = make()
    b.record("gemini", False, "rate_limit")
    b.record("gemini", False, "overloaded")
    assert not b.is_open("gemini")
    clock.t += 61  # first two fall out of the window
    b.record("gemini", False, "timeout")
    assert not b.is_open("gemini")
    b.record("gemini", False, "timeout")
    b.record("gemini", False, "connection")
    assert b.is_open("gemini") and b.state("gemini")["state"] == "open"
    assert b.state("gemini")["retry_in_s"] == 30


def test_non_retryable_errors_never_trip_it():
    b, _ = make()
    for _ in range(10):
        b.record("openai", False, "bad_request")
        b.record("openai", False, "auth")
    assert not b.is_open("openai") and b.states() == {}


def test_exhausted_quota_opens_at_once_for_longer():
    b, clock = make()
    b.record("gemini", False, "quota_exhausted")
    assert b.is_open("gemini") and b.state("gemini")["reason"] == "quota exhausted"
    clock.t += 300
    assert b.is_open("gemini")
    clock.t += 301
    assert not b.is_open("gemini")


def test_half_open_success_closes_failure_reopens():
    b, clock = make()
    for _ in range(3):
        b.record("anthropic", False, "overloaded")
    clock.t += 31
    assert not b.is_open("anthropic") and b.state("anthropic")["state"] == "half_open"
    b.record("anthropic", False, "overloaded")  # trial call failed
    assert b.is_open("anthropic")
    clock.t += 31
    b.record("anthropic", True)
    assert b.state("anthropic")["state"] == "closed" and "anthropic" not in b.states()


class BreakerEnv(FakeEnv):
    def __init__(self, down=(), **kw):
        super().__init__(**kw)
        self.down = set(down)

    def circuit_open(self, provider):
        return provider in self.down


def test_resolution_skips_open_circuits():
    t = touchpoint(
        "t.cb", default="anthropic/claude-sonnet-5-5", fallbacks=["gemini/gemini-3.8-flash", "openai/gpt-5.5"]
    )
    r = t.resolve(env=BreakerEnv(down={"anthropic"}))
    assert r.candidates == ["gemini/gemini-3.8-flash", "openai/gpt-5.5"]
    assert ("anthropic/claude-sonnet-5-5", "provider circuit open (recent failures)") in r.skipped
    r = t.resolve(env=BreakerEnv(down={"gemini"}))
    assert r.candidates == ["anthropic/claude-sonnet-5-5", "openai/gpt-5.5"]
    # Everything down: still try, in the original order.
    r = t.resolve(env=BreakerEnv(down={"anthropic", "gemini", "openai"}))
    assert r.model == "anthropic/claude-sonnet-5-5" and "trying anyway" in r.notes[-1]
