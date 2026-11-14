"""Waiting out short rate limits before falling back, and the error when every model fails."""

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from llm_kit.core import config as settings
from llm_kit.core.errors import AllModelsFailed
from llm_kit.core.observability.classify import classify_error, is_retryable
from llm_kit.core.touchpoints import RetryPolicy, backends, retry_after, touchpoint
from llm_kit.core.touchpoints.backends import TouchpointChatModel
from llm_kit.core.touchpoints.retry import RETRY_KEY, stated_delay
from tests.test_touchpoints_core import FakeEnv


class RateLimitError(Exception):
    status_code = 429


GEMINI_MINUTE = (
    "429 RESOURCE_EXHAUSTED. You exceeded your current quota, please check your plan and billing details. "
    "quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 4.2s. "
    '"retryDelay": "4s"'
)
GEMINI_DAY = (
    "429 RESOURCE_EXHAUSTED. You exceeded your current quota, please check your plan and billing details. "
    'quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier. "retryDelay": "31s"'
)

# ref -> errors to raise before answering (popped one per call)
FAILURES: dict[str, list[BaseException]] = {}
CALLS: list[str] = []


class Scripted(GenericFakeChatModel):
    ref: str = ""

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        CALLS.append(self.ref)
        queue = FAILURES.get(self.ref) or []
        if queue:
            raise queue.pop(0)
        self.messages = iter([AIMessage(content=f"{self.ref}: {messages[-1].content}")])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kw)

    def _stream(self, messages, stop=None, run_manager=None, **kw):
        CALLS.append(self.ref)
        queue = FAILURES.get(self.ref) or []
        if queue:
            raise queue.pop(0)
        self.messages = iter([AIMessage(content=f"{self.ref}: {messages[-1].content}")])
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kw)

    def bind_tools(self, tools, **kw):
        return self.bind(tools=[t.name for t in tools])


def backend(ref, params, keys):
    return Scripted(messages=iter([]), ref=ref)


SLEPT: list[float] = []


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    FAILURES.clear()
    CALLS.clear()
    SLEPT.clear()

    async def asleep(seconds):
        SLEPT.append(seconds)

    monkeypatch.setattr(backends, "_sleep", SLEPT.append)
    monkeypatch.setattr(backends, "_asleep", asleep)


def tp(**kw):
    kw.setdefault("default", "anthropic/claude-sonnet-5-5")
    kw.setdefault("fallbacks", ["gemini/gemini-3.8-flash"])
    return touchpoint("t.retry", backend="tests.test_retry:backend", **kw)


# ------------------------------------------------------------------ reading the error


def test_gemini_per_minute_limit_is_a_rate_limit_with_its_stated_delay():
    exc = RateLimitError(GEMINI_MINUTE)
    assert classify_error(exc) == "rate_limit" and is_retryable(exc)
    assert retry_after(exc) == 4.0


def test_daily_quota_is_not_waited_out():
    exc = RateLimitError(GEMINI_DAY)
    assert classify_error(exc) == "quota_exhausted"
    assert retry_after(exc) is None


def test_stated_delay_sources():
    class Response:
        headers = {"retry-after": "3"}

    with_header = RateLimitError("slow down")
    with_header.response = Response()
    assert stated_delay(with_header) == 3.0

    class MsResponse:
        headers = {"retry-after-ms": "1500"}

    ms = RateLimitError("slow down")
    ms.response = MsResponse()
    assert stated_delay(ms) == 1.5

    attr = RateLimitError("slow down")
    attr.retry_after = 2
    assert stated_delay(attr) == 2.0
    assert stated_delay(RateLimitError("Rate limit reached. Please try again in 250ms.")) == 0.25
    assert stated_delay(RateLimitError("Rate limit reached")) is None


def test_unstated_delays_use_defaults_and_others_are_not_waited():
    assert retry_after(RateLimitError("rate limit")) == 20.0
    assert retry_after(RuntimeError("503 overloaded")) == 5.0
    assert retry_after(ConnectionError("provider down")) is None
    assert retry_after(RuntimeError("401 invalid api key")) is None


def test_policy_limits_attempts_and_wait():
    policy = RetryPolicy(attempts=2, max_wait=5)
    exc = RateLimitError(GEMINI_MINUTE)
    assert policy.delay(exc, 0) == 4.0 and policy.delay(exc, 1) == 4.0 and policy.delay(exc, 2) is None
    assert RetryPolicy(attempts=2, max_wait=3).delay(exc, 0) is None  # longer than this caller waits
    assert RetryPolicy.NEVER.delay(exc, 0) is None
    assert RetryPolicy.coerce({"attempts": 3, "max_wait": 10}) == RetryPolicy(3, 10.0)
    assert RetryPolicy.coerce("batch") is RetryPolicy.BATCH
    with pytest.raises(TypeError):
        RetryPolicy.coerce(5)


# ------------------------------------------------------------------ calling


def test_short_rate_limit_is_waited_out_on_the_same_model():
    FAILURES["anthropic/claude-sonnet-5-5"] = [RateLimitError(GEMINI_MINUTE)]
    reply = tp().model(env=FakeEnv()).invoke("hi")
    assert reply.content == "anthropic/claude-sonnet-5-5: hi"
    assert CALLS == ["anthropic/claude-sonnet-5-5"] * 2 and SLEPT == [4.0]


def test_too_long_a_wait_falls_back_at_once():
    FAILURES["anthropic/claude-sonnet-5-5"] = [RateLimitError("rate limit, retry in 40s")]
    reply = tp().model(env=FakeEnv()).invoke("hi")
    assert reply.content == "gemini/gemini-3.8-flash: hi" and SLEPT == []


def test_a_batch_policy_waits_longer_and_more_often():
    FAILURES["anthropic/claude-sonnet-5-5"] = [RateLimitError("rate limit, retry in 40s")] * 2
    reply = tp(retry=RetryPolicy.BATCH).model(env=FakeEnv()).invoke("hi")
    assert reply.content == "anthropic/claude-sonnet-5-5: hi" and SLEPT == [40.0, 40.0]


def test_single_candidate_waits_too():
    FAILURES["anthropic/claude-sonnet-5-5"] = [RateLimitError(GEMINI_MINUTE)]
    llm = tp(fallbacks=[]).model(env=FakeEnv())
    assert isinstance(llm, TouchpointChatModel)
    assert llm.invoke("hi").content == "anthropic/claude-sonnet-5-5: hi" and SLEPT == [4.0]


def test_single_candidate_raises_its_own_error():
    FAILURES["anthropic/claude-sonnet-5-5"] = [ConnectionError("provider down")]
    with pytest.raises(ConnectionError):
        tp(fallbacks=[]).model(env=FakeEnv()).invoke("hi")


def test_every_model_failing_lists_each_error():
    FAILURES["anthropic/claude-sonnet-5-5"] = [RateLimitError(GEMINI_DAY)]
    FAILURES["gemini/gemini-3.8-flash"] = [ConnectionError("provider down")]
    with pytest.raises(AllModelsFailed) as info:
        tp().model(env=FakeEnv()).invoke("hi")
    failed = info.value
    assert [m for m, _ in failed.errors] == ["anthropic/claude-sonnet-5-5", "gemini/gemini-3.8-flash"]
    assert isinstance(failed.last, ConnectionError) and failed.__cause__ is failed.last
    assert "anthropic/claude-sonnet-5-5: RateLimitError" in str(failed) and "provider down" in str(failed)
    assert classify_error(failed) == "quota_exhausted"  # the primary model's reason
    assert SLEPT == []


def test_async_and_streaming_wait_then_fall_back():
    llm = tp().model(env=FakeEnv())
    FAILURES["anthropic/claude-sonnet-5-5"] = [RateLimitError(GEMINI_MINUTE)]
    assert asyncio.run(llm.ainvoke("hi")).content == "anthropic/claude-sonnet-5-5: hi"
    FAILURES["anthropic/claude-sonnet-5-5"] = [RateLimitError(GEMINI_MINUTE), RateLimitError(GEMINI_MINUTE)]
    assert "".join(c.content for c in llm.stream("hi")) == "gemini/gemini-3.8-flash: hi"

    async def collect():
        return "".join([c.content async for c in llm.astream("hi")])

    FAILURES["anthropic/claude-sonnet-5-5"] = [RateLimitError(GEMINI_MINUTE)]
    assert asyncio.run(collect()) == "anthropic/claude-sonnet-5-5: hi"
    assert SLEPT == [4.0, 4.0, 4.0]


def test_async_every_model_failing():
    FAILURES["anthropic/claude-sonnet-5-5"] = [ConnectionError("a down")]
    FAILURES["gemini/gemini-3.8-flash"] = [ConnectionError("b down")]
    with pytest.raises(AllModelsFailed):
        asyncio.run(tp().model(env=FakeEnv()).ainvoke("hi"))


def test_bound_tools_keep_the_policy():
    from langchain_core.tools import tool

    @tool
    def double(x: int) -> int:
        """Double x."""
        return 2 * x

    FAILURES["anthropic/claude-sonnet-5-5"] = [RateLimitError(GEMINI_MINUTE)]
    llm = tp().model(env=FakeEnv()).bind_tools([double])
    assert llm.retry == RetryPolicy.INTERACTIVE  # LLM_KIT["RETRY"]'s default
    assert llm.invoke("hi").content == "anthropic/claude-sonnet-5-5: hi" and SLEPT == [4.0]


def test_policy_precedence(monkeypatch):
    t = tp(retry=RetryPolicy(attempts=2, max_wait=10))
    assert t.model(env=FakeEnv()).retry == RetryPolicy(2, 10.0)
    assert t.model(env=FakeEnv(), retry="never").retry == RetryPolicy.NEVER
    run = {"configurable": {RETRY_KEY: RetryPolicy.BATCH}}
    assert t.model(env=FakeEnv(), config=run).retry == RetryPolicy.BATCH
    assert t.model(env=FakeEnv(), config=run, retry="interactive").retry == RetryPolicy.INTERACTIVE
    monkeypatch.setattr(settings, "_source", lambda: {"RETRY": {"attempts": 3, "max_wait": 1}})
    assert tp().model(env=FakeEnv()).retry == RetryPolicy(3, 1.0)


def test_a_text_only_overload_is_waited_out_briefly():
    exc = RuntimeError(
        "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is experiencing high demand.'}}"
    )
    assert retry_after(exc) == 5.0


def test_fallbacks_can_be_turned_off_for_one_call_and_it_still_waits():
    FAILURES["anthropic/claude-sonnet-5-5"] = [RateLimitError(GEMINI_MINUTE), ConnectionError("down")]
    llm = tp().model(env=FakeEnv(), fallbacks=False)
    assert isinstance(llm, TouchpointChatModel) and len(llm.models) == 1
    with pytest.raises(ConnectionError):  # waited once, then its own error: no fallback tried
        llm.invoke("hi")
    assert CALLS == ["anthropic/claude-sonnet-5-5"] * 2 and SLEPT == [4.0]
