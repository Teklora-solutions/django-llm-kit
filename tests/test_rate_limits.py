"""Per-model rate limits: a slot per call, shared counts, waiting, then the next fallback."""

import os

import pytest
from rest_framework.test import APIClient

from llm_kit import ratelimits
from llm_kit.core.errors import AllModelsFailed
from llm_kit.core.observability.classify import classify_error
from llm_kit.core.touchpoints import get_touchpoint
from llm_kit.core.touchpoints.ratelimit import Limiter, Limits, MemoryStore, RateLimited
from llm_kit.core.touchpoints.retry import RetryPolicy
from llm_kit.models import RateLimit
from llm_kit.ratelimits.stores import CacheStore
from tests.graphs import SCRIPTS

SONNET, FLASH = "anthropic/claude-sonnet-5-5", "gemini/gemini-3.8-flash"


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


# ------------------------------------------------------------------ the limiter (no Django)


def test_requests_per_minute_slide_and_a_refusal_takes_no_slot():
    clock, store = Clock(), MemoryStore()
    limiter = Limiter(store, lambda model: Limits(rpm=2), clock=clock)
    limiter.acquire("m")
    clock.now += 10
    limiter.acquire("m")
    with pytest.raises(RateLimited) as e:
        limiter.acquire("m")
    assert e.value.reason == "rpm" and round(e.value.retry_in) == 50 and classify_error(e.value) == "rate_limit"
    assert store.usage("m", clock.now)["rpm"] == 2  # the refused one wasn't counted
    clock.now += 51  # the first one has left the window
    limiter.acquire("m")


def test_tokens_per_minute_bind_from_the_next_call():
    clock = Clock()
    limiter = Limiter(MemoryStore(), lambda model: Limits(tpm=100), clock=clock)
    limiter.acquire("m")
    limiter.record("m", 150)  # one call may overshoot
    with pytest.raises(RateLimited, match="tokens per minute"):
        limiter.acquire("m")
    clock.now += 60
    limiter.acquire("m")


def test_the_daily_limit_is_a_quota():
    limiter = Limiter(MemoryStore(), lambda model: Limits(rpd=1), clock=Clock())
    limiter.acquire("m")
    with pytest.raises(RateLimited) as e:
        limiter.acquire("m", max_wait=30)  # a day away: not worth waiting for
    assert classify_error(e.value) == "quota_exhausted"


def test_waits_for_a_slot_when_one_frees_up_in_time(monkeypatch):
    import llm_kit.core.touchpoints.ratelimit as module

    clock = Clock()
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(module.time, "sleep", sleep)
    limiter = Limiter(MemoryStore(), lambda model: Limits(rpm=1), clock=clock)
    limiter.acquire("m")
    clock.now += 58
    limiter.acquire("m", max_wait=5)  # a slot in 2s
    assert slept and round(sum(slept)) == 2


def test_everything_is_counted_even_without_limits():
    clock, store = Clock(), MemoryStore()
    limiter = Limiter(store, lambda model: None, clock=clock)
    limiter.acquire("m")
    limiter.record("m", 40)
    assert limiter.usage("m") == {
        "model": "m",
        "rpm_used": 1,
        "tpm_used": 40,
        "rpd_used": 1,
        "rpm_limit": 0,
        "tpm_limit": 0,
        "rpd_limit": 0,
    }


def test_limits_take_only_known_keys():
    with pytest.raises(Exception, match="rpm, tpm and rpd"):
        Limits.coerce({"rps": 1})


# ------------------------------------------------------------------ shared stores


@pytest.fixture
def cache_store():
    from django.core.cache import cache

    cache.clear()
    return CacheStore(cache)


def test_the_cache_store_counts_and_refuses_without_taking_slots(cache_store):
    now = 1_000_020.0
    assert cache_store.take("m", Limits(rpm=1, rpd=5), now)[0]
    granted, reason, wait = cache_store.take("m", Limits(rpm=1, rpd=5), now)
    assert not granted and reason == "rpm" and 0 < wait <= 60
    assert cache_store.usage("m", now) == {"rpm": 1, "tpm": 0, "rpd": 1}
    cache_store.add_tokens("m", 30, now)
    assert not cache_store.take("m", Limits(tpm=30), now + 60 * 0)[0]
    assert cache_store.take("m", Limits(rpd=2), now + 60)[0]  # a new minute
    assert not cache_store.take("m", Limits(rpd=2), now + 120)[0]  # but the day is used up


@pytest.mark.skipif(not os.environ.get("LLM_KIT_TEST_REDIS"), reason="set LLM_KIT_TEST_REDIS=redis://... to run")
def test_the_redis_store_slides_and_is_atomic():
    import redis

    from llm_kit.ratelimits.stores import RedisStore

    client = redis.Redis.from_url(os.environ["LLM_KIT_TEST_REDIS"])
    client.flushdb()
    store, now = RedisStore(client), 2_000_010.0
    assert store.take("m", Limits(rpm=2), now)[0] and store.take("m", Limits(rpm=2), now + 30)[0]
    granted, reason, wait = store.take("m", Limits(rpm=2), now + 40)
    assert not granted and reason == "rpm" and round(wait) == 20
    assert store.take("m", Limits(rpm=2), now + 61)[0]  # the first left the window
    store.add_tokens("m", 500, now + 61)
    assert store.take("m", Limits(tpm=500), now + 61)[1] == "tpm"
    assert store.usage("m", now + 61) == {"rpm": 2, "tpm": 500, "rpd": 3}


# ------------------------------------------------------------------ settings, rows, calls


@pytest.fixture
def limits(settings, keys):
    from django.core.cache import cache

    cache.clear()

    def set_(value):
        settings.LLM_KIT = {**settings.LLM_KIT, "RATE_LIMITS": value, "RETRY": {"attempts": 0, "max_wait": 0}}

    yield set_
    cache.clear()


@pytest.mark.django_db
def test_no_limits_means_no_limiter(settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RATE_LIMITS": {}}
    assert ratelimits.limiter() is None


@pytest.mark.django_db
def test_rows_win_over_settings_and_the_most_specific_pattern_applies(limits):
    limits({"anthropic/*": {"rpm": 50}, "gemini/*": {"rpm": 5}})
    assert ratelimits.limits_for(SONNET) == Limits(rpm=50)
    RateLimit.objects.create(model="anthropic/*", rpm=7, note="tier 1")
    RateLimit.objects.create(model=SONNET, tpm=1000)
    assert ratelimits.limits_for(SONNET) == Limits(tpm=1000)
    assert ratelimits.limits_for("anthropic/claude-haiku-5-5") == Limits(rpm=7)
    assert ratelimits.limits_for("openai/gpt-6") is None


@pytest.mark.django_db
def test_a_model_at_its_limit_moves_the_call_to_the_fallback(limits):
    limits({SONNET: {"rpm": 1}})
    chat = get_touchpoint("tests.chat")  # sonnet, then gemini flash
    assert chat.complete("one").content == "echo: one"
    assert chat.complete("two").content == "echo: two"  # sonnet is full: flash answers
    used = {u["model"]: u for u in ratelimits.usage()}
    assert used[SONNET]["rpm_used"] == 1


@pytest.mark.django_db
def test_when_no_model_has_a_slot_the_call_fails_classified(limits):
    limits({SONNET: {"rpm": 1}, FLASH: {"rpm": 1}})
    chat = get_touchpoint("tests.chat")
    chat.complete("one")
    chat.complete("two")
    with pytest.raises(AllModelsFailed) as e:
        chat.complete("three")
    assert classify_error(e.value) == "rate_limit" and all(isinstance(err, RateLimited) for _, err in e.value.errors)


@pytest.mark.django_db
def test_a_single_model_touchpoint_raises_rate_limited_and_counts_tokens(limits):
    limits({"ollama/script-helper": {"tpm": 15}})  # each scripted reply reports 15 tokens
    helper = get_touchpoint("tests.helper")
    SCRIPTS["ollama/script-helper"] = ["a", "b"]
    try:
        helper.complete("hi")
        with pytest.raises(RateLimited, match="tokens per minute"):
            helper.complete("again")
    finally:
        SCRIPTS.clear()


@pytest.mark.django_db
def test_streams_take_a_slot_too(limits):
    limits({SONNET: {"rpm": 1}, FLASH: {"rpm": 0}})
    model = get_touchpoint("tests.chat").model(retry=RetryPolicy.NEVER)
    assert "".join(c.content for c in model.stream("hi")) == "echo: hi"
    assert "".join(c.content for c in model.stream("hi")) == "echo: hi"  # flash
    used = {u["model"]: u for u in ratelimits.usage()}
    assert used[SONNET]["rpm_used"] == 1 and used[FLASH]["rpm_used"] == 1


@pytest.mark.django_db
async def test_async_calls_take_a_slot(limits):
    limits({SONNET: {"rpm": 1}, FLASH: {"rpm": 1}})
    from asgiref.sync import sync_to_async

    model = await sync_to_async(get_touchpoint("tests.chat").model)(retry=RetryPolicy.NEVER)
    await model.ainvoke("one")
    await model.ainvoke("two")
    with pytest.raises(AllModelsFailed):
        await model.ainvoke("three")


def test_calls_on_keys_a_caller_brought_are_not_limited():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from llm_kit.core.touchpoints.backends import TouchpointChatModel

    limiter = Limiter(MemoryStore(), lambda model: Limits(rpm=1), clock=Clock())
    model = TouchpointChatModel(
        runnable=FakeListChatModel(responses=["a", "b"]),
        fallbacks=[],
        refs=["m"],
        owners=["tenant"],
        limiter=limiter,
        retry=RetryPolicy.NEVER,
    )
    assert model.invoke("x").content == "a" and model.invoke("y").content == "b"
    assert limiter.usage("m")["rpm_used"] == 0


@pytest.mark.django_db
def test_rate_limits_api(limits, django_user_model):
    limits({SONNET: {"rpm": 3}})
    get_touchpoint("tests.chat").complete("hi")
    api = APIClient()
    api.force_authenticate(django_user_model.objects.create_user("staff", is_staff=True))
    rows = {r["model"]: r for r in api.get("/llm/api/monitoring/rate-limits/").data["models"]}
    assert rows[SONNET]["rpm_used"] == 1 and rows[SONNET]["rpm_limit"] == 3


@pytest.mark.skipif(not os.environ.get("LLM_KIT_TEST_REDIS"), reason="set LLM_KIT_TEST_REDIS=redis://... to run")
@pytest.mark.django_db
def test_auto_uses_redis_when_the_default_cache_is_redis(limits, settings):
    from llm_kit.ratelimits.stores import RedisStore

    settings.CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": os.environ["LLM_KIT_TEST_REDIS"],
        }
    }
    limits({SONNET: {"rpm": 1}})
    assert isinstance(ratelimits.limiter().store, RedisStore)
    chat = get_touchpoint("tests.chat")
    chat.complete("one")
    chat.complete("two")  # sonnet full: flash
    assert {u["model"]: u["rpm_used"] for u in ratelimits.usage()}[SONNET] == 1
