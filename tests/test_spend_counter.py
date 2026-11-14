"""Budgets across processes: a shared counter in Django's cache carries every worker's spend."""

from decimal import Decimal

import pytest
from django.core.cache import cache

from llm_kit.core.observability import spend
from llm_kit.core.observability.records import CallRecord
from llm_kit.core.observability.spend import SpendKey, SpendTracker
from llm_kit.models import ScopedBudget
from llm_kit.observability import spend_counter

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def clean():
    cache.clear()
    yield
    cache.clear()


def call(**kw):
    kw.setdefault("status", "ok")
    return CallRecord(call_id="c", run_id=None, started_at=None, **kw)


def test_another_process_spend_shows_up_at_once(monkeypatch):
    from llm_kit.observability.budgets import period_start

    monkeypatch.setattr(spend, "REFRESH", 3600.0)  # no database re-reads during the test
    ScopedBudget.objects.create(scope="tenant", scope_id="3", period="day", amount=10)
    counter = spend_counter.CacheCounter()
    this, other = SpendTracker(), SpendTracker()  # two workers, one cache
    for tracker in (this, other):
        tracker.configure(lambda key, start: Decimal("1"), period_start, counter)
    key = SpendKey("tenant", "3", period="day")
    assert this.spent(key) == Decimal("1")  # read: $1 recorded

    priced_elsewhere = spend_counter.SharedObserver(counter, period_start)
    priced_elsewhere(call(tenant="3", user="7", cost=Decimal("2.5")))  # in the other worker
    assert this.spent(key) == Decimal("3.5")  # seen without a database read
    priced_elsewhere(call(tenant="4", cost=Decimal("9")))  # another tenant's
    priced_elsewhere(call(tenant="3", cost=Decimal("9"), status="error"))  # failed calls cost nothing
    assert this.spent(key) == Decimal("3.5")


def test_without_a_counter_only_this_process_counts(monkeypatch):
    from llm_kit.observability.budgets import period_start

    monkeypatch.setattr(spend, "REFRESH", 3600.0)
    tracker = SpendTracker()
    tracker.configure(lambda key, start: Decimal("1"), period_start)
    key = SpendKey("tenant", "3")
    tracker.spent(key)
    tracker.observe(call(tenant="3", cost=Decimal("0.5")))
    assert tracker.spent(key) == Decimal("1.5")


def test_a_call_counts_towards_every_budget_it_falls_under():
    from llm_kit.core.touchpoints import touchpoint
    from llm_kit.core.touchpoints.touchpoint import _unregister

    ScopedBudget.objects.create(scope="tenant", scope_id="3", period="month", amount=10)
    ScopedBudget.objects.create(scope="member", scope_id="3", period="day", amount=1)
    ScopedBudget.objects.create(scope="member", scope_id="3:7", period="day", amount=5, touchpoint="t.k")
    ScopedBudget.objects.create(scope="user", scope_id="7", period="day", amount=2, counts="platform_keys")
    ScopedBudget.objects.create(scope="user", scope_id="8", period="day", amount=2)  # someone else
    touchpoint("t.k", default="ollama/x", daily_budget=4)
    try:
        keys = spend_counter.keys_for(call(tenant="3", user="7", touchpoint="t.k", key_owner="tenant"))
    finally:
        _unregister("t.k")
    assert set(keys) == {
        SpendKey(touchpoint="t.k"),
        SpendKey("tenant", "3", period="month"),
        SpendKey("member", "3:7", period="day"),
        SpendKey("member", "3:7", "t.k", "day"),
    }  # not the user budget: it counts platform keys only, and this call used the tenant's


def test_the_counter_is_on_only_for_a_shared_cache(settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "SPEND_COUNTER": "auto"}
    settings.CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}
    assert not spend_counter.enabled()
    settings.CACHES = {"default": {"BACKEND": "django.core.cache.backends.redis.RedisCache", "LOCATION": "x"}}
    assert spend_counter.enabled()
    settings.LLM_KIT = {**settings.LLM_KIT, "SPEND_COUNTER": None}
    assert not spend_counter.enabled()
    settings.LLM_KIT = {**settings.LLM_KIT, "SPEND_COUNTER": "cache"}
    settings.CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}
    assert spend_counter.enabled()


def test_counter_adds_up_in_micro_dollars():
    from datetime import datetime

    counter = spend_counter.CacheCounter()
    key, start = SpendKey("user", "7"), datetime(2026, 11, 14)
    counter.add(key, start, Decimal("0.0000004"))  # rounds up to a micro-dollar
    counter.add(key, start, Decimal("1.25"))
    assert counter.get(key, start) == Decimal("1.250001")
    assert counter.get(key, datetime(2026, 11, 15)) == 0  # a new period
