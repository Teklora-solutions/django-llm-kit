"""Budgets per tenant, per member of a tenant and per user: who sets them, enforcement, the API."""

from decimal import Decimal
from uuid import uuid4

import pytest
from django.test import Client
from django.utils import timezone

from llm_kit import agents
from llm_kit.core.errors import BudgetExceeded
from llm_kit.core.touchpoints import touchpoint
from llm_kit.core.touchpoints.touchpoint import _unregister
from llm_kit.scoped import budgets
from llm_kit.scoped.service import NotPermitted
from llm_kit.tools.caller import CONFIG_KEY, caller_for
from tests.graphs import SCRIPTS

pytestmark = pytest.mark.django_db(transaction=True)

H = "ollama/script-helper"


@pytest.fixture(autouse=True)
def setup(settings, monkeypatch, world):
    from llm_kit.core.observability import spend
    from llm_kit.observability.budgets import period_start, seed

    monkeypatch.setattr(spend, "REFRESH", 0.0)  # read recorded spend on every check
    spend.spend.configure(seed, period_start)
    settings.LLM_KIT = {
        **settings.LLM_KIT,
        "SCOPE_PERMISSIONS": {
            "tenant": "tests.testapp.tenancy:can_manage",
            "platform": "tests.testapp.tenancy:can_manage",
        },
        "BUDGET_EDITORS": {"tenant": ["platform"], "member": ["tenant"], "user": ["user"]},
        "BUDGET_PERIODS": ["day", "month"],
    }
    SCRIPTS.clear()
    touchpoint("t.metered", default=H)
    yield
    _unregister("t.metered")
    SCRIPTS.clear()


def spent(amount, **who):
    from llm_kit.models import LLMCall

    LLMCall.objects.create(call_id=uuid4(), started_at=timezone.now(), cost=Decimal(amount), **who)


def call_as(user):
    from llm_kit.core.touchpoints import get_touchpoint

    SCRIPTS[H] = ["ok"]
    return get_touchpoint("t.metered").complete("hi", config={"configurable": {CONFIG_KEY: caller_for(user)}})


def test_a_tenants_budget_limits_every_member(world):
    alice, carol, bob = world["alice"], world["carol"], world["bob"]
    acme = str(world["acme"].pk)
    budgets.set_budget(caller_for(alice), "tenant", "day", 1)  # alice manages the platform in these tests
    assert call_as(carol).content == "ok"
    spent("1", tenant=acme)
    with pytest.raises(BudgetExceeded, match=r"workspace budget of \$1 is used up \(\$1.00 spent today\)"):
        call_as(carol)
    assert call_as(bob).content == "ok"  # globex has no budget


def test_a_per_member_budget_limits_each_member_alone(world):
    alice, carol = world["alice"], world["carol"]
    acme = str(world["acme"].pk)
    budgets.set_budget(caller_for(alice), "member", "month", "0.5")
    spent("0.5", tenant=acme, user=str(carol.pk))
    with pytest.raises(BudgetExceeded, match="per-member"):
        call_as(carol)
    assert call_as(alice).content == "ok"


def test_calls_on_their_own_keys_can_be_left_out(world):
    acme = str(world["acme"].pk)
    budgets.set_budget(caller_for(world["alice"]), "tenant", "day", 1, counts="platform_keys")
    spent("5", tenant=acme, key_owner="tenant")  # paid with acme's own key
    assert call_as(world["carol"]).content == "ok"
    spent("1", tenant=acme, key_owner="platform")
    with pytest.raises(BudgetExceeded):
        call_as(world["carol"])


def test_a_budget_for_one_touchpoint(world):
    touchpoint("t.other", default=H)
    try:
        budgets.set_budget(caller_for(world["carol"]), "user", "day", 1, touchpoint="t.other")
        spent("2", user=str(world["carol"].pk), touchpoint="t.other")
        assert call_as(world["carol"]).content == "ok"  # t.metered isn't limited
        from llm_kit.core.touchpoints import get_touchpoint

        with pytest.raises(BudgetExceeded, match="'t.other'"):
            get_touchpoint("t.other").model(config={"configurable": {CONFIG_KEY: caller_for(world["carol"])}})
    finally:
        _unregister("t.other")


def test_who_may_set_which_budget(world, settings):
    carol = caller_for(world["carol"])
    with pytest.raises(NotPermitted):
        budgets.set_budget(carol, "tenant", "day", 1)  # only the platform
    with pytest.raises(NotPermitted):
        budgets.set_budget(carol, "member", "day", 1)  # only acme's managers
    budgets.set_budget(carol, "user", "day", 1)  # her own
    with pytest.raises(ValueError, match="period"):
        budgets.set_budget(carol, "user", "week", 1)  # not in BUDGET_PERIODS
    with pytest.raises(ValueError, match="negative"):
        budgets.set_budget(carol, "user", "day", -1)
    settings.LLM_KIT = {**settings.LLM_KIT, "BUDGET_EDITORS": {}}
    with pytest.raises(NotPermitted):
        budgets.set_budget(carol, "user", "day", 2)  # nobody, until the developer says who


def test_an_agent_over_its_tenants_budget_stops(world):
    budgets.set_budget(caller_for(world["alice"]), "tenant", "day", 1)
    spent("1", tenant=str(world["acme"].pk))
    SCRIPTS["ollama/script-assistant"] = ["never"]
    result = agents.run("tests.assistant", "x", user=world["carol"])
    assert result.stopped == "budget" and "workspace budget" in result.text


def test_the_budgets_api(world):
    carol = Client()
    carol.force_login(world["carol"])
    URL = "/llm/api/budgets/"
    r = carol.put(URL, {"scope": "user", "period": "day", "amount": "2.5"}, content_type="application/json")
    assert r.status_code == 200 and r.json()["amount"] == "2.5"
    spent("1", user=str(world["carol"].pk))
    body = carol.get(URL, {"scope": "user"}).json()
    assert body["periods"] == ["day", "month"] and body["editable"] is True
    [mine] = body["budgets"]
    assert (mine["spent"], mine["remaining"], mine["period"]) == ("1.0000", "1.5000", "day")
    assert mine["resets_at"] > timezone.localtime().isoformat()

    tenant = carol.get(URL, {"scope": "tenant"}).json()
    assert tenant["editable"] is False and tenant["budgets"] == []
    assert (
        carol.put(URL, {"scope": "tenant", "period": "day", "amount": 9}, content_type="application/json").status_code
        == 403
    )
    assert (
        carol.put(URL, {"scope": "user", "period": "decade", "amount": 9}, content_type="application/json").status_code
        == 400
    )
    assert carol.delete(URL, {"scope": "user", "period": "day"}, content_type="application/json").status_code == 204
    assert carol.get(URL, {"scope": "user"}).json()["budgets"] == []
