"""Platform admins managing any tenant or user (scope_id), and budgets for one member of a tenant."""

from decimal import Decimal
from uuid import uuid4

import pytest
from django.test import Client
from django.utils import timezone

from llm_kit.core.errors import BudgetExceeded
from llm_kit.core.touchpoints import touchpoint
from llm_kit.core.touchpoints.touchpoint import _unregister
from llm_kit.models import ProviderKey, ScopedBudget, ScopedSetting
from llm_kit.scoped import budgets, service
from llm_kit.tools.caller import caller_for
from tests.graphs import SCRIPTS
from tests.test_scoped import GEMINI, tp  # noqa: F401  (fixture)

pytestmark = pytest.mark.django_db

H = "ollama/script-helper"


@pytest.fixture
def root(django_user_model):
    return django_user_model.objects.create_superuser("root", "root@example.com", "pw")


@pytest.fixture(autouse=True)
def setup(settings, monkeypatch):
    from llm_kit.core.observability import spend
    from llm_kit.observability.budgets import period_start, seed

    monkeypatch.setattr(spend, "REFRESH", 0.0)
    spend.spend.configure(seed, period_start)
    settings.LLM_KIT = {
        **settings.LLM_KIT,
        "SCOPE_PERMISSIONS": {"tenant": "tests.testapp.tenancy:can_manage"},
        "BYOK_SCOPES": ("tenant", "user"),
        "BUDGET_EDITORS": {"tenant": ["platform"], "member": ["tenant"], "user": ["user"]},
        "TENANT_MEMBERS": "tests.testapp.tenancy:members_of",
    }


def client(user):
    c = Client()
    c.force_login(user)
    return c


def test_a_platform_admin_manages_another_tenant_through_the_api(tp, world, root):  # noqa: F811
    globex = str(world["globex"].pk)
    c = client(root)
    r = c.patch(
        "/llm/api/settings/",
        {"scope": "tenant", "scope_id": globex, "touchpoint": "t.settable", "key": "model", "value": GEMINI},
        content_type="application/json",
    )
    assert r.status_code == 200
    assert ScopedSetting.objects.get().scope_id == globex
    assert tp.resolve(config={"configurable": {"__llm_kit_caller": caller_for(world["bob"])}}).model == GEMINI

    body = c.get("/llm/api/settings/", {"scope": "tenant", "scope_id": globex}).json()
    assert body["editable"] and body["sections"][0]["settings"][0]["value"] == GEMINI

    assert (
        c.post(
            "/llm/api/keys/",
            {"scope": "tenant", "scope_id": globex, "provider": "openai", "key": "sk-globex-1234"},
            content_type="application/json",
        ).status_code
        == 201
    )
    assert ProviderKey.objects.get().scope_id == globex
    r = c.put(
        "/llm/api/budgets/",
        {"scope": "tenant", "scope_id": globex, "period": "month", "amount": 50},
        content_type="application/json",
    )
    assert r.status_code == 200 and ScopedBudget.objects.get().scope_id == globex


def test_others_cant_act_on_another_scope(tp, world):  # noqa: F811
    alice = client(world["alice"])  # manages acme, not the platform
    r = alice.get("/llm/api/settings/", {"scope": "tenant", "scope_id": str(world["globex"].pk)})
    assert r.status_code == 403
    r = alice.put(
        "/llm/api/budgets/",
        {"scope": "user", "scope_id": str(world["bob"].pk), "period": "day", "amount": 1},
        content_type="application/json",
    )
    assert r.status_code == 403 and not ScopedBudget.objects.exists()


def spent(amount, **who):
    from llm_kit.models import LLMCall

    LLMCall.objects.create(call_id=uuid4(), started_at=timezone.now(), cost=Decimal(amount), **who)


def test_a_member_budget_for_one_member_replaces_the_all_members_one(world):
    alice, carol = caller_for(world["alice"]), caller_for(world["carol"])
    acme = str(world["acme"].pk)
    budgets.set_budget(alice, "member", "day", 1)  # everyone in acme: $1
    budgets.set_budget(alice, "member", "day", 5, member=world["carol"].pk)  # carol: $5 instead
    spent("2", tenant=acme, user=str(world["carol"].pk))
    spent("2", tenant=acme, user=str(world["alice"].pk))
    touchpoint("t.metered", default=H)
    try:
        from llm_kit.core.touchpoints import get_touchpoint

        t = get_touchpoint("t.metered")
        SCRIPTS[H] = ["ok"]
        assert t.complete("hi", caller=carol).content == "ok"  # $2 of her $5
        with pytest.raises(BudgetExceeded):
            t.complete("hi", caller=alice)  # $2 of the $1 everyone else has
    finally:
        _unregister("t.metered")
        SCRIPTS.clear()

    listed = budgets.list_for(alice, "member", world["carol"].pk)
    assert [(b["member"], b["amount"], b["spent"]) for b in listed] == [(str(world["carol"].pk), "5", "2.0000")]
    assert [g["member"] for g in budgets.member_overrides(alice)] == [str(world["carol"].pk)]


def test_only_members_can_be_named(world, settings):
    alice = caller_for(world["alice"])
    with pytest.raises(ValueError, match="isn't a member"):
        budgets.set_budget(alice, "member", "day", 1, member=world["bob"].pk)  # bob is in globex
    with pytest.raises(service.NotPermitted):
        budgets.set_budget(caller_for(world["carol"]), "member", "day", 1, member=world["alice"].pk)  # not a manager
    settings.LLM_KIT = {**settings.LLM_KIT, "TENANT_MEMBERS": None}
    with pytest.raises(service.NotPermitted, match="TENANT_MEMBERS"):
        budgets.set_budget(alice, "member", "day", 1, member=world["carol"].pk)


def test_the_budgets_api_and_template_for_members(world):
    alice = client(world["alice"])
    carol_id = world["carol"].pk
    r = alice.put(
        "/llm/api/budgets/",
        {"scope": "member", "period": "day", "amount": 3, "member": carol_id},
        content_type="application/json",
    )
    assert r.status_code == 200 and r.json()["member"] == str(carol_id)
    body = alice.get("/llm/api/budgets/", {"scope": "member"}).json()
    assert sorted(body["members"]) == sorted([str(world["alice"].pk), str(carol_id)])
    assert body["member_budgets"][0]["member"] == str(carol_id)

    from django.template import Context, Template
    from django.test import RequestFactory

    request = RequestFactory().get("/team/")
    request.user, request.session = world["alice"], {}
    html = Template('{% load llm_kit %}{% llm_kit_budgets scope="member" %}').render(Context({"request": request}))
    assert "Members with their own budget" in html and 'name="member"' in html and "Every member" in html

    alice.post("/llm/settings/budgets/save/", {"scope": "member", "period": "day", "member": carol_id, "remove": "1"})
    assert not ScopedBudget.objects.exists()


def test_the_template_tags_take_a_scope_id_for_admin_pages(tp, world, root):  # noqa: F811
    from django.template import Context, Template
    from django.test import RequestFactory

    request = RequestFactory().get("/staff/globex/")
    request.user, request.session = root, {}
    source = '{% load llm_kit %}{% llm_kit_settings scope="tenant" scope_id=tid %}'
    html = Template(source).render(Context({"request": request, "tid": world["globex"].pk}))
    assert f'name="scope_id" value="{world["globex"].pk}"' in html and ">Save</button>" in html
    request.user = world["alice"]
    html = Template(source).render(Context({"request": request, "tid": world["globex"].pk}))
    assert "no tenant settings" in html
