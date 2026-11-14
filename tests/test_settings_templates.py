"""The settings template tags render forms; their posts save through the same service as the API."""

import pytest
from django.contrib.messages import get_messages
from django.template import Context, Template
from django.test import Client, RequestFactory

from llm_kit.models import ProviderKey, ScopedBudget, ScopedSetting
from tests.test_scoped import CLAUDE, GEMINI, tp  # noqa: F401  (fixture)

pytestmark = pytest.mark.django_db


@pytest.fixture
def configured(settings, tp):  # noqa: F811
    settings.LLM_KIT = {
        **settings.LLM_KIT,
        "BYOK_SCOPES": ("tenant",),
        "BUDGET_EDITORS": {"member": ["tenant"]},
        "SCOPE_PERMISSIONS": {"tenant": "tests.testapp.tenancy:can_manage"},
    }
    return tp


def render(user, source):
    from django.middleware.csrf import get_token

    request = RequestFactory().get("/account/ai/")
    request.user = user
    request.session = {}
    return Template("{% load llm_kit %}" + source).render(
        Context({"request": request, "csrf_token": get_token(request)})
    )


def test_the_settings_tag_renders_a_form_per_setting(configured, world):
    html = render(world["alice"], '{% llm_kit_settings scope="tenant" %}')
    assert 'data-touchpoint="t.settable"' in html and "Answers support questions." in html
    assert f'<option value="{GEMINI}"' in html and 'name="key" value="model"' in html
    assert "param.temperature" not in html  # user-only
    assert ">Save</button>" in html and "/llm/settings/save/" in html

    carol = render(world["carol"], '{% llm_kit_settings scope="tenant" %}')
    assert "disabled" in carol and ">Save</button>" not in carol  # she can see, not change
    assert "param.temperature" in render(world["carol"], "{% llm_kit_settings %}")  # her own scope


def test_saving_and_clearing_through_the_form(configured, world):
    c = Client()
    c.force_login(world["alice"])
    form = {"scope": "tenant", "touchpoint": "t.settable", "key": "model", "next": "/account/ai/"}
    r = c.post("/llm/settings/save/", {**form, "value": GEMINI})
    assert r.status_code == 302 and r["Location"] == "/account/ai/"
    assert ScopedSetting.objects.get().value == GEMINI
    assert "Saved." in [m.message for m in get_messages(r.wsgi_request)]

    r = c.post("/llm/settings/save/", {**form, "value": "x/nope"})
    assert ScopedSetting.objects.get().value == GEMINI
    assert any("allowed" in m.message for m in get_messages(r.wsgi_request))

    c.post("/llm/settings/save/", {**form, "clear": "1"})
    assert not ScopedSetting.objects.exists()
    r = c.post("/llm/settings/save/", {**form, "value": GEMINI, "next": "https://evil.example/"})
    assert r["Location"] == "/"  # never redirects off-site


def test_keys_and_budgets_tags_and_forms(configured, world):
    alice = Client()
    alice.force_login(world["alice"])
    alice.post("/llm/settings/keys/add/", {"scope": "tenant", "provider": "anthropic", "key": "sk-ant-zzzz98765"})
    row = ProviderKey.objects.get()
    html = render(world["alice"], '{% llm_kit_keys scope="tenant" %}')
    assert "…8765" in html and "sk-ant" not in html and "Add key" in html

    alice.post("/llm/settings/budgets/save/", {"scope": "member", "period": "day", "amount": "2"})
    assert ScopedBudget.objects.get().amount == 2
    html = render(world["alice"], '{% llm_kit_budgets scope="member" %}')
    assert "$2" in html and "Set budget" in html
    assert "Set budget" not in render(world["carol"], '{% llm_kit_budgets scope="member" %}')

    alice.post(f"/llm/settings/keys/{row.pk}/remove/", {"scope": "tenant"})
    alice.post("/llm/settings/budgets/save/", {"scope": "member", "period": "day", "remove": "1"})
    assert not ProviderKey.objects.exists() and not ScopedBudget.objects.exists()

    carol = Client()
    carol.force_login(world["carol"])
    r = carol.post("/llm/settings/keys/add/", {"scope": "tenant", "provider": "openai", "key": "sk-xxxxxxxxxx"})
    assert not ProviderKey.objects.exists()
    assert any("can't" in m.message for m in get_messages(r.wsgi_request))
    assert Client().post("/llm/settings/save/", {"scope": "user"}).status_code == 302  # to the login page
    assert not ScopedSetting.objects.exists()
