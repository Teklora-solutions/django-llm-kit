"""Testing a provider key with one tiny call: the check, recording it, and its API, form and admin."""

import pytest
from django.test import Client
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from llm_kit.credentials.check import check_key, model_for
from llm_kit.models import ProviderKey

pytestmark = pytest.mark.django_db

CALLS: list[tuple[str, list[str], dict]] = []


class KeyCheckModel(GenericFakeChatModel):
    keys: list[str] = []

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if any(k.startswith("sk-bad") for k in self.keys):
            raise PermissionError("401 Unauthorized: invalid x-api-key")
        if any(k.startswith("sk-busy") for k in self.keys):
            raise RuntimeError("429 rate limit exceeded, retry in 5s")
        self.messages = iter([AIMessage(content="OK")])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kw)


def backend(ref, params, keys):
    CALLS.append((ref, list(keys), dict(params)))
    return KeyCheckModel(messages=iter([]), keys=list(keys))


@pytest.fixture(autouse=True)
def setup(settings, world):
    from llm_kit import catalogue
    from tests.catalogue_payloads import FakeSources

    catalogue.sync(fetch=FakeSources())
    CALLS.clear()
    settings.LLM_KIT = {
        **settings.LLM_KIT,
        "MODEL_BACKEND": "tests.test_key_checks:backend",
        "BYOK_SCOPES": ("tenant", "user"),
        "SCOPE_PERMISSIONS": {"tenant": "tests.testapp.tenancy:can_manage"},
    }


def test_a_key_is_tried_alone_on_a_cheap_model():
    assert check_key("anthropic", "sk-good-1234").ok
    ref, keys, params = CALLS[-1]
    assert ref.startswith("anthropic/") and keys == ["sk-good-1234"] and params["max_tokens"] == 5
    assert ref == model_for("anthropic")


def test_refused_and_accepted_keys():
    bad = check_key("anthropic", "sk-bad")
    assert not bad.ok and bad.error == "auth" and "401" in bad.message
    busy = check_key("anthropic", "sk-busy")  # rate limited: the provider accepted the key
    assert busy.ok and "rate_limit" in busy.message
    unknown = check_key("nosuchprovider", "sk-x-12345")
    assert not unknown.ok and unknown.error == "no_model"


def test_the_api_tests_and_verifies_keys(world):
    c = Client()
    c.force_login(world["carol"])
    r = c.post(
        "/llm/api/keys/",
        {"scope": "user", "provider": "anthropic", "key": "sk-bad-key-0001", "verify": True},
        content_type="application/json",
    )
    assert r.status_code == 400 and "refused" in r.json()["detail"] and not ProviderKey.objects.exists()

    r = c.post(
        "/llm/api/keys/",
        {"scope": "user", "provider": "anthropic", "key": "sk-good-5678", "verify": True},
        content_type="application/json",
    )
    assert r.status_code == 201 and r.json()["works"] is True and r.json()["checked_at"]

    row = ProviderKey.objects.get()
    row.set_key("sk-bad-key-0002")
    row.save()
    r = c.post(f"/llm/api/keys/{row.pk}/test/?scope=user")
    assert r.status_code == 200 and r.json()["works"] is False and r.json()["check_error"].startswith("auth")
    assert c.post("/llm/api/keys/999/test/?scope=user").status_code == 404
    other = Client()
    other.force_login(world["bob"])
    assert other.post(f"/llm/api/keys/{row.pk}/test/?scope=user").status_code == 404  # not his


def test_the_form_and_the_admin(world, django_user_model):
    from django.contrib.messages import get_messages

    c = Client()
    c.force_login(world["carol"])
    r = c.post(
        "/llm/settings/keys/add/", {"scope": "user", "provider": "anthropic", "key": "sk-bad-key-0001", "verify": "on"}
    )
    assert not ProviderKey.objects.exists()
    assert any("refused" in m.message for m in get_messages(r.wsgi_request))
    c.post("/llm/settings/keys/add/", {"scope": "user", "provider": "anthropic", "key": "sk-good-9999"})
    row = ProviderKey.objects.get()
    assert row.last_check_ok is None  # not checked: the box was off
    r = c.post(f"/llm/settings/keys/{row.pk}/test/", {"scope": "user"})
    row.refresh_from_db()
    assert row.last_check_ok is True and any("works" in m.message for m in get_messages(r.wsgi_request))

    admin = django_user_model.objects.create_superuser("root", "root@example.com", "pw")
    a = Client()
    a.force_login(admin)
    row.set_key("sk-bad-key-0002")
    row.save()
    a.post("/admin/llm_kit/providerkey/", {"action": "test_keys", "_selected_action": [row.pk]})
    row.refresh_from_db()
    assert row.last_check_ok is False
