from io import StringIO

import pytest
from django.core.management import CommandError, call_command

from llm_kit import catalogue
from llm_kit.core.errors import ConfigError, TouchpointDisabled
from llm_kit.core.touchpoints import get_touchpoint
from llm_kit.credentials import api_keys
from llm_kit.models import ProviderKey, TouchpointOverride
from tests.catalogue_payloads import FakeSources

pytestmark = pytest.mark.django_db


@pytest.fixture
def synced(db):
    catalogue.sync(fetch=FakeSources())


def test_django_env_is_installed_and_uses_the_catalogue(synced, keys):
    tp = get_touchpoint("tests.chat")
    r = tp.resolve("google:gemini-3.8-flash")  # no env passed: the app's DjangoEnv
    assert (r.model, r.chosen_by) == ("gemini/gemini-3.8-flash", "client")


def test_override_applies_without_restart_and_is_invalidated(synced, keys):
    tp = get_touchpoint("tests.chat")
    assert tp.resolve().model == "anthropic/claude-sonnet-5-5"
    o = TouchpointOverride.objects.create(name="tests.chat", default="gemini/gemini-3.8-flash", fallbacks=[])
    r = tp.resolve()
    assert (r.model, r.fallbacks, r.chosen_by) == ("gemini/gemini-3.8-flash", [], "admin")
    o.disabled = True
    o.save()
    with pytest.raises(TouchpointDisabled):
        tp.resolve()
    o.delete()
    assert tp.resolve().model == "anthropic/claude-sonnet-5-5"


def test_provider_keys_are_encrypted_and_come_before_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.setenv("ANTHROPIC_API_KEYS", "pool-1, pool-2")
    k = ProviderKey(provider="anthropic", label="prod")
    k.set_key("sk-ant-secret-1234")
    k.save()
    stored = ProviderKey.objects.get(pk=k.pk)
    assert "secret" not in stored.key_encrypted and stored.fingerprint.startswith("…1234")
    assert stored.get_key() == "sk-ant-secret-1234"
    assert api_keys("anthropic") == ["sk-ant-secret-1234", "pool-1", "pool-2", "env-key"]

    stored.active = False
    stored.save()  # signal drops the cached keys
    assert api_keys("anthropic") == ["pool-1", "pool-2", "env-key"]


SECRETS = {"vertex_ai": "/run/secrets/sa.json", "codex": ["tok-1", "", "tok-2"]}


def test_a_credentials_source_comes_first_and_unlocks_resolution(monkeypatch, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "CREDENTIALS": lambda provider: SECRETS.get(provider)}
    monkeypatch.setenv("CODEX_API_KEY", "env-key")
    assert api_keys("codex") == ["tok-1", "tok-2", "env-key"]
    assert api_keys("google-vertex") == ["/run/secrets/sa.json"]  # provider names are normalised first
    assert api_keys("mistral") == []
    from llm_kit.core.touchpoints.env import has_credentials
    from llm_kit.touchpoints.env import DjangoEnv

    assert has_credentials(DjangoEnv(), "codex/gpt-5-codex")
    settings.LLM_KIT = {**settings.LLM_KIT, "CREDENTIALS": "tests.test_touchpoints_django:SECRETS.get"}
    assert api_keys("vertex_ai") == ["/run/secrets/sa.json"]  # or a dotted path


def test_encryption_key_is_required(settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "ENCRYPTION_KEYS": None}
    with pytest.raises(ConfigError, match="ENCRYPTION_KEYS"):
        ProviderKey(provider="openai").set_key("x")


def test_key_rotation(settings):
    from llm_kit.credentials import crypto

    settings.LLM_KIT = {**settings.LLM_KIT, "ENCRYPTION_KEYS": ["old"]}
    token = crypto.encrypt("value")
    settings.LLM_KIT = {**settings.LLM_KIT, "ENCRYPTION_KEYS": ["new", "old"]}
    assert crypto.decrypt(token) == "value"
    rotated = crypto.rotate(token)
    settings.LLM_KIT = {**settings.LLM_KIT, "ENCRYPTION_KEYS": ["new"]}
    assert crypto.decrypt(rotated) == "value"


def test_admin_forms(client, django_user_model, synced, keys):
    client.force_login(django_user_model.objects.create_superuser("admin", "a@example.com", "pw"))

    r = client.post(
        "/admin/llm_kit/providerkey/add/",
        {"provider": "openrouter", "label": "main", "key": "sk-or-9876", "active": "on"},
    )
    assert r.status_code == 302, r.content.decode()[:500]
    k = ProviderKey.objects.get(provider="openrouter")
    assert k.get_key() == "sk-or-9876" and k.fingerprint.startswith("…9876")
    page = client.get(f"/admin/llm_kit/providerkey/{k.pk}/change/").content.decode()
    assert "sk-or-9876" not in page  # never rendered back

    page = client.get("/admin/llm_kit/touchpointoverride/add/").content.decode()
    assert "tests.chat" in page  # name choices come from code
    r = client.post(
        "/admin/llm_kit/touchpointoverride/add/",
        {
            "name": "tests.chat",
            "default": "gemini/gemini-3.8-flash",
            "params": "{}",
            "fallbacks": "null",
            "allow": "null",
        },
    )
    assert r.status_code == 302, r.content.decode()[:800]
    o = TouchpointOverride.objects.get(name="tests.chat")
    assert o.fallbacks is None and o.allow is None  # "null" = keep the code's values
    listing = client.get("/admin/llm_kit/touchpointoverride/").content.decode()
    # Code fallback (gemini) equals the new default, so it is dropped: resolves to gemini alone.
    assert '<td class="field-resolves_to">gemini/gemini-3.8-flash</td>' in listing


def test_touchpoints_command(synced, keys):
    out = StringIO()
    call_command("llm_kit_touchpoints", stdout=out)
    text = out.getvalue()
    assert "tests.chat" in text and "resolves to: anthropic/claude-sonnet-5-5 → gemini/gemini-3.8-flash" in text
    TouchpointOverride.objects.create(name="tests.chat", default="openai/not-a-model")
    with pytest.raises(CommandError, match="problem"):
        call_command("llm_kit_touchpoints", "--check", stdout=StringIO())
