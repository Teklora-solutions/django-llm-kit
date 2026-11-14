"""Provider readiness: configured providers and their credentials, without any calls or secrets (#33)."""

import json

import pytest
from rest_framework.test import APIClient

from llm_kit import monitoring
from llm_kit.models import ProviderKey

pytestmark = pytest.mark.django_db

SECRET = "sk-ant-very-secret-value-1234567890"


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEYS", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEYS"):
        monkeypatch.delenv(var, raising=False)


def _tp(report, name):
    return next(t for t in report["touchpoints"] if t["name"] == name)


def _provider(report, name):
    return next(p for p in report["providers"] if p["provider"] == name)


def test_a_fallback_without_credentials_is_flagged_before_any_call(clean_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    report = monitoring.readiness()
    chat = _tp(report, "tests.chat")  # default anthropic, fallback gemini
    assert [(m["role"], m["provider"], m["ready"]) for m in chat["models"]] == [
        ("default", "anthropic", True),
        ("fallback", "gemini", False),
    ]
    assert chat["ready"] is True and chat["failover_ready"] is False and report["ready"] is False
    gemini = _provider(report, "gemini")
    assert gemini["configured"] is False and gemini["auth_type"] == "missing" and "tests.chat" in gemini["touchpoints"]
    anthropic = _provider(report, "anthropic")
    assert anthropic["auth_type"] == "api_key" and anthropic["sources"] == {"environment": 1}
    assert SECRET not in json.dumps(report, default=str)


def test_stored_keys_and_their_last_test(clean_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    key = ProviderKey(provider="gemini", label="main")
    key.set_key("AIza-test-key-1234567890")
    key.save()
    gemini = _provider(monitoring.readiness(), "gemini")
    assert gemini["ready"] and gemini["sources"] == {"database": 1} and gemini["key_checks"]["unchecked"] == 1

    ProviderKey.objects.filter(pk=key.pk).update(last_check_ok=False, last_check_error="401 invalid key")
    gemini = _provider(monitoring.readiness(), "gemini")
    assert gemini["configured"] and not gemini["ready"]
    assert gemini["problems"] == ["every stored key failed its last test"]
    assert gemini["key_checks"]["failed"] == 1 and gemini["key_checks"]["last"]["error"] == "401 invalid key"
    assert "AIza-test-key" not in json.dumps(monitoring.readiness(), default=str)


def test_keyless_and_credentials_source_providers(clean_env, settings):
    settings.LLM_KIT = {
        **settings.LLM_KIT,
        "CREDENTIALS": lambda provider: "vault-token" if provider == "gemini" else None,
    }
    report = monitoring.readiness()
    ollama = _provider(report, "ollama")  # the script touchpoints run on ollama
    assert ollama["ready"] and ollama["auth_type"] == "none" and ollama["sources"] == {}
    gemini = _provider(report, "gemini")
    assert gemini["ready"] and gemini["auth_type"] == "credentials_source" and gemini["sources"] == {"credentials": 1}


def test_a_failing_credentials_source_is_reported(clean_env, settings):
    def broken(provider):
        raise RuntimeError("vault down")

    settings.LLM_KIT = {**settings.LLM_KIT, "CREDENTIALS": broken}
    gemini = _provider(monitoring.readiness(), "gemini")
    assert not gemini["ready"] and gemini["problems"] == ["CREDENTIALS source failed: RuntimeError"]


def test_admin_overrides_are_what_counts(clean_env, monkeypatch):
    from llm_kit.models import TouchpointOverride

    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    TouchpointOverride.objects.create(name="tests.chat", fallbacks=[])
    chat = _tp(monitoring.readiness(), "tests.chat")
    assert [m["role"] for m in chat["models"]] == ["default"] and chat["failover_ready"] is None


def test_api_is_for_monitoring_staff_only(clean_env, django_user_model):
    api = APIClient()
    api.force_authenticate(django_user_model.objects.create_user("plain"))
    assert api.get("/llm/api/monitoring/readiness/").status_code == 403
    api.force_authenticate(django_user_model.objects.create_user("staff", is_staff=True))
    r = api.get("/llm/api/monitoring/readiness/")
    assert r.status_code == 200 and {"ready", "touchpoints", "providers"} <= set(r.data)
    assert api.get("/llm/api/monitoring/readiness/?period=1h").status_code == 400
