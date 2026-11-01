import pytest
from django.core.exceptions import ImproperlyConfigured

from llm_kit.models import LLMModel
from llm_kit.runtime import legacy_catalogue as catalogue

pytestmark = pytest.mark.django_db


def test_db_entry_wins_over_settings_alias(fake_init, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "MODELS": {"fast": {"model": "openai:gpt-5-mini", "temperature": 1}}}
    assert catalogue.resolve("fast").source == "settings"
    LLMModel.objects.create(name="fast", model="anthropic:claude-haiku-5-5", params={"temperature": 0})
    catalogue.get_chat_model("fast", max_tokens=10)
    assert fake_init[-1] == ("anthropic:claude-haiku-5-5", {"temperature": 0, "max_tokens": 10})


def test_inactive_entries_are_ignored(settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "MODELS": {}}
    LLMModel.objects.create(name="old", model="openai:gpt-4", is_active=False)
    with pytest.raises(catalogue.UnknownModel):
        catalogue.resolve("old")


def test_raw_provider_id_and_unknown_name():
    assert catalogue.resolve("openai:gpt-5").source == "raw"
    with pytest.raises(catalogue.UnknownModel):
        catalogue.resolve("nope")


def test_default_resolution(settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "DEFAULT_MODEL": "anthropic:claude-sonnet-5-5"}
    assert catalogue.resolve().model == "anthropic:claude-sonnet-5-5"
    LLMModel.objects.create(name="a", model="x:a", is_default=True)
    b = LLMModel.objects.create(name="b", model="x:b", is_default=True)
    assert list(LLMModel.objects.filter(is_default=True)) == [b]  # only one default
    assert catalogue.resolve().name == "b"


def test_no_default_raises(settings):
    settings.LLM_KIT = {k: v for k, v in settings.LLM_KIT.items() if k != "DEFAULT_MODEL"}
    with pytest.raises(catalogue.UnknownModel):
        catalogue.resolve()


def test_api_key_env(fake_init, monkeypatch):
    LLMModel.objects.create(name="k", model="openai:gpt-5", api_key_env="MY_KEY")
    monkeypatch.delenv("MY_KEY", raising=False)
    with pytest.raises(ImproperlyConfigured):
        catalogue.get_chat_model("k")
    monkeypatch.setenv("MY_KEY", "secret")
    catalogue.get_chat_model("k")
    assert fake_init[-1][1]["api_key"] == "secret"


def test_available_merges_db_and_settings():
    LLMModel.objects.create(name="db-one", model="x:y")
    names = [s.name for s in catalogue.available()]
    assert names == ["db-one", "fake"]
