"""Model construction backends: LangChain's init_chat_model and LiteLLM."""

import importlib.util

import pytest

from llm_kit.core.touchpoints import touchpoint
from llm_kit.core.touchpoints.backends import TouchpointChatModel, get_factory
from tests.test_touchpoints_core import FakeEnv


def test_langchain_backend_maps_providers_and_spreads_keys(monkeypatch):
    calls = []

    def fake_init(model, **kwargs):
        from tests.test_touchpoints_core import Echo

        calls.append((model, kwargs))
        return Echo(messages=iter([]))

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init)
    t = touchpoint(
        "t.lc", default="gemini/gemini-3.8-flash", fallbacks=["vertex_ai/gemini-3.8-flash"], backend="langchain"
    )
    t.model(env=FakeEnv(keys={"gemini": ["g1", "g2"]}), temperature=0)
    (m1, kw1), (m2, kw2) = calls
    assert m1 == "gemini-3.8-flash" and kw1["model_provider"] == "google_genai" and kw1["api_key"] in ("g1", "g2")
    assert kw1["temperature"] == 0
    assert m2 == "gemini-3.8-flash" and kw2["model_provider"] == "google_vertexai" and "api_key" not in kw2  # IAM auth


def test_auto_backend_picks_what_is_installed(monkeypatch):
    import llm_kit.core.touchpoints.backends as b

    monkeypatch.setattr(b, "_litellm_installed", lambda: False)
    assert get_factory("auto") is b._langchain_model
    monkeypatch.setattr(b, "_litellm_installed", lambda: True)
    assert get_factory("auto") is b._litellm_model


HAS_LITELLM = importlib.util.find_spec("langchain_litellm") is not None  # without importing litellm

needs_litellm = pytest.mark.skipif(not HAS_LITELLM, reason="pip install django-llm-kit[litellm]")


@needs_litellm
def test_litellm_single_key_and_params():
    from langchain_litellm import ChatLiteLLM

    t = touchpoint(
        "t.ll", default="anthropic/claude-sonnet-5-5", params={"temperature": 0, "custom_option": 5}, backend="litellm"
    )
    llm = t.model(env=FakeEnv(keys={"anthropic": ["sk-1"]}))
    assert isinstance(llm, ChatLiteLLM)
    assert llm.model == "anthropic/claude-sonnet-5-5" and llm.api_key == "sk-1" and llm.temperature == 0
    assert llm.model_kwargs["custom_option"] == 5  # not a ChatLiteLLM field -> passed through
    assert llm.metadata["llm_kit_touchpoint"] == "t.ll"


@needs_litellm
def test_litellm_key_pool_uses_a_router():
    from langchain_litellm import ChatLiteLLMRouter

    t = touchpoint("t.pool", default="anthropic/claude-sonnet-5-5", backend="litellm")
    llm = t.model(env=FakeEnv(keys={"anthropic": ["sk-1", "sk-2", "sk-3"]}))
    assert isinstance(llm, ChatLiteLLMRouter)
    assert [d["litellm_params"]["api_key"] for d in llm.router.model_list] == ["sk-1", "sk-2", "sk-3"]
    # Built once per (model, keys): the Router (with its state) is reused across calls.
    assert t.model(env=FakeEnv(keys={"anthropic": ["sk-1", "sk-2", "sk-3"]})).router is llm.router


@needs_litellm
def test_litellm_call_and_fallback_with_mock_responses():
    t = touchpoint(
        "t.mock",
        default="anthropic/claude-sonnet-5-5",
        fallbacks=["gemini/gemini-3.8-flash"],
        params={"mock_response": "mocked reply"},
        backend="litellm",
    )
    llm = t.model(env=FakeEnv())
    assert isinstance(llm, TouchpointChatModel)
    assert llm.invoke("hi").content == "mocked reply"
    assert "".join(c.content for c in llm.stream("hi")) == "mocked reply"


@needs_litellm
def test_litellm_import_does_not_fetch_the_remote_price_map(monkeypatch):
    import os

    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    touchpoint("t.env", default="anthropic/claude-sonnet-5-5", backend="litellm").model(env=FakeEnv())
    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "False")  # an explicit choice is kept
    touchpoint("t.env", default="anthropic/claude-sonnet-5-5", backend="litellm").model(env=FakeEnv())
    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "False"
