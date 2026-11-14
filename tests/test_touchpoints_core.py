"""Touchpoint policy, resolution and model building, without Django."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from llm_kit.core.errors import ConfigError, ModelNotAllowed, TouchpointDisabled, UnknownTouchpoint
from llm_kit.core.touchpoints import (
    ClientChoice,
    ModelInfo,
    Override,
    ServerChoice,
    get_touchpoint,
    problems,
    touchpoint,
)
from llm_kit.core.touchpoints.backends import TouchpointChatModel
from llm_kit.core.touchpoints.structured import extract_json

CHAT = {
    "kind": "chat",
    "tool_call": True,
    "input_modalities": ["text", "image"],
    "price_input": "2",
    "price_output": "10",
}


class FakeEnv:
    def __init__(self, catalogue=None, overrides=None, keys=None):
        self.catalogue = (
            catalogue
            if catalogue is not None
            else {
                "anthropic/claude-sonnet-5-5": ModelInfo("anthropic/claude-sonnet-5-5", CHAT),
                "anthropic/claude-opus-5-5": ModelInfo("anthropic/claude-opus-5-5", {**CHAT, "price_input": "15"}),
                "anthropic/claude-hidden": ModelInfo("anthropic/claude-hidden", CHAT, hidden=True),
                "gemini/gemini-3.8-flash": ModelInfo("gemini/gemini-3.8-flash", {**CHAT, "price_input": "0.75"}),
                "openai/gpt-5.5": ModelInfo("openai/gpt-5.5", CHAT),
                "openai/text-only": ModelInfo("openai/text-only", {**CHAT, "tool_call": False}),
            }
        )
        self.overrides = overrides or {}
        self.keys = keys if keys is not None else {"anthropic": ["a"], "gemini": ["g"], "openai": ["o"]}

    def lookup(self, ref):
        ref = ref.replace(":", "/", 1) if "/" not in ref else ref
        return self.catalogue.get(ref.replace("google/", "gemini/"))

    def override(self, name):
        return self.overrides.get(name)

    def api_keys(self, provider):
        return self.keys.get(provider, [])


class Echo(GenericFakeChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.messages = iter([AIMessage(content=f"echo: {messages[-1].content}")])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kw)

    def bind_tools(self, tools, **kw):
        return self.bind(tools=[t.name for t in tools])


class Down(Echo):
    def _generate(self, *a, **k):
        raise ConnectionError("provider down")


BUILT = []


def backend(ref, params, keys):
    BUILT.append((ref, params, keys))
    return Down(messages=iter([])) if ref.startswith("down/") else Echo(messages=iter([]))


@pytest.fixture(autouse=True)
def _reset():
    BUILT.clear()


def tp(name="t.chat", **kw):
    kw.setdefault("default", "anthropic/claude-sonnet-5-5")
    return touchpoint(name, backend="tests.test_touchpoints_core:backend", **kw)


# ------------------------------------------------------------------ policy


def test_server_choice_ignores_or_rejects_requests():
    env = FakeEnv()
    r = tp().resolve("gemini/gemini-3.8-flash", env=env)
    assert r.model == "anthropic/claude-sonnet-5-5" and r.chosen_by == "code"
    assert "lets the server choose; ignored" in r.notes[0]
    with pytest.raises(ModelNotAllowed):
        tp(choice=ServerChoice(on_requested="reject")).resolve("gemini/gemini-3.8-flash", env=env)


def test_client_choice_allowed_and_normalised():
    t = tp(choice=ClientChoice(allow=("anthropic/*", "gemini/*")), fallbacks=["gemini/gemini-3.8-flash"])
    r = t.resolve("anthropic:claude-opus-5-5", env=FakeEnv())  # init_chat_model spelling
    assert (r.model, r.chosen_by, r.fallbacks) == ("anthropic/claude-opus-5-5", "client", ["gemini/gemini-3.8-flash"])
    r = t.resolve(config={"configurable": {"model": "google/gemini-3.8-flash"}}, env=FakeEnv())
    assert r.model == "gemini/gemini-3.8-flash" and r.fallbacks == []  # fallback equal to primary is dropped


@pytest.mark.parametrize(
    ("choice", "requested", "reason"),
    [
        (ClientChoice(allow=("anthropic/*",)), "openai/gpt-5.5", "not in allowed patterns"),
        (ClientChoice(requires={"tool_call": True}), "openai/text-only", "tool_call is False"),
        (ClientChoice(requires={"input": ["audio"]}), "openai/gpt-5.5", "input modalities lack ['audio']"),
        (ClientChoice(max_price_input=5), "anthropic/claude-opus-5-5", "price_input 15 exceeds cap 5"),
        (ClientChoice(requires={"tool_call": True}), "mystery/model", "not in the model catalogue"),
        (ClientChoice(), "anthropic/claude-hidden", "hidden by an admin"),
        (ClientChoice(), "openrouter/some-model", "no API key configured"),
    ],
)
def test_client_choice_rejections(choice, requested, reason):
    with pytest.raises(ModelNotAllowed) as e:
        tp(choice=choice).resolve(requested, env=FakeEnv())
    assert any(reason in r for r in e.value.reasons), e.value.reasons


def test_touchpoint_requires_apply_to_client_picks_too():
    t = tp(requires={"tool_call": True}, choice=ClientChoice())
    with pytest.raises(ModelNotAllowed):
        t.resolve("openai/text-only", env=FakeEnv())


def test_client_choice_without_requirements_accepts_unknown_models():
    r = tp(choice=ClientChoice()).resolve("mystery/model", env=FakeEnv(keys={"mystery": ["k"], "anthropic": ["a"]}))
    assert r.model == "mystery/model" and r.chosen_by == "client"


def test_disallowed_pick_can_fall_back_to_default():
    t = tp(choice=ClientChoice(allow=("anthropic/*",), on_disallowed="ignore"))
    r = t.resolve("openai/gpt-5.5", env=FakeEnv())
    assert r.model == "anthropic/claude-sonnet-5-5" and r.chosen_by == "code" and "not allowed" in r.notes[0]


def test_client_pick_can_drop_fallbacks():
    t = tp(choice=ClientChoice(use_fallbacks=False), fallbacks=["gemini/gemini-3.8-flash"])
    assert t.resolve("openai/gpt-5.5", env=FakeEnv()).fallbacks == []
    assert t.resolve(env=FakeEnv()).fallbacks == ["gemini/gemini-3.8-flash"]


# --------------------------------------------------------------- overrides


def test_admin_override_changes_default_fallbacks_params():
    env = FakeEnv(
        overrides={"t.chat": Override(default="gemini/gemini-3.8-flash", fallbacks=[], params={"max_tokens": 5})}
    )
    r = tp(fallbacks=["openai/gpt-5.5"], params={"temperature": 0}).resolve(env=env)
    assert (r.model, r.fallbacks, r.chosen_by) == ("gemini/gemini-3.8-flash", [], "admin")
    assert r.params == {"temperature": 0, "max_tokens": 5}


def test_admin_allow_narrows_but_never_widens():
    t = tp(choice=ClientChoice(allow=("anthropic/*",)))
    env = FakeEnv(overrides={"t.chat": Override(allow=["anthropic/claude-sonnet-*", "openai/*"])})
    assert t.resolve("anthropic/claude-sonnet-5-5", env=env).chosen_by == "client"
    with pytest.raises(ModelNotAllowed, match="admin-allowed"):
        t.resolve("anthropic/claude-opus-5-5", env=env)  # narrowed away by admin
    with pytest.raises(ModelNotAllowed, match="not in allowed patterns"):
        t.resolve("openai/gpt-5.5", env=env)  # admin can't widen past code


def test_kill_switch():
    with pytest.raises(TouchpointDisabled):
        tp().resolve(env=FakeEnv(overrides={"t.chat": Override(disabled=True)}))


# ------------------------------------------------------------- credentials


def test_fallbacks_without_keys_are_skipped_and_primary_promoted():
    env = FakeEnv(keys={"gemini": ["g"]})
    r = tp(fallbacks=["openai/gpt-5.5", "gemini/gemini-3.8-flash"]).resolve(env=env)
    assert r.model == "gemini/gemini-3.8-flash" and r.fallbacks == []
    assert ("openai/gpt-5.5", "no API key configured") in r.skipped
    assert ("anthropic/claude-sonnet-5-5", "no API key configured") in r.skipped


def test_keyless_providers_count_as_configured():
    r = tp(default="vertex_ai/gemini-3.8-flash", fallbacks=["ollama/llama4"]).resolve(env=FakeEnv(keys={}))
    assert r.candidates == ["vertex_ai/gemini-3.8-flash", "ollama/llama4"]


def test_problems_report():
    env = FakeEnv(keys={"anthropic": ["a"]})
    env.catalogue["anthropic/claude-sonnet-5-5"].values.update(status="deprecated")
    found = problems(tp(fallbacks=["gemini/gemini-3.8-flash", "nope/x"], requires={"reasoning": True}), env)
    joined = "\n".join(found)
    assert "default anthropic/claude-sonnet-5-5: reasoning is None, need True" in joined
    assert "default anthropic/claude-sonnet-5-5 is deprecated" in joined
    assert "fallback gemini/gemini-3.8-flash has no API key configured" in joined
    assert "fallback nope/x is not in the catalogue" in joined
    env.catalogue["anthropic/claude-sonnet-5-5"].values.pop("status")


# ------------------------------------------------------------------ models


def test_model_single_candidate_is_plain_chat_model_with_metadata():
    llm = tp(params={"temperature": 0}).model(env=FakeEnv(), max_tokens=7, retry="never")
    assert isinstance(llm, Echo)
    assert llm.metadata["llm_kit_touchpoint"] == "t.chat" and llm.metadata["llm_kit_chosen_by"] == "code"
    assert "llm_kit:t.chat" in llm.tags
    assert BUILT == [("anthropic/claude-sonnet-5-5", {"temperature": 0, "max_tokens": 7}, ["a"])]


def test_fallback_is_used_when_primary_fails():
    env = FakeEnv(keys={"down": ["x"], "gemini": ["g"]})
    env.catalogue["down/model"] = ModelInfo("down/model", CHAT)
    llm = tp(default="down/model", fallbacks=["gemini/gemini-3.8-flash"]).model(env=env)
    assert isinstance(llm, TouchpointChatModel)
    assert llm.invoke("hi").content == "echo: hi"


def test_bind_tools_keeps_fallbacks():
    @tool
    def double(x: int) -> int:
        """Double x."""
        return 2 * x

    llm = tp(fallbacks=["gemini/gemini-3.8-flash"]).model(env=FakeEnv()).bind_tools([double])
    assert isinstance(llm, TouchpointChatModel) and len(llm.models) == 2
    assert all(m.kwargs["tools"] == ["double"] for m in llm.models)


class Lead(BaseModel):
    name: str
    score: int


class JsonReply(Echo):
    """Answers in prose-wrapped JSON instead of calling the structured-output tool."""

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.messages = iter([AIMessage(content='Sure!\n```json\n{"name": "Ada", "score": 9}\n```')])
        return GenericFakeChatModel._generate(self, messages, stop=stop, run_manager=run_manager, **kw)


class JsonReplyNoTools(JsonReply):
    def bind_tools(self, tools, **kw):
        raise NotImplementedError


def json_backend(ref, params, keys):
    return JsonReply(messages=iter([]))


def json_no_tools_backend(ref, params, keys):
    return JsonReplyNoTools(messages=iter([]))


@pytest.mark.parametrize("backend_name", ["json_backend", "json_no_tools_backend"])
def test_complete_returns_schema_even_when_model_answers_in_text(backend_name):
    t = touchpoint(
        "t.json", default="anthropic/claude-sonnet-5-5", backend=f"tests.test_touchpoints_core:{backend_name}"
    )
    assert t.complete("Extract the lead", schema=Lead, env=FakeEnv()) == Lead(name="Ada", score=9)


def test_complete_without_schema_returns_message():
    assert tp().complete("plain", env=FakeEnv()).content == "echo: plain"


async def test_acomplete():
    assert (await tp().acomplete("hey", env=FakeEnv())).content == "echo: hey"


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Here you go: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    assert extract_json("```\n[1, 2]\n```") == [1, 2]
    with pytest.raises(ValueError):
        extract_json("no json here")


# ---------------------------------------------------------------- registry


def test_registry_and_validation():
    t = tp("t.registry")
    assert get_touchpoint("t.registry") is t
    with pytest.raises(UnknownTouchpoint):
        get_touchpoint("t.nope")
    with pytest.raises(ConfigError, match="unknown requirement"):
        touchpoint("t.bad", default="x/y", requires={"vision": True})


def test_custom_backend_must_be_callable():
    from llm_kit.core.touchpoints.backends import get_factory

    with pytest.raises(ConfigError):
        get_factory("llm_kit.core.config:DEFAULTS")
