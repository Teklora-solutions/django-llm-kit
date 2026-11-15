"""adapt_input: each candidate model gets input it can read, fallbacks included."""

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from llm_kit.core.errors import AllModelsFailed
from llm_kit.core.touchpoints.ratelimit import Limiter, Limits, MemoryStore
from tests.test_touchpoints_core import FakeEnv, tp

SEEN: list = []


def tag_with_ref(input, ref, info):
    """Tells each model which model it is, and whether the catalogue knew it."""
    SEEN.append((ref, info.ref if info else None))
    text = input if isinstance(input, str) else input[-1].content
    return [HumanMessage(content=f"[{ref}] {text}")]


async def async_tag(input, ref, info):
    return tag_with_ref(input, ref, info)


def only_gemini(input, ref, info):
    """Media only Gemini can open: other models can't take this input."""
    if not ref.startswith("gemini/"):
        raise ValueError(f"{ref} can't open a Files API reference")
    return tag_with_ref(input, ref, info)


@pytest.fixture(autouse=True)
def clean():
    SEEN.clear()
    yield
    SEEN.clear()


def _chat(**kw):
    kw.setdefault("fallbacks", ["gemini/gemini-3.8-flash"])
    return tp(**kw)


def test_the_adapter_runs_for_each_model_tried(monkeypatch):
    t = _chat(default="down/model", adapt_input="tests.test_adapt_input:tag_with_ref")
    env = FakeEnv(keys={"down": ["d"], "gemini": ["g"]})
    reply = t.model(env=env, retry="never").invoke("hello")
    assert reply.content == "echo: [gemini/gemini-3.8-flash] hello"  # the fallback got its own input
    assert SEEN == [("down/model", None), ("gemini/gemini-3.8-flash", "gemini/gemini-3.8-flash")]


async def test_async_calls_and_async_adapters():
    t = _chat(adapt_input=async_tag)
    llm = t.model(env=FakeEnv(), retry="never")
    assert (await llm.ainvoke("hi")).content == "echo: [anthropic/claude-sonnet-5-5] hi"
    chunks = [c.content async for c in llm.astream("hi")]
    assert "".join(chunks) == "echo: [anthropic/claude-sonnet-5-5] hi"


def test_streams_and_a_sync_call_with_an_async_adapter():
    llm = _chat(adapt_input=async_tag).model(env=FakeEnv(), retry="never")
    assert "".join(c.content for c in llm.stream("hi")) == "echo: [anthropic/claude-sonnet-5-5] hi"
    assert llm.invoke("hi").content == "echo: [anthropic/claude-sonnet-5-5] hi"


def test_a_model_the_adapter_cant_serve_is_skipped():
    llm = _chat(adapt_input=only_gemini).model(env=FakeEnv(), retry="never")
    assert llm.invoke("listen").content == "echo: [gemini/gemini-3.8-flash] listen"
    alone = _chat(adapt_input=only_gemini, fallbacks=[]).model(env=FakeEnv(), retry="never")
    with pytest.raises(ValueError, match="Files API"):
        alone.invoke("listen")
    nobody = _chat(adapt_input=only_gemini, fallbacks=["openai/gpt-5.5"]).model(env=FakeEnv(), retry="never")
    with pytest.raises(AllModelsFailed):
        nobody.invoke("listen")


def test_it_survives_bind_tools_and_structured_output():
    @tool
    def double(x: int) -> int:
        """Double x."""
        return 2 * x

    llm = _chat(adapt_input=tag_with_ref).model(env=FakeEnv(), retry="never")
    assert llm.bind_tools([double]).invoke("hi").content == "echo: [anthropic/claude-sonnet-5-5] hi"

    # with_structured_output goes through the same _map as bind_tools: every candidate, same adapter.
    mapped = llm._map(lambda m: m.bind(stop=["\n\n"]))
    assert mapped.adapt is llm.adapt and mapped.infos == llm.infos
    assert mapped.invoke("hi").content == "echo: [anthropic/claude-sonnet-5-5] hi"


def test_it_runs_before_the_rate_limit_slot():
    limiter = Limiter(MemoryStore(), lambda model: Limits(rpm=5))

    class Env(FakeEnv):
        def rate_limiter(self):
            return limiter

    llm = _chat(adapt_input=only_gemini).model(env=Env(), retry="never")
    llm.invoke("listen")
    assert limiter.usage("anthropic/claude-sonnet-5-5")["rpm_used"] == 0  # skipped before taking a slot
    assert limiter.usage("gemini/gemini-3.8-flash")["rpm_used"] == 1


def test_per_call_override_and_turning_it_off():
    t = _chat(adapt_input=only_gemini)
    assert (
        t.model(env=FakeEnv(), retry="never", adapt_input=tag_with_ref)
        .invoke("x")
        .content.startswith("echo: [anthropic/")
    )
    plain = t.model(env=FakeEnv(), retry="never", adapt_input=False)
    assert plain.invoke("x").content == "echo: x"
    assert t.complete("y", env=FakeEnv(), retry="never", adapt_input=tag_with_ref).content.startswith(
        "echo: [anthropic/"
    )


def test_a_bad_adapter_is_refused():
    from llm_kit.core.errors import ConfigError

    with pytest.raises(ConfigError, match="adapt_input"):
        _chat(adapt_input=42).model(env=FakeEnv(), retry="never")


# ------------------------------------------------------------------ recording


@pytest.fixture
def sink():
    from llm_kit.core.observability import flush, recorder
    from tests.test_observability_core import MemorySink

    flush()  # calls queued by earlier tests go nowhere, not into this sink
    s = MemorySink()
    recorder.configure([s], None)
    yield s
    flush()
    recorder.configure([], None)


@pytest.mark.parametrize("capture", [True, False])
def test_recording_sees_the_adapted_input_only_when_capturing(sink, capture, monkeypatch):
    from llm_kit.core import config
    from llm_kit.core.observability import flush

    real = config.get
    monkeypatch.setattr(config, "get", lambda key: capture if key == "CAPTURE_CONTENT" else real(key))
    _chat(adapt_input=tag_with_ref).model(env=FakeEnv(), retry="never").invoke("secret words")
    assert flush()
    (call,) = sink.calls
    if capture:
        assert "[anthropic/claude-sonnet-5-5] secret words" in str(call.prompt)
    else:
        assert call.prompt is None and call.completion is None
