"""Call/run/tool recording without Django: handler, recorder, pricing, error classes."""

import asyncio
from decimal import Decimal

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from llm_kit.core.observability import classify_error, context, flush, recorder
from llm_kit.core.observability.cost import price_call
from llm_kit.core.runs import build_config
from llm_kit.core.touchpoints import touchpoint
from tests.test_touchpoints_core import FakeEnv

PRICES = {
    "anthropic/claude-sonnet-5-5": {"price_input": "2", "price_output": "10", "price_cache_read": "0.2"},
    "gemini/gemini-3.8-flash": {
        "price_input": "0.75",
        "price_output": "3.75",
        "price_tiers": [{"above_tokens": 200000, "price_input": "1.5", "price_output": "7.5"}],
    },
}


class MemorySink:
    def __init__(self):
        self.calls, self.runs, self.tools = [], [], []

    def write(self, calls, runs, tools):
        self.calls += calls
        self.runs += runs
        self.tools += tools


@pytest.fixture
def sink():
    s = MemorySink()
    recorder.configure([s], PRICES.get)
    yield s
    flush()
    recorder.configure([], None)


def done(sink):
    assert flush()
    return sink


# ----------------------------------------------------------- fake models


class UsageModel(GenericFakeChatModel):
    """Replies with realistic usage metadata, like a provider would."""

    reply: str = "ok"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        usage = {
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "input_token_details": {"cache_read": 400},
            "output_token_details": {"reasoning": 50},
        }
        self.messages = iter(
            [AIMessage(content=self.reply, usage_metadata=usage, response_metadata={"model_name": "served-x"})]
        )
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kw)


class Down(GenericFakeChatModel):
    def _generate(self, *a, **k):
        err = RuntimeError("429 Too Many Requests")
        err.status_code = 429
        raise err


def usage_backend(ref, params, keys):
    return Down(messages=iter([])) if ref.startswith("down/") else UsageModel(messages=iter([]))


def no_usage_backend(ref, params, keys):
    return GenericFakeChatModel(messages=iter([AIMessage(content="twelve chars")]))


def tp(name="obs.tp", backend="usage_backend", **kw):
    kw.setdefault("default", "anthropic/claude-sonnet-5-5")
    return touchpoint(name, backend=f"tests.test_observability_core:{backend}", **kw)


# ------------------------------------------------------------- pricing


def test_price_call():
    p = PRICES["anthropic/claude-sonnet-5-5"]
    # 600 plain input * 2 + 400 cached * 0.2 + 200 output * 10, per 1M
    assert price_call(p, input_tokens=1000, output_tokens=200, cache_read_tokens=400) == Decimal("0.0032800000")
    tiered = PRICES["gemini/gemini-3.8-flash"]
    assert price_call(tiered, input_tokens=100, output_tokens=0) == Decimal("0.0000750000")
    assert price_call(tiered, input_tokens=300000, output_tokens=0) == Decimal("0.4500000000")  # above-200k rate
    assert price_call({"price_input": "1"}, input_tokens=1, output_tokens=1) is None  # no output price
    assert price_call(None, input_tokens=1, output_tokens=1) is None
    with_reasoning = {"price_input": "1", "price_output": "4", "price_reasoning": "8"}
    assert price_call(with_reasoning, input_tokens=0, output_tokens=100, reasoning_tokens=40) == Decimal("0.0005600000")


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (type("RateLimitError", (Exception,), {"status_code": 429})("slow down"), "rate_limit"),
        (
            type("RateLimitError", (Exception,), {"status_code": 429})("You exceeded your current quota"),
            "quota_exhausted",
        ),
        (RuntimeError("429 RESOURCE_EXHAUSTED: Resource has been exhausted (e.g. check quota)"), "quota_exhausted"),
        (TimeoutError("read timed out"), "timeout"),
        (type("AuthenticationError", (Exception,), {"status_code": 401})("bad key"), "auth"),
        (type("APIStatusError", (Exception,), {"status_code": 529})("Overloaded"), "overloaded"),
        (ValueError("This model's maximum context length is 200000 tokens"), "context_length"),
        (ConnectionError("connection reset"), "connection"),
        (type("BadRequestError", (Exception,), {"status_code": 400})("invalid"), "bad_request"),
        (asyncio.CancelledError(), "cancelled"),
        (ValueError("something else"), "other"),
        # Status and Google status names only in the message, as LangChain's Google client raises them.
        (RuntimeError("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'high demand'}}"), "overloaded"),
        (RuntimeError("{'error': {'code': 503, 'status': 'UNAVAILABLE'}}"), "overloaded"),
        (RuntimeError("429 RESOURCE_EXHAUSTED. Try later."), "rate_limit"),
        (RuntimeError("429 rate limit exceeded"), "rate_limit"),
        (RuntimeError("401 API key not valid"), "auth"),
        # Not provider statuses: lower-case "unavailable" is ordinary English.
        (type("ModelUnavailable", (Exception,), {})("no provider is usable"), "other"),
        (RuntimeError("asked for an unavailable tool 'run_sql'"), "other"),
    ],
)
def test_classify_error(exc, expected):
    assert classify_error(exc) == expected


# ------------------------------------------------------------ recording


def test_complete_records_a_run_and_a_priced_call(sink):
    with context(user="u-1", tenant="acme", tags={"job": "nightly"}):
        tp().complete("hello", env=FakeEnv())
    done(sink)
    (run,), (call,) = sink.runs, sink.calls
    assert run.name == "obs.tp" and run.status == "ok" and run.calls == 1 and run.user == "u-1"
    assert call.run_id == run.run_id and call.touchpoint == "obs.tp" and call.chosen_by == "code"
    assert (call.input_tokens, call.output_tokens, call.cache_read_tokens, call.reasoning_tokens) == (
        1000,
        200,
        400,
        50,
    )
    assert call.model == "anthropic/claude-sonnet-5-5" and call.served_model == "served-x"
    assert call.cost == Decimal("0.0032800000") and not call.unpriced and not call.usage_estimated
    assert run.cost == call.cost and run.input_tokens == 1000
    assert (call.user, call.tenant, call.tags) == ("u-1", "acme", {"job": "nightly"})
    assert call.prompt is None and len(call.prompt_hash) == 16  # content off by default


def test_fallback_attempts_are_separate_calls(sink):
    env = FakeEnv(keys={"down": ["k"], "gemini": ["g"]})
    tp(default="down/model", fallbacks=["gemini/gemini-3.8-flash"]).complete("hi", env=env)
    done(sink)
    failed, served = sorted(sink.calls, key=lambda c: c.fallback_index)
    assert (failed.fallback_index, failed.status, failed.error_class, failed.cost) == (0, "error", "rate_limit", None)
    assert failed.model == "down/model" and "429" in failed.error_message
    assert (served.fallback_index, served.status, served.model) == (1, "ok", "gemini/gemini-3.8-flash")
    assert served.resolved_model == "down/model"  # what resolution picked, vs what answered
    (run,) = sink.runs
    assert run.calls == 2 and run.status == "ok" and run.cost == served.cost


def test_missing_usage_is_estimated_and_unpriced_models_flagged(sink):
    tp(default="openai/gpt-5.5", backend="no_usage_backend").complete("x" * 400, env=FakeEnv())
    done(sink)
    (call,) = sink.calls
    assert call.usage_estimated and call.input_tokens > 0 and call.output_tokens == 3  # 12 chars / 4
    assert call.unpriced and call.cost is None
    assert sink.runs[0].unpriced_calls == 1


def test_capture_is_opt_in_per_touchpoint(sink):
    tp("obs.capture", capture=True).complete("remember me", env=FakeEnv())
    done(sink)
    (call,) = sink.calls
    assert call.prompt == [{"type": "human", "content": "remember me"}]
    assert call.completion["content"] == "ok"


def test_graph_run_records_tools_and_calls(sink):
    @tool
    def lookup(city: str) -> str:
        """Weather for a city."""
        return f"sunny in {city}"

    replies = iter(
        [
            AIMessage(content="checking", tool_calls=[{"name": "lookup", "args": {"city": "Nairobi"}, "id": "t1"}]),
            AIMessage(content="It is sunny in Nairobi"),
        ]
    )

    class ToolModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    model = ToolModel(messages=replies)

    def agent(state: MessagesState, config):
        return {"messages": [model.invoke(state["messages"], config)]}

    b = StateGraph(MessagesState)
    b.add_node("agent", agent)
    b.add_node("tools", ToolNode([lookup]))
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", tools_condition)
    b.add_edge("tools", "agent")
    graph = b.compile(checkpointer=InMemorySaver())

    cfg = build_config("thread-1", {}, {"llm_kit_graph": "weather", "user_id": "u-9"})

    async def run():
        # "updates" mode: the fake model only keeps tool_calls when not streaming tokens
        return [c async for c in graph.astream({"messages": [("user", "weather?")]}, cfg, stream_mode="updates")]

    asyncio.run(run())
    done(sink)
    (run_rec,) = sink.runs
    assert (run_rec.graph, run_rec.thread_id, run_rec.user, run_rec.status) == ("weather", "thread-1", "u-9", "ok")
    assert run_rec.calls == 2 and run_rec.tool_calls == 1
    (tool_rec,) = sink.tools
    assert tool_rec.name == "lookup" and tool_rec.status == "ok" and tool_rec.run_id == run_rec.run_id
    assert {c.node for c in sink.calls} == {"agent"} and all(c.run_id == run_rec.run_id for c in sink.calls)


def test_streamed_calls_record_time_to_first_token(sink):
    model = GenericFakeChatModel(messages=iter([AIMessage(content="one two three")]))

    def agent(state: MessagesState, config):
        return {"messages": [model.invoke(state["messages"], config)]}

    b = StateGraph(MessagesState)
    b.add_node("agent", agent)
    b.add_edge(START, "agent")
    graph = b.compile()

    async def run():
        cfg = build_config("t-stream", {})
        return [c async for c in graph.astream({"messages": [("user", "hi")]}, cfg, stream_mode="messages")]

    asyncio.run(run())
    done(sink)
    (call,) = sink.calls
    assert call.streamed and call.time_to_first_token_ms is not None and call.time_to_first_token_ms <= call.latency_ms


def test_interrupted_run_is_not_an_error(sink):
    def ask(state: MessagesState):
        interrupt("approve?")
        return {}

    b = StateGraph(MessagesState)
    b.add_node("ask", ask)
    b.add_edge(START, "ask")
    b.add_edge("ask", END)
    graph = b.compile(checkpointer=InMemorySaver())
    graph.invoke({"messages": []}, build_config("t-int", {}))
    done(sink)
    (run,) = sink.runs
    assert run.status in ("ok", "interrupted") and run.error_class == ""


def test_a_failing_sink_never_breaks_calls_or_other_sinks(sink):
    class Broken:
        def write(self, *a):
            raise RuntimeError("disk full")

    recorder.configure([Broken(), sink], PRICES.get)
    assert tp().complete("still works", env=FakeEnv()).content == "ok"
    done(sink)
    assert len(sink.calls) == 1


def test_recording_can_be_switched_off(sink, monkeypatch):
    from llm_kit.core import config

    monkeypatch.setattr(config, "_source", lambda: {"OBSERVABILITY": False})
    tp().complete("quiet", env=FakeEnv())
    done(sink)
    assert sink.calls == [] and sink.runs == []


@pytest.mark.skipif(not __import__("importlib.util").util.find_spec("langchain_litellm"), reason="needs litellm")
def test_real_litellm_usage_is_recorded(sink):
    t = touchpoint(
        "obs.litellm", default="anthropic/claude-sonnet-5-5", params={"mock_response": "hi"}, backend="litellm"
    )
    t.complete("hello", env=FakeEnv())
    done(sink)
    (call,) = sink.calls
    assert (call.input_tokens, call.output_tokens, call.usage_estimated) == (10, 20, False)  # litellm mock usage
    assert call.served_model and call.cost == Decimal("0.0002200000")  # 10*2 + 20*10 per 1M


@pytest.mark.skipif(
    not __import__("importlib.util").util.find_spec("opentelemetry.sdk"), reason="needs opentelemetry-sdk"
)
def test_otel_spans_follow_genai_conventions():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from llm_kit.core.observability import OTelSink

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    sink = OTelSink()
    sink._tracer = provider.get_tracer("llm_kit")
    recorder.configure([sink], PRICES.get)
    try:
        env = FakeEnv(keys={"down": ["k"], "gemini": ["g"]})
        tp(default="down/model", fallbacks=["gemini/gemini-3.8-flash"]).complete("hi", env=env)
        assert flush()
    finally:
        recorder.configure([], None)
    spans = sorted(exporter.get_finished_spans(), key=lambda s: s.attributes["llm_kit.fallback_index"])
    failed, ok = spans
    assert failed.name == "chat model" and failed.attributes["error.type"] == "rate_limit"
    assert failed.status.status_code.name == "ERROR"
    assert ok.attributes["gen_ai.provider.name"] == "gemini"
    assert ok.attributes["gen_ai.request.model"] == "gemini-3.8-flash"
    assert ok.attributes["gen_ai.usage.input_tokens"] == 1000 and ok.attributes["llm_kit.cost_usd"] > 0
    assert ok.end_time >= ok.start_time  # latency is whole milliseconds; fakes answer in <1 ms
