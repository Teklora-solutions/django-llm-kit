"""Embedding touchpoints: resolution, fallbacks, retry waits, backends and call records."""

import asyncio

import pytest

from llm_kit.core.errors import AllModelsFailed, ConfigError
from llm_kit.core.observability import recorder
from llm_kit.core.touchpoints import backends, touchpoint
from llm_kit.core.touchpoints.embeddings import Embedded, LangChainEmbedder, LiteLLMEmbedder, as_embedder
from tests.test_retry import GEMINI_MINUTE, RateLimitError
from tests.test_touchpoints_core import FakeEnv

FAILURES: dict[str, list[BaseException]] = {}
CALLS: list[tuple[str, list, dict]] = []


class Scripted:
    def __init__(self, ref, params, keys):
        self.ref, self.params = ref, params

    def embed(self, inputs, **params):
        CALLS.append((self.ref, list(inputs), {**self.params, **params}))
        if FAILURES.get(self.ref):
            raise FAILURES[self.ref].pop(0)
        size = int(self.params.get("dimensions") or 3)
        return Embedded(vectors=[[float(len(str(i)))] * size for i in inputs], input_tokens=7, served_model=self.ref)

    async def aembed(self, inputs, **params):
        return self.embed(inputs, **params)


def backend(ref, params, keys):
    return Scripted(ref, params, keys)


@pytest.fixture(autouse=True)
def _setup(settings, monkeypatch):
    FAILURES.clear()
    CALLS.clear()
    # No recording (its background pricing needs a database), except where a test turns it on.
    backend_path = "tests.test_embeddings:backend"
    settings.LLM_KIT = {**settings.LLM_KIT, "EMBEDDING_BACKEND": backend_path, "OBSERVABILITY": False}
    monkeypatch.setattr(backends, "_sleep", lambda s: None)

    async def asleep(s):
        return None

    monkeypatch.setattr(backends, "_asleep", asleep)


def tp(**kw):
    kw.setdefault("default", "gemini/gemini-embedding-2")
    kw.setdefault("fallbacks", ["openai/text-embedding-3-small"])
    kw.setdefault("params", {"dimensions": 4})
    return touchpoint("t.embed", **kw)


def test_embed_returns_one_vector_per_input_with_the_touchpoints_params():
    assert tp().embed(["ab", "abc"], env=FakeEnv()) == [[2.0] * 4, [3.0] * 4]
    assert tp().embed("x", env=FakeEnv(), task_type="RETRIEVAL_QUERY") == [[1.0] * 4]  # a string is one input
    assert CALLS[-1][2] == {"dimensions": 4, "task_type": "RETRIEVAL_QUERY"}


def test_media_parts_reach_the_backend_as_given():
    part = {"type": "image", "data": b"\x89PNG", "mime_type": "image/png"}
    tp().embed([part, "caption"], env=FakeEnv())
    assert CALLS[-1][1] == [part, "caption"]


def test_falls_back_and_waits_like_chat_calls():
    FAILURES["gemini/gemini-embedding-2"] = [RateLimitError(GEMINI_MINUTE)]
    assert tp().embed("x", env=FakeEnv()) == [[1.0] * 4]
    assert [c[0] for c in CALLS] == ["gemini/gemini-embedding-2"] * 2  # waited, same model

    CALLS.clear()
    FAILURES["gemini/gemini-embedding-2"] = [ConnectionError("down")]
    tp().embed("x", env=FakeEnv())
    assert [c[0] for c in CALLS] == ["gemini/gemini-embedding-2", "openai/text-embedding-3-small"]

    FAILURES["gemini/gemini-embedding-2"] = [ConnectionError("a")]
    FAILURES["openai/text-embedding-3-small"] = [ConnectionError("b")]
    with pytest.raises(AllModelsFailed):
        tp().embed("x", env=FakeEnv())
    FAILURES["gemini/gemini-embedding-2"] = [ConnectionError("only")]
    with pytest.raises(ConnectionError):
        tp().embed("x", env=FakeEnv(), fallbacks=False)


def test_async():
    FAILURES["gemini/gemini-embedding-2"] = [ConnectionError("down")]
    assert asyncio.run(tp().aembed(["a", "bb"], env=FakeEnv())) == [[1.0] * 4, [2.0] * 4]


def test_calls_are_recorded_as_embeddings(monkeypatch, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "OBSERVABILITY": True}
    seen = []
    monkeypatch.setattr(recorder, "emit", seen.append)
    FAILURES["gemini/gemini-embedding-2"] = [ConnectionError("down")]
    tp().embed(["a"], env=FakeEnv())
    failed, ok = seen
    assert (failed.kind, failed.status, failed.model) == ("embedding", "error", "gemini/gemini-embedding-2")
    assert (ok.kind, ok.status, ok.fallback_index, ok.input_tokens) == ("embedding", "ok", 1, 7)
    assert ok.touchpoint == "t.embed" and ok.provider == "openai"


def test_langchain_embeddings_and_bad_factories():
    from langchain_core.embeddings import Embeddings

    class Fives(Embeddings):
        def embed_documents(self, texts):
            return [[0.5] * 5 for _ in texts]

        def embed_query(self, text):
            return [0.5] * 5

    wrapped = as_embedder(Fives())
    assert isinstance(wrapped, LangChainEmbedder)
    assert len(wrapped.embed(["a", "b"]).vectors[1]) == 5
    with pytest.raises(ConfigError, match="text only"):
        wrapped.embed([{"type": "image"}])
    with pytest.raises(ConfigError, match="not an Embedder"):
        as_embedder(object())


def test_litellm_backend_passes_model_inputs_key_and_reads_usage(monkeypatch):
    import types

    sent = {}

    def embedding(**kw):
        sent.update(kw)
        return {"data": [{"embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 3}, "model": "served"}

    fake = types.SimpleNamespace(embedding=embedding)
    monkeypatch.setattr("llm_kit.core.touchpoints.embeddings._litellm", lambda: fake)
    result = LiteLLMEmbedder("gemini/gemini-embedding-2", {"dimensions": 2}, ["k1"]).embed(["hi"])
    assert result == Embedded(vectors=[[0.1, 0.2]], input_tokens=3, served_model="served")
    assert sent == {"model": "gemini/gemini-embedding-2", "input": ["hi"], "dimensions": 2, "api_key": "k1"}


@pytest.mark.django_db(transaction=True)
def test_an_embedding_inside_an_agent_is_part_of_its_run(world, settings):
    from llm_kit import agents
    from llm_kit.core.agents import spec
    from llm_kit.core.agents.limits import tree_of
    from llm_kit.core.observability import flush
    from llm_kit.models import LLMCall, LLMRun
    from llm_kit.observability.setup import configure

    settings.LLM_KIT = {**settings.LLM_KIT, "OBSERVABILITY": True, "RECORD_SINKS": ["database"]}
    configure()
    spent = {}

    def index(task, ctx):
        from llm_kit.core.touchpoints.embeddings import _running

        vectors = tp().embed([task, task])
        spent["usage"] = dict(tree_of(_running()).usage)
        return len(vectors)

    agents.agent.code("t.indexer", description="for a test")(index)
    try:
        assert agents.run("t.indexer", "doc", user=world["alice"], thread_id="ix-1").text == "2"
        assert flush()
    finally:
        spec._unregister("t.indexer")
        recorder.configure([], None)
    run = LLMRun.objects.get(thread_id="ix-1")
    call = LLMCall.objects.get(kind="embedding")
    assert (call.run_id, call.user, run.calls) == (run.run_id, str(world["alice"].pk), 1)
    assert spent["usage"] == {"gemini/gemini-embedding-2": [7, 0, 0, 0, 0]}  # counts towards a budget


def test_a_touchpoints_own_backend_is_used_for_embeddings(settings):
    """A dotted path (or factory) on touchpoint(backend=) or embedding_backend= wins over EMBEDDING_BACKEND."""
    settings.LLM_KIT = {**settings.LLM_KIT, "EMBEDDING_BACKEND": "litellm"}  # would need litellm installed
    own = touchpoint("t.embed.own", default="vertex_ai/gemini-embedding-2", backend="tests.test_embeddings:backend")
    assert own.embed(["ab"], env=FakeEnv(), fallbacks=False) == [[2.0, 2.0, 2.0]]

    made = []

    def factory(ref, params, keys):
        made.append(ref)
        return Scripted(ref, params, keys)

    chat_and_embed = touchpoint(
        "t.embed.split",
        default="openai/text-embedding-3-small",
        backend="tests.graphs:fake_backend",
        embedding_backend=factory,
    )
    assert chat_and_embed.embed("x", env=FakeEnv(), fallbacks=False) == [[1.0, 1.0, 1.0]] and made

    wrong = touchpoint("t.embed.wrong", default="openai/text-embedding-3-small", backend="tests.graphs:fake_backend")
    with pytest.raises(ConfigError, match="embedding_backend="):
        wrong.embed("x", env=FakeEnv(), fallbacks=False)
