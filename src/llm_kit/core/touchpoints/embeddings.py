"""Embedding calls through a touchpoint: the same resolution, keys, fallbacks, retry waits,
circuit breaker and call records as chat calls::

    search = touchpoint(
        "search.embed",
        default="gemini/gemini-embedding-2",
        fallbacks=["openai/text-embedding-3-small"],
        params={"dimensions": 768},
        requires={"kind": "embedding"},
    )
    vectors = search.embed(["first passage", "second passage"])        # list[list[float]]
    [query] = await search.aembed("what is due?", task_type="RETRIEVAL_QUERY")

Backends (``LLM_KIT["EMBEDDING_BACKEND"]``, or a touchpoint's ``backend`` when it is
``"litellm"`` or ``"langchain"``):

* ``litellm``: ``litellm.embedding`` / ``aembedding``. Inputs are passed as given, so a
  provider that embeds images, audio or PDFs gets the parts LiteLLM accepts for it.
* ``langchain``: ``init_embeddings`` with the provider's ``langchain-*`` package (text only).
* a dotted path to ``fn(ref, params, api_keys)`` returning an ``Embedder`` (below) or a
  LangChain ``Embeddings``. Use this for multimodal inputs a built-in backend doesn't pass on.
* ``auto`` (default): ``litellm`` when installed, else ``langchain``.

``params`` (touchpoint, admin override and per call) go to the backend; ``dimensions``
is the common name for the output size (OpenAI's ``dimensions``, Gemini's
``output_dimensionality`` through LiteLLM).
"""

from __future__ import annotations

import random
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from llm_kit.core import config
from llm_kit.core.errors import AllModelsFailed, ConfigError
from llm_kit.core.graphs import import_target
from llm_kit.core.touchpoints.env import KEYLESS_PROVIDERS, provider_of
from llm_kit.core.touchpoints.retry import RetryPolicy


@dataclass
class Embedded:
    """What a backend returns: one vector per input, and the tokens billed when it says."""

    vectors: list[list[float]]
    input_tokens: int | None = None
    served_model: str = ""


@runtime_checkable
class Embedder(Protocol):
    def embed(self, inputs: list[Any], **params: Any) -> Embedded: ...

    async def aembed(self, inputs: list[Any], **params: Any) -> Embedded: ...


EmbedderFactory = Callable[[str, dict[str, Any], list[str]], Any]


# ------------------------------------------------------------------ backends


class LiteLLMEmbedder:
    def __init__(self, ref: str, params: dict[str, Any], keys: list[str]) -> None:
        self.ref, self.params, self.keys = ref, dict(params), list(keys)

    def _kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        kwargs = {**self.params, **params}
        if self.keys:
            kwargs.setdefault("api_key", random.choice(self.keys))
        return kwargs

    @staticmethod
    def _result(response: Any) -> Embedded:
        data = response["data"] if isinstance(response, dict) else response.data
        vectors = [list(item["embedding"] if isinstance(item, dict) else item.embedding) for item in data]
        usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
        tokens = None
        if usage is not None:
            tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else getattr(usage, "prompt_tokens", None)
        served = (response.get("model") if isinstance(response, dict) else getattr(response, "model", "")) or ""
        return Embedded(vectors=vectors, input_tokens=tokens, served_model=str(served))

    def embed(self, inputs: list[Any], **params: Any) -> Embedded:
        litellm = _litellm()
        return self._result(litellm.embedding(model=self.ref, input=inputs, **self._kwargs(params)))

    async def aembed(self, inputs: list[Any], **params: Any) -> Embedded:
        litellm = _litellm()
        return self._result(await litellm.aembedding(model=self.ref, input=inputs, **self._kwargs(params)))


def _litellm():
    from llm_kit.core.touchpoints.backends import _quiet_litellm_import

    _quiet_litellm_import()
    try:
        import litellm
    except ImportError as e:
        raise ConfigError("The litellm embedding backend needs: pip install django-llm-kit[litellm]") from e
    return litellm


class LangChainEmbedder:
    """Wraps a LangChain ``Embeddings`` (text only; it reports no token counts)."""

    def __init__(self, embeddings: Any) -> None:
        self.embeddings = embeddings

    @staticmethod
    def _texts(inputs: list[Any]) -> list[str]:
        if not all(isinstance(i, str) for i in inputs):
            raise ConfigError("This embedding backend takes text only; use a backend that accepts media parts.")
        return list(inputs)

    def embed(self, inputs: list[Any], **params: Any) -> Embedded:
        return Embedded(vectors=self.embeddings.embed_documents(self._texts(inputs), **params))

    async def aembed(self, inputs: list[Any], **params: Any) -> Embedded:
        return Embedded(vectors=await self.embeddings.aembed_documents(self._texts(inputs), **params))


def _langchain_embedder(ref: str, params: dict[str, Any], keys: list[str]) -> LangChainEmbedder:
    from langchain.embeddings import init_embeddings

    from llm_kit.core.touchpoints.backends import LANGCHAIN_PROVIDERS

    provider, _, model_id = ref.partition("/")
    kwargs = dict(params)
    if keys and provider not in KEYLESS_PROVIDERS:
        kwargs.setdefault("api_key", random.choice(keys))
    return LangChainEmbedder(init_embeddings(model_id, provider=LANGCHAIN_PROVIDERS.get(provider, provider), **kwargs))


_factories: dict[str, EmbedderFactory] = {}
_lock = threading.Lock()


def get_embedder_factory(name: str | None) -> EmbedderFactory:
    name = name or config.get("EMBEDDING_BACKEND") or "auto"
    if name == "auto":
        from llm_kit.core.touchpoints.backends import _litellm_installed

        name = "litellm" if _litellm_installed() else "langchain"
    if name == "litellm":
        return LiteLLMEmbedder
    if name == "langchain":
        return _langchain_embedder
    with _lock:
        if name not in _factories:
            factory = import_target(name)
            if not callable(factory):
                raise ConfigError(f"EMBEDDING_BACKEND {name!r} is not callable.")
            _factories[name] = factory
    return _factories[name]


def as_embedder(made: Any) -> Embedder:
    if isinstance(made, Embedder):
        return made
    if hasattr(made, "embed_documents"):
        return LangChainEmbedder(made)
    raise ConfigError(f"An embedding backend returned {type(made).__name__}, not an Embedder or LangChain Embeddings.")


# ------------------------------------------------------------------ calling


def _inputs(inputs: Any) -> list[Any]:
    if isinstance(inputs, (str, bytes, dict)):
        return [inputs]
    if isinstance(inputs, Sequence):
        return list(inputs)
    return [inputs]


def _text_of(inputs: list[Any]) -> str:
    """The text in the inputs, to estimate tokens when the provider reports none."""
    out = []
    for item in inputs:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            out.append(item["text"])
    return "\n".join(out)


@dataclass
class _Call:
    tp: Any
    refs: list[str]
    embedders: list[Embedder]
    retry: RetryPolicy
    resolution: Any
    params: dict[str, Any]
    owners: list[str] = None  # whose keys each attempt uses


def prepare(tp, inputs, *, requested, config_, env, retry, fallbacks, params) -> tuple[_Call, list[Any]]:
    from llm_kit.core.observability.spend import check_daily_budget
    from llm_kit.core.touchpoints.env import default_env
    from llm_kit.core.touchpoints.retry import policy_for

    check_daily_budget(tp)
    env = tp.env_for(config_, env or default_env())
    if (check_budgets := getattr(env, "check_budgets", None)) is not None:
        check_budgets(tp)
    resolution = tp.resolve(requested, config=config_, env=env)
    if not fallbacks:
        resolution.fallbacks = []
    backend = tp.backend if tp.backend in ("litellm", "langchain", "auto") else None
    factory = get_embedder_factory(backend)
    merged = {**resolution.params, **params}
    from llm_kit.core.touchpoints.backends import key_sets

    refs, owners, embedders = [], [], []
    for ref in resolution.candidates:
        for owner, keys in key_sets(env, ref, tp):  # BYOK: the caller's keys, then the platform's
            refs.append(ref)
            owners.append(owner)
            embedders.append(as_embedder(factory(ref, dict(merged), keys)))
    if not embedders:
        from llm_kit.core.touchpoints.backends import no_key

        raise no_key(tp, resolution.candidates)
    run = config_ if config_ is not None else _running()
    call = _Call(tp, refs, embedders, policy_for(tp, run, retry), resolution, merged, owners)
    return call, _inputs(inputs)


def _running():
    from llm_kit.core.agents.limits import running_config

    return running_config()


def run_sync(call: _Call, inputs: list[Any], extra: dict[str, Any]) -> list[list[float]]:
    from llm_kit.core.touchpoints import backends

    errors: list[tuple[str, BaseException]] = []
    for index, (ref, embedder) in enumerate(zip(call.refs, call.embedders, strict=True)):
        attempt = 0
        while True:
            rec, t0 = _start(call, ref, index, inputs, call.owners[index] if call.owners else "platform")
            try:
                result = embedder.embed(inputs, **extra)
            except Exception as e:
                _end(rec, t0, inputs, error=e)
                delay = call.retry.delay(e, attempt)
                if delay is None:
                    errors.append((ref, e))
                    break
                attempt += 1
                backends._sleep(delay)
                continue
            _end(rec, t0, inputs, result=result)
            return result.vectors
    raise _failure(call, errors)


async def run_async(call: _Call, inputs: list[Any], extra: dict[str, Any]) -> list[list[float]]:
    from llm_kit.core.touchpoints import backends

    errors: list[tuple[str, BaseException]] = []
    for index, (ref, embedder) in enumerate(zip(call.refs, call.embedders, strict=True)):
        attempt = 0
        while True:
            rec, t0 = _start(call, ref, index, inputs, call.owners[index] if call.owners else "platform")
            try:
                result = await embedder.aembed(inputs, **extra)
            except Exception as e:
                _end(rec, t0, inputs, error=e)
                delay = call.retry.delay(e, attempt)
                if delay is None:
                    errors.append((ref, e))
                    break
                attempt += 1
                await backends._asleep(delay)
                continue
            _end(rec, t0, inputs, result=result)
            return result.vectors
    raise _failure(call, errors)


def _failure(call: _Call, errors: list[tuple[str, BaseException]]) -> BaseException:
    if len(errors) == 1:
        return errors[0][1]
    failed = AllModelsFailed(call.tp.name, errors)
    failed.__cause__ = errors[-1][1]
    return failed


# ------------------------------------------------------------------ recording


def _start(call: _Call, ref: str, index: int, inputs: list[Any], owner: str = "platform"):
    from llm_kit.core.observability.handler import get_handler

    if get_handler() is None:
        return None, time.perf_counter()
    from llm_kit.core.catalogue.fields import normalise_provider
    from llm_kit.core.observability.context import current
    from llm_kit.core.observability.handler import _hash
    from llm_kit.core.observability.records import CallRecord

    parent = _running() or {}
    meta, ctx = parent.get("metadata") or {}, current()
    rec = CallRecord(
        call_id=str(uuid.uuid4()),
        run_id=_root_run(parent),
        started_at=datetime.now(UTC),
        kind="embedding",
        key_owner=owner,
        touchpoint=call.tp.name,
        graph=str(meta.get("llm_kit_graph", "")),
        agent=str(meta.get("llm_kit_agent") or ""),
        thread_id=str(meta.get("llm_kit_thread") or meta.get("thread_id") or ""),
        user=str(meta.get("user_id") or ctx.user or ""),
        tenant=str(meta.get("tenant") or ctx.tenant or ""),
        tags=dict(ctx.tags),
        requested_model=str(call.resolution.requested or ""),
        resolved_model=call.resolution.model,
        model=ref,
        provider=normalise_provider(provider_of(ref)),
        chosen_by=call.resolution.chosen_by,
        fallback_index=index,
        prompt_hash=_hash([i if isinstance(i, str) else repr(i)[:200] for i in inputs]),
    )
    return rec, time.perf_counter()


def _root_run(parent: dict) -> str | None:
    """The request run an embedding made inside a running agent belongs to."""
    from llm_kit.core.observability.handler import handler

    manager = parent.get("callbacks")
    run_id = getattr(manager, "parent_run_id", None)
    if run_id is None:
        return None
    return str(handler._root.get(run_id, run_id))


def _end(rec, t0: float, inputs: list[Any], *, result: Embedded | None = None, error: BaseException | None = None):
    if rec is None:
        return
    from llm_kit.core.observability.cost import estimate_tokens
    from llm_kit.core.observability.handler import _status_for, handler
    from llm_kit.core.observability.recorder import recorder

    rec.latency_ms = int((time.perf_counter() - t0) * 1000)
    if error is not None:
        rec.status, rec.error_class = _status_for(error)
        rec.error_message = f"{type(error).__name__}: {error}"[:500]
    elif result is not None:
        if result.input_tokens is not None:
            rec.input_tokens = int(result.input_tokens)
        else:
            rec.input_tokens = estimate_tokens(_text_of(inputs))
            rec.usage_estimated = True
        rec.served_model = result.served_model
        handler.count_call(rec)
        tree = _tree()
        if tree is not None:  # the request's budget
            tree.add_usage(rec.model, (rec.input_tokens, 0, 0, 0, 0))
    recorder.emit(rec)


def _tree():
    from llm_kit.core.agents.limits import tree_of

    return tree_of(_running())
