"""Build LangChain chat models for resolved touchpoints.

Backends (``LLM_KIT["MODEL_BACKEND"]`` or ``touchpoint(..., backend=)``):

* ``litellm``: ``ChatLiteLLM``; with several API keys for a provider, a
  ``ChatLiteLLMRouter`` that spreads calls across them. Needs ``langchain-litellm``.
* ``langchain``: ``init_chat_model`` with the provider's ``langchain-*`` package.
* a dotted path to ``fn(ref, params, api_keys) -> BaseChatModel``.
* ``auto`` (default): ``litellm`` when installed, else ``langchain``.

Every model carries the touchpoint in its ``metadata``/``tags``, which tracing
and llm-kit's call recording read.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import random
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig, RunnableWithFallbacks
from langchain_core.runnables.config import (
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
    set_config_context,
)
from langchain_core.runnables.utils import coro_with_context

from llm_kit.core import config
from llm_kit.core.errors import AllModelsFailed, ConfigError, NoKey
from llm_kit.core.graphs import import_target
from llm_kit.core.touchpoints.env import KEYLESS_PROVIDERS, ResolutionEnv, provider_of
from llm_kit.core.touchpoints.resolve import Resolution
from llm_kit.core.touchpoints.retry import RetryPolicy

logger = logging.getLogger("llm_kit.touchpoints")

Factory = Callable[[str, dict[str, Any], list[str]], BaseChatModel]

# LiteLLM provider name -> LangChain init_chat_model provider name.
LANGCHAIN_PROVIDERS = {
    "gemini": "google_genai",
    "vertex_ai": "google_vertexai",
    "bedrock": "bedrock_converse",
    "together_ai": "together",
    "fireworks_ai": "fireworks",
    "mistral": "mistralai",
    "nvidia_nim": "nvidia",
    "azure": "azure_openai",
    "huggingface": "huggingface",
}


class TouchpointModel(RunnableBinding):
    """A factory's chat model with the touchpoint's metadata, tags and recording handler added as
    call config. The model itself is never changed, so a factory may build a model once and
    return it for every call (keeping its HTTP client).

    ``bind_tools`` and ``with_structured_output`` keep that config (a plain ``RunnableBinding``
    would drop it), and a callback passed twice (the binding's and the caller's) runs once."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> TouchpointModel:
        return self._rebind(self.bound.bind_tools(tools, **kwargs))

    def with_structured_output(self, schema: Any, **kwargs: Any) -> TouchpointModel:
        return self._rebind(self.bound.with_structured_output(schema, **kwargs))

    def _rebind(self, runnable: Runnable) -> TouchpointModel:
        return TouchpointModel(bound=runnable, config=self.config, kwargs={})

    def _merge_configs(self, *configs: RunnableConfig | None) -> RunnableConfig:
        config = super()._merge_configs(*configs)
        callbacks = config.get("callbacks")
        if isinstance(callbacks, list):
            unique: list[Any] = []
            for cb in callbacks:
                if not any(cb is seen for seen in unique):
                    unique.append(cb)
            config["callbacks"] = unique
        return config


def wrap(model: Runnable, metadata: dict[str, Any], tags: Sequence[str], callbacks: Sequence[Any]) -> TouchpointModel:
    """``model`` with call config added, leaving the model itself as it is."""
    config: dict[str, Any] = {"metadata": dict(metadata), "tags": list(tags)}
    if callbacks:
        config["callbacks"] = list(callbacks)
    return TouchpointModel(bound=model, config=config, kwargs={})


class TouchpointChatModel(RunnableWithFallbacks):
    """A primary chat model with fallbacks, waiting out short rate limits on each.

    Each candidate is tried in turn. A rate limit or overload that the touchpoint's
    ``RetryPolicy`` allows is waited out and the same model tried again; anything else
    moves to the next candidate. When every candidate fails, ``AllModelsFailed`` lists
    each model's error (a single candidate re-raises its own error).

    ``RunnableWithFallbacks`` forwards ``bind_tools`` only when the method's return
    annotation resolves at runtime; otherwise fallbacks are dropped or an error is raised.
    These explicit methods always apply to every candidate.
    """

    retry: Any = None  # RetryPolicy
    refs: list[str] = []  # the model behind each candidate, for errors
    touchpoint: str = ""
    limiter: Any = None  # ratelimit.Limiter: a slot per call, for candidates on the platform's keys
    owners: list[str] = []  # whose keys each candidate uses (BYOK keys aren't your account's limits)
    adapt: Any = None  # fn(input, ref, info) -> input, run for each candidate before it is called
    infos: list[Any] = []  # each candidate's catalogue entry (ModelInfo) or None, for ``adapt``

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> TouchpointChatModel:
        return self._map(lambda m: m.bind_tools(tools, **kwargs))

    def with_structured_output(self, schema: Any, **kwargs: Any) -> TouchpointChatModel:
        return self._map(lambda m: m.with_structured_output(schema, **kwargs))

    def _map(self, fn: Callable[[Any], Runnable]) -> TouchpointChatModel:
        return TouchpointChatModel(
            runnable=fn(self.runnable),
            fallbacks=[fn(f) for f in self.fallbacks],
            exceptions_to_handle=self.exceptions_to_handle,
            retry=self.retry,
            refs=self.refs,
            touchpoint=self.touchpoint,
            limiter=self.limiter,
            owners=self.owners,
            adapt=self.adapt,
            infos=self.infos,
        )

    @property
    def models(self) -> list[Runnable]:
        return [self.runnable, *self.fallbacks]

    # -------------------------------------------------------------- calling

    def _ref(self, index: int) -> str:
        return self.refs[index] if index < len(self.refs) else f"candidate {index}"

    def _wait_for(self, exc: BaseException, attempt: int, index: int) -> float | None:
        policy = self.retry or RetryPolicy.NEVER
        delay = policy.delay(exc, attempt)
        if delay is not None:
            logger.info(
                "%s: %s refused (%s); waiting %.1fs before trying it again (%d of %d)",
                self.touchpoint or "touchpoint",
                self._ref(index),
                type(exc).__name__,
                delay,
                attempt + 1,
                policy.attempts,
            )
        return delay

    # -------------------------------------------------------------- input per candidate

    def _info(self, index: int) -> Any:
        return self.infos[index] if index < len(self.infos) else None

    def _adapt_failed(self, index: int, e: BaseException) -> BaseException:
        logger.warning(
            "%s: adapt_input failed for %s (%s: %s); trying the next model",
            self.touchpoint or "touchpoint",
            self._ref(index),
            type(e).__name__,
            e,
        )
        return e

    def _adapted(self, index: int, input: Any) -> Any:
        """``input`` as candidate ``index`` should get it (sync calls)."""
        if self.adapt is None:
            return input
        out = self.adapt(input, self._ref(index), self._info(index))
        if inspect.isawaitable(out):  # an async adapter, from a sync call
            from asgiref.sync import async_to_sync

            out = async_to_sync(_awaited)(out)
        return out

    async def _aadapted(self, index: int, input: Any) -> Any:
        if self.adapt is None:
            return input
        out = self.adapt(input, self._ref(index), self._info(index))
        return await out if inspect.isawaitable(out) else out

    # -------------------------------------------------------------- rate limits

    def _limited(self, index: int) -> bool:
        if self.limiter is None:
            return False
        owner = self.owners[index] if index < len(self.owners) else "platform"
        return owner in ("", "platform")

    def _limit_wait(self) -> float:
        policy = self.retry or RetryPolicy.NEVER
        return policy.max_wait if policy.waits else 0.0

    def _refused(self, index: int, e: BaseException) -> BaseException:
        logger.info(
            "%s: no slot for %s (%s); trying the next model", self.touchpoint or "touchpoint", self._ref(index), e
        )
        return e

    def _slot(self, index: int) -> BaseException | None:
        """None when the call may go ahead; the ``RateLimited`` refusal otherwise."""
        if not self._limited(index):
            return None
        from llm_kit.core.touchpoints.ratelimit import RateLimited

        try:
            self.limiter.acquire(self._ref(index), self._limit_wait())
        except RateLimited as e:
            return self._refused(index, e)
        return None

    async def _aslot(self, index: int) -> BaseException | None:
        if not self._limited(index):
            return None
        from llm_kit.core.touchpoints.ratelimit import RateLimited

        try:
            await self.limiter.aacquire(self._ref(index), self._limit_wait())
        except RateLimited as e:
            return self._refused(index, e)
        return None

    def _count(self, index: int, output: Any) -> None:
        if not self._limited(index) or output is None:
            return
        from llm_kit.core.touchpoints.ratelimit import tokens_of

        try:
            self.limiter.record(self._ref(index), tokens_of(output))
        except Exception:  # counting must never break an answer
            logger.warning("Could not count tokens for %s", self._ref(index), exc_info=True)

    def _failure(self, errors: list[tuple[str, BaseException]]) -> BaseException:
        if len(errors) == 1:
            return errors[0][1]
        failed = AllModelsFailed(self.touchpoint, errors)
        failed.__cause__ = errors[-1][1]
        return failed

    def _start(self, input: Any, config: RunnableConfig | None, callback_manager):
        return callback_manager.on_chain_start(
            None, input, name=config.get("run_name") or self.get_name(), run_id=config.pop("run_id", None)
        )

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        config = ensure_config(config)
        run_manager = self._start(input, config, get_callback_manager_for_config(config))
        errors: list[tuple[str, BaseException]] = []
        for index, runnable in enumerate(self.runnables):
            try:
                candidate_input = self._adapted(index, input)
            except Exception as e:
                errors.append((self._ref(index), self._adapt_failed(index, e)))
                continue
            attempt = 0
            while True:
                if (refused := self._slot(index)) is not None:
                    errors.append((self._ref(index), refused))
                    break
                try:
                    child = patch_config(config, callbacks=run_manager.get_child())
                    with set_config_context(child) as context:
                        output = context.run(runnable.invoke, candidate_input, child, **kwargs)
                except self.exceptions_to_handle as e:
                    delay = self._wait_for(e, attempt, index)
                    if delay is None:
                        errors.append((self._ref(index), e))
                        break
                    attempt += 1
                    _sleep(delay)
                except BaseException as e:
                    run_manager.on_chain_error(e)
                    raise
                else:
                    self._count(index, output)
                    run_manager.on_chain_end(output)
                    return output
        failure = self._failure(errors)
        run_manager.on_chain_error(failure)
        raise failure

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        config = ensure_config(config)
        run_manager = await self._start(input, config, get_async_callback_manager_for_config(config))
        errors: list[tuple[str, BaseException]] = []
        for index, runnable in enumerate(self.runnables):
            try:
                candidate_input = await self._aadapted(index, input)
            except Exception as e:
                errors.append((self._ref(index), self._adapt_failed(index, e)))
                continue
            attempt = 0
            while True:
                if (refused := await self._aslot(index)) is not None:
                    errors.append((self._ref(index), refused))
                    break
                try:
                    child = patch_config(config, callbacks=run_manager.get_child())
                    with set_config_context(child) as context:
                        coro = context.run(runnable.ainvoke, candidate_input, child, **kwargs)
                        output = await coro_with_context(coro, context)
                except self.exceptions_to_handle as e:
                    delay = self._wait_for(e, attempt, index)
                    if delay is None:
                        errors.append((self._ref(index), e))
                        break
                    attempt += 1
                    await _asleep(delay)
                except BaseException as e:
                    await run_manager.on_chain_error(e)
                    raise
                else:
                    self._count(index, output)
                    await run_manager.on_chain_end(output)
                    return output
        failure = self._failure(errors)
        await run_manager.on_chain_error(failure)
        raise failure

    # A failure after the first chunk can't move to another model (the caller already has
    # part of the answer), so streams wait and fall back only before the first chunk.

    def stream(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Iterator[Any]:
        config = ensure_config(config)
        run_manager = self._start(input, config, get_callback_manager_for_config(config))
        errors: list[tuple[str, BaseException]] = []
        stream = None
        for index, runnable in enumerate(self.runnables):
            try:
                candidate_input = self._adapted(index, input)
            except Exception as e:
                errors.append((self._ref(index), self._adapt_failed(index, e)))
                continue
            attempt = 0
            while stream is None:
                if (refused := self._slot(index)) is not None:
                    errors.append((self._ref(index), refused))
                    break
                try:
                    child = patch_config(config, callbacks=run_manager.get_child())
                    with set_config_context(child) as context:
                        candidate = context.run(runnable.stream, candidate_input, child, **kwargs)
                        chunk = context.run(next, candidate)
                    stream, streamed = candidate, index
                except self.exceptions_to_handle as e:
                    delay = self._wait_for(e, attempt, index)
                    if delay is None:
                        errors.append((self._ref(index), e))
                        break
                    attempt += 1
                    _sleep(delay)
                except BaseException as e:
                    run_manager.on_chain_error(e)
                    raise
            if stream is not None:
                break
        if stream is None:
            failure = self._failure(errors)
            run_manager.on_chain_error(failure)
            raise failure
        yield chunk
        output = chunk
        try:
            for chunk in stream:
                yield chunk
                try:
                    output = output + chunk
                except TypeError:
                    output = None
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        self._count(streamed, output)
        run_manager.on_chain_end(output)

    async def astream(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> AsyncIterator[Any]:
        config = ensure_config(config)
        run_manager = await self._start(input, config, get_async_callback_manager_for_config(config))
        errors: list[tuple[str, BaseException]] = []
        stream = None
        for index, runnable in enumerate(self.runnables):
            try:
                candidate_input = await self._aadapted(index, input)
            except Exception as e:
                errors.append((self._ref(index), self._adapt_failed(index, e)))
                continue
            attempt = 0
            while stream is None:
                if (refused := await self._aslot(index)) is not None:
                    errors.append((self._ref(index), refused))
                    break
                try:
                    child = patch_config(config, callbacks=run_manager.get_child())
                    with set_config_context(child) as context:
                        candidate = runnable.astream(candidate_input, child, **kwargs)
                        chunk = await coro_with_context(anext(candidate), context)
                    stream, streamed = candidate, index
                except self.exceptions_to_handle as e:
                    delay = self._wait_for(e, attempt, index)
                    if delay is None:
                        errors.append((self._ref(index), e))
                        break
                    attempt += 1
                    await _asleep(delay)
                except BaseException as e:
                    await run_manager.on_chain_error(e)
                    raise
            if stream is not None:
                break
        if stream is None:
            failure = self._failure(errors)
            await run_manager.on_chain_error(failure)
            raise failure
        yield chunk
        output = chunk
        try:
            async for chunk in stream:
                yield chunk
                try:
                    output = output + chunk
                except TypeError:
                    output = None
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        self._count(streamed, output)
        await run_manager.on_chain_end(output)

    # One call per input through the loop above, instead of RunnableWithFallbacks' batching.
    batch = Runnable.batch
    abatch = Runnable.abatch


async def _awaited(awaitable):
    return await awaitable


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


async def _asleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


# ------------------------------------------------------------------ backends


def _quiet_litellm_import() -> None:
    """``import litellm`` downloads its price map from GitHub unless told not to (35 s on a
    host with broken IPv6). llm-kit's catalogue already syncs that map on a schedule, so use
    the copy bundled with litellm. Respects an explicit ``LITELLM_LOCAL_MODEL_COST_MAP``."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _litellm_model(ref: str, params: dict[str, Any], keys: list[str]) -> BaseChatModel:
    _quiet_litellm_import()
    try:
        from langchain_litellm import ChatLiteLLM, ChatLiteLLMRouter
    except ImportError as e:
        raise ConfigError("The litellm backend needs: pip install django-llm-kit[litellm]") from e
    fields = set(ChatLiteLLM.model_fields)
    direct = {k: v for k, v in params.items() if k in fields}
    extra = {k: v for k, v in params.items() if k not in fields}
    if len(keys) > 1:
        router = _router(ref, tuple(keys), json.dumps(config.get("LITELLM_ROUTER"), sort_keys=True))
        return ChatLiteLLMRouter(router=router, model_name=ref, model=ref, model_kwargs=extra, **direct)
    return ChatLiteLLM(model=ref, api_key=keys[0] if keys else None, model_kwargs=extra, **direct)


@functools.lru_cache(maxsize=64)
def _router(ref: str, keys: tuple[str, ...], router_kwargs_json: str):
    _quiet_litellm_import()
    from litellm import Router

    model_list = [{"model_name": ref, "litellm_params": {"model": ref, "api_key": k}} for k in keys]
    return Router(model_list=model_list, **json.loads(router_kwargs_json))


def _langchain_model(ref: str, params: dict[str, Any], keys: list[str]) -> BaseChatModel:
    from langchain.chat_models import init_chat_model

    provider, _, model_id = ref.partition("/")
    kwargs = dict(params)
    if keys and provider not in KEYLESS_PROVIDERS:
        kwargs.setdefault("api_key", random.choice(keys))  # naive spread across a key pool
    return init_chat_model(model_id, model_provider=LANGCHAIN_PROVIDERS.get(provider, provider), **kwargs)


def _litellm_installed() -> bool:
    try:
        import langchain_litellm  # noqa: F401
    except ImportError:
        return False
    return True


_factories: dict[str, Factory] = {}
_factory_lock = threading.Lock()


def get_factory(name: str | None) -> Factory:
    name = name or config.get("MODEL_BACKEND") or "auto"
    if name == "auto":
        name = "litellm" if _litellm_installed() else "langchain"
    if name == "litellm":
        return _litellm_model
    if name == "langchain":
        return _langchain_model
    with _factory_lock:
        if name not in _factories:
            factory = import_target(name)
            if not callable(factory):
                raise ConfigError(f"MODEL_BACKEND {name!r} is not callable.")
            _factories[name] = factory
    return _factories[name]


def build_for(
    tp, resolution: Resolution, env: ResolutionEnv, retry: RetryPolicy | None = None, adapt_input: Any = None
) -> Runnable:
    """The runnable for a resolution: the chat model itself when there is one candidate and
    nothing to wait for, otherwise a ``TouchpointChatModel`` (fallbacks and/or retry waits)."""
    factory = get_factory(tp.backend)
    metadata = {
        "llm_kit_touchpoint": tp.name,
        "llm_kit_chosen_by": resolution.chosen_by,
        "llm_kit_requested": resolution.requested,
        "llm_kit_resolved": resolution.model,
    }
    if tp.capture is not None:
        metadata["llm_kit_capture"] = tp.capture
    from llm_kit.core.observability.handler import get_handler

    handler = get_handler()
    limiter_for = getattr(env, "rate_limiter", None)
    limiter = limiter_for() if callable(limiter_for) else None
    adapt = load_adapter(tp.adapt_input if adapt_input is None else adapt_input)
    models, refs, owners, infos = [], [], [], []
    for index, ref in enumerate(resolution.candidates):
        for owner, keys in key_sets(env, ref, tp):  # one attempt per owner (BYOK: theirs, then yours)
            built = factory(ref, tp.params_for(ref, resolution.params), keys)
            # Added as call config, never set on the factory's model: a factory may return one
            # shared model for every call (its client lives there), and must find it unchanged.
            model = wrap(
                built,
                {**metadata, "llm_kit_candidate": index, "llm_kit_model": ref, "llm_kit_key_owner": owner},
                [f"llm_kit:{tp.name}"],
                [handler] if handler is not None else [],
            )
            models.append(model)
            refs.append(ref)
            owners.append(owner)
            infos.append(_lookup(env, ref) if adapt is not None else None)
    if not models:
        raise no_key(tp, resolution.candidates)
    if len(models) == 1 and not (retry and retry.waits) and limiter is None and adapt is None:
        return models[0]
    return TouchpointChatModel(
        runnable=models[0],
        fallbacks=models[1:],
        retry=retry or RetryPolicy.NEVER,
        refs=refs,
        touchpoint=tp.name,
        limiter=limiter,
        owners=owners,
        adapt=adapt,
        infos=infos,
    )


def load_adapter(value: Any) -> Callable[..., Any] | None:
    """``adapt_input``: a callable or a dotted path to one; ``None``/``False`` for none."""
    if value is None or value is False:
        return None
    if isinstance(value, str):
        value = import_target(value)
    if not callable(value):
        raise ConfigError(f"adapt_input must be fn(input, ref, info) or a dotted path to one, not {value!r}.")
    return value


def _lookup(env: ResolutionEnv, ref: str) -> Any:
    try:
        return env.lookup(ref)
    except Exception:
        return None


def key_sets(env: ResolutionEnv, ref: str, tp=None) -> list[tuple[str, list[str]]]:
    """``[(owner, keys), ...]`` to try for ``ref``, in order: an env with ``key_sets`` (bring your
    own key) may give the caller's keys before, or instead of, the platform's."""
    provider = provider_of(ref)
    sets = getattr(env, "key_sets", None)
    if sets is not None:
        return sets(provider, tp)  # may be empty: the caller must bring a key and has none
    return [("platform", env.api_keys(provider))]


def no_key(tp, refs: list[str]) -> ConfigError:
    providers = sorted({provider_of(r) for r in refs})
    return NoKey(
        f"No API key for {', '.join(providers)}: {getattr(tp, 'name', 'this touchpoint')} uses only "
        "keys its tenant or user brought, and they have none for it."
    )
