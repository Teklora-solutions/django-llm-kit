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
import json
import logging
import os
import random
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig, RunnableWithFallbacks
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
            attempt = 0
            while True:
                try:
                    child = patch_config(config, callbacks=run_manager.get_child())
                    with set_config_context(child) as context:
                        output = context.run(runnable.invoke, input, child, **kwargs)
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
            attempt = 0
            while True:
                try:
                    child = patch_config(config, callbacks=run_manager.get_child())
                    with set_config_context(child) as context:
                        coro = context.run(runnable.ainvoke, input, child, **kwargs)
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
            attempt = 0
            while stream is None:
                try:
                    child = patch_config(config, callbacks=run_manager.get_child())
                    with set_config_context(child) as context:
                        candidate = context.run(runnable.stream, input, child, **kwargs)
                        chunk = context.run(next, candidate)
                    stream = candidate
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
        run_manager.on_chain_end(output)

    async def astream(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> AsyncIterator[Any]:
        config = ensure_config(config)
        run_manager = await self._start(input, config, get_async_callback_manager_for_config(config))
        errors: list[tuple[str, BaseException]] = []
        stream = None
        for index, runnable in enumerate(self.runnables):
            attempt = 0
            while stream is None:
                try:
                    child = patch_config(config, callbacks=run_manager.get_child())
                    with set_config_context(child) as context:
                        candidate = runnable.astream(input, child, **kwargs)
                        chunk = await coro_with_context(anext(candidate), context)
                    stream = candidate
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
        await run_manager.on_chain_end(output)

    # One call per input through the loop above, instead of RunnableWithFallbacks' batching.
    batch = Runnable.batch
    abatch = Runnable.abatch


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


def build_for(tp, resolution: Resolution, env: ResolutionEnv, retry: RetryPolicy | None = None) -> Runnable:
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
    models, refs = [], []
    for index, ref in enumerate(resolution.candidates):
        for owner, keys in key_sets(env, ref, tp):  # one attempt per owner (BYOK: theirs, then yours)
            model = factory(ref, dict(resolution.params), keys)
            model.metadata = {
                **(model.metadata or {}),
                **metadata,
                "llm_kit_candidate": index,
                "llm_kit_model": ref,
                "llm_kit_key_owner": owner,
            }
            model.tags = [*(model.tags or []), f"llm_kit:{tp.name}"]
            if handler is not None:
                existing = model.callbacks if isinstance(model.callbacks, list) else []
                model.callbacks = [*existing, handler]
            models.append(model)
            refs.append(ref)
    if not models:
        raise no_key(tp, resolution.candidates)
    if len(models) == 1 and not (retry and retry.waits):
        return models[0]
    return TouchpointChatModel(
        runnable=models[0],
        fallbacks=models[1:],
        retry=retry or RetryPolicy.NEVER,
        refs=refs,
        touchpoint=tp.name,
    )


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
