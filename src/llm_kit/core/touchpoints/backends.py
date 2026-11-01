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

import functools
import json
import os
import random
import threading
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableWithFallbacks

from llm_kit.core import config
from llm_kit.core.errors import ConfigError
from llm_kit.core.graphs import import_target
from llm_kit.core.touchpoints.env import KEYLESS_PROVIDERS, ResolutionEnv, provider_of
from llm_kit.core.touchpoints.resolve import Resolution

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
    """A primary chat model with fallbacks.

    ``RunnableWithFallbacks`` forwards ``bind_tools`` only when the method's
    return annotation resolves at runtime; otherwise fallbacks are dropped or an
    error is raised. These explicit methods always apply to every candidate.
    """

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> TouchpointChatModel:
        return self._map(lambda m: m.bind_tools(tools, **kwargs))

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableWithFallbacks:
        return RunnableWithFallbacks(
            runnable=self.runnable.with_structured_output(schema, **kwargs),
            fallbacks=[f.with_structured_output(schema, **kwargs) for f in self.fallbacks],
            exceptions_to_handle=self.exceptions_to_handle,
        )

    def _map(self, fn: Callable[[Any], Runnable]) -> TouchpointChatModel:
        return TouchpointChatModel(
            runnable=fn(self.runnable),
            fallbacks=[fn(f) for f in self.fallbacks],
            exceptions_to_handle=self.exceptions_to_handle,
        )

    @property
    def models(self) -> list[Runnable]:
        return [self.runnable, *self.fallbacks]


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


def build_for(tp, resolution: Resolution, env: ResolutionEnv) -> Runnable:
    """The runnable for a resolution: one chat model, or a ``TouchpointChatModel`` with fallbacks."""
    factory = get_factory(tp.backend)
    metadata = {
        "llm_kit_touchpoint": tp.name,
        "llm_kit_chosen_by": resolution.chosen_by,
        "llm_kit_requested": resolution.requested,
        "llm_kit_resolved": resolution.model,
    }
    models = []
    for index, ref in enumerate(resolution.candidates):
        model = factory(ref, dict(resolution.params), env.api_keys(provider_of(ref)))
        model.metadata = {**(model.metadata or {}), **metadata, "llm_kit_candidate": index, "llm_kit_model": ref}
        model.tags = [*(model.tags or []), f"llm_kit:{tp.name}"]
        models.append(model)
    if len(models) == 1:
        return models[0]
    return TouchpointChatModel(runnable=models[0], fallbacks=models[1:])
