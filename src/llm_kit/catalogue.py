"""Model catalogue: resolve a name to a LangChain chat model.

Lookup order for ``get_chat_model(name)``:

1. An active ``LLMModel`` row with that name (editable in the admin).
2. An alias in ``LLM_KIT["MODELS"]``.
3. The name itself as an ``init_chat_model`` id, e.g. ``"openai:gpt-5"``.

With no name, the catalogue default (``is_default=True``) is used, then
``LLM_KIT["DEFAULT_MODEL"]``.

Inside a graph node, pick the model chosen for the run with::

    def call_model(state, config):
        llm = model_from_config(config)   # reads config["configurable"]["model"]
        return {"messages": [llm.invoke(state["messages"])]}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from django.core.exceptions import ImproperlyConfigured
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from llm_kit import conf


class UnknownModel(LookupError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: str
    provider: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    api_key_env: str = ""
    description: str = ""
    source: str = "settings"  # "db" | "settings" | "raw"

    def kwargs(self) -> dict[str, Any]:
        kw = dict(self.params)
        if self.provider:
            kw["model_provider"] = self.provider
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ImproperlyConfigured(
                    f"Model {self.name!r} needs environment variable {self.api_key_env!r}, which is not set."
                )
            kw["api_key"] = key
        return kw

    def build(self, **overrides: Any) -> BaseChatModel:
        return init_chat_model(self.model, **{**self.kwargs(), **overrides})


def _from_settings(name: str) -> ModelSpec | None:
    entry = conf.get("MODELS").get(name)
    if entry is None:
        return None
    if isinstance(entry, str):
        return ModelSpec(name=name, model=entry)
    entry = dict(entry)
    return ModelSpec(
        name=name,
        model=entry.pop("model"),
        provider=entry.pop("provider", ""),
        api_key_env=entry.pop("api_key_env", ""),
        description=entry.pop("description", ""),
        params=entry.pop("params", {}) | entry,  # remaining keys are params too
    )


def _from_row(row) -> ModelSpec:
    return ModelSpec(
        name=row.name,
        model=row.model,
        provider=row.provider,
        params=row.params or {},
        api_key_env=row.api_key_env,
        description=row.description,
        source="db",
    )


def resolve(name: str | None = None) -> ModelSpec:
    """Return the ModelSpec for ``name`` (or the default) without building it."""
    from llm_kit.models import LLMModel

    if not name:
        row = LLMModel.objects.filter(is_active=True, is_default=True).first()
        if row:
            return _from_row(row)
        default = conf.get("DEFAULT_MODEL")
        if not default:
            raise UnknownModel(
                "No model requested and no default set. Mark an LLMModel as default or set LLM_KIT['DEFAULT_MODEL']."
            )
        name = default

    row = LLMModel.objects.filter(is_active=True, name=name).first()
    if row:
        return _from_row(row)
    if spec := _from_settings(name):
        return spec
    if ":" in name:
        return ModelSpec(name=name, model=name, source="raw")
    raise UnknownModel(f"Unknown model {name!r}: not in the catalogue, LLM_KIT['MODELS'], or a 'provider:model' id.")


def get_chat_model(name: str | None = None, **overrides: Any) -> BaseChatModel:
    """Build a chat model from the catalogue. ``overrides`` go to ``init_chat_model``."""
    return resolve(name).build(**overrides)


async def aget_chat_model(name: str | None = None, **overrides: Any) -> BaseChatModel:
    from asgiref.sync import sync_to_async

    spec = await sync_to_async(resolve)(name)
    return spec.build(**overrides)


def model_from_config(config: RunnableConfig | None, **overrides: Any) -> BaseChatModel:
    """Build the model selected for this run (``configurable.model``), or the default."""
    name = ((config or {}).get("configurable") or {}).get("model")
    return get_chat_model(name, **overrides)


async def amodel_from_config(config: RunnableConfig | None, **overrides: Any) -> BaseChatModel:
    name = ((config or {}).get("configurable") or {}).get("model")
    return await aget_chat_model(name, **overrides)


def available() -> list[ModelSpec]:
    """All selectable models: active DB rows plus settings aliases not shadowed by them."""
    from llm_kit.models import LLMModel

    specs = {row.name: _from_row(row) for row in LLMModel.objects.filter(is_active=True)}
    for name in conf.get("MODELS"):
        if name not in specs:
            specs[name] = _from_settings(name)
    return sorted(specs.values(), key=lambda s: s.name)
