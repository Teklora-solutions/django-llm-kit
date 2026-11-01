"""What resolution needs from the outside world, so the core stays framework-free."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from llm_kit.core.catalogue.fields import normalise_provider

# Providers authenticated by something other than an API key (cloud IAM, local servers).
KEYLESS_PROVIDERS = frozenset(
    {"vertex_ai", "bedrock", "sagemaker", "ollama", "ollama_chat", "lm_studio", "hosted_vllm"}
)


@dataclass
class ModelInfo:
    """What the catalogue knows about one model."""

    ref: str  # canonical provider/model_id
    values: dict[str, Any] = field(default_factory=dict)
    hidden: bool = False


@dataclass
class Override:
    """Admin changes to one touchpoint (stored by the framework adapter)."""

    default: str | None = None
    fallbacks: list[str] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    allow: list[str] | None = None  # narrows ClientChoice.allow; never widens it
    disabled: bool = False


class ResolutionEnv(Protocol):
    def lookup(self, ref: str) -> ModelInfo | None: ...

    def override(self, touchpoint: str) -> Override | None: ...

    def api_keys(self, provider: str) -> list[str]: ...


def env_api_keys(provider: str, extra_env_names: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Keys from the environment: ``{P}_API_KEYS`` (comma-separated pool), then single-key variables."""
    p = normalise_provider(provider).upper()
    keys: list[str] = []
    pool = os.environ.get(f"{p}_API_KEYS", "")
    keys += [k.strip() for k in pool.split(",") if k.strip()]
    for name in (*extra_env_names, f"{p}_API_KEY"):
        v = os.environ.get(name)
        if v and v not in keys:
            keys.append(v)
    return keys


class NullEnv:
    """No catalogue, no overrides; keys from the environment only."""

    def lookup(self, ref: str) -> ModelInfo | None:
        return None

    def override(self, touchpoint: str) -> Override | None:
        return None

    def api_keys(self, provider: str) -> list[str]:
        return env_api_keys(provider)


_default_env: ResolutionEnv = NullEnv()


def set_default_env(env: ResolutionEnv) -> None:
    """Framework adapters install their env here (the Django app does it in ``ready()``)."""
    global _default_env
    _default_env = env


def default_env() -> ResolutionEnv:
    return _default_env


def provider_of(ref: str) -> str:
    return normalise_provider(ref.split("/", 1)[0]) if "/" in ref else ""


def has_credentials(env: ResolutionEnv, ref: str) -> bool:
    provider = provider_of(ref)
    return provider in KEYLESS_PROVIDERS or bool(env.api_keys(provider))
