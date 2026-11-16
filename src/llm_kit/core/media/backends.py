"""Who makes the media: a backend per model, built by a factory like the embedding one.

``touchpoint(..., media_backend=)``, else ``LLM_KIT["MEDIA_BACKEND"]``: a name below, a dotted
path, or the factory itself. A factory is ``fn(ref, params, api_keys)`` returning a backend::

    class MyMusic:
        def __init__(self, ref, params, api_keys): ...
        def generate(self, kind, request, **params) -> MediaResult: ...
        async def agenerate(self, kind, request, **params) -> MediaResult: ...
        # a long kind (video, music): submit, then check until done
        def submit(self, kind, request, **params) -> str: ...        # the provider's job id
        def poll(self, kind, job_id) -> JobState: ...
        async def asubmit(...) / apoll(...)                           # optional: else run on a thread
        def cancel(self, kind, job_id) -> None: ...                   # optional

* ``litellm`` (and ``auto``, the default): LiteLLM's ``image_generation``, ``image_edit``,
  ``speech``, ``transcription``, ``ocr`` and ``video_generation`` / ``video_status`` /
  ``video_content``, for every provider LiteLLM supports.
* a dotted path or a factory: any other provider, or a kind of your own.
"""

from __future__ import annotations

import asyncio
import random
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from llm_kit.core import config
from llm_kit.core.errors import ConfigError
from llm_kit.core.graphs import import_target
from llm_kit.core.media.types import MediaResult

MediaFactory = Callable[[str, dict[str, Any], list[str]], Any]

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"


@dataclass
class JobState:
    """Where a long job is at the provider."""

    status: str  # pending | running | done | failed
    progress: float | None = None  # 0-100 when the provider says
    error: str = ""
    result: MediaResult | None = None  # when done

    @property
    def finished(self) -> bool:
        return self.status in (DONE, FAILED)


@runtime_checkable
class MediaBackend(Protocol):
    def generate(self, kind: str, request: dict[str, Any], **params: Any) -> MediaResult: ...


def supports_jobs(backend: Any) -> bool:
    return callable(getattr(backend, "submit", None)) and callable(getattr(backend, "poll", None))


async def agenerate(backend: Any, kind: str, request: dict[str, Any], **params: Any) -> MediaResult:
    if callable(getattr(backend, "agenerate", None)):
        return await backend.agenerate(kind, request, **params)
    return await asyncio.to_thread(backend.generate, kind, request, **params)


async def asubmit(backend: Any, kind: str, request: dict[str, Any], **params: Any) -> str:
    if callable(getattr(backend, "asubmit", None)):
        return await backend.asubmit(kind, request, **params)
    return await asyncio.to_thread(backend.submit, kind, request, **params)


async def apoll(backend: Any, kind: str, job_id: str) -> JobState:
    if callable(getattr(backend, "apoll", None)):
        return await backend.apoll(kind, job_id)
    return await asyncio.to_thread(backend.poll, kind, job_id)


async def acancel(backend: Any, kind: str, job_id: str) -> None:
    if callable(getattr(backend, "acancel", None)):
        await backend.acancel(kind, job_id)
    elif callable(getattr(backend, "cancel", None)):
        await asyncio.to_thread(backend.cancel, kind, job_id)


# ------------------------------------------------------------------ LiteLLM


def _litellm():
    from llm_kit.core.touchpoints.backends import _quiet_litellm_import

    _quiet_litellm_import()
    try:
        import litellm
    except ImportError as e:
        raise ConfigError("The litellm media backend needs: pip install django-llm-kit[litellm]") from e
    return litellm


def get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def reported_cost(response: Any) -> Any:
    """The cost LiteLLM worked out for a response, when it did."""
    hidden = getattr(response, "_hidden_params", None) or {}
    cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
    return cost if cost not in (None, 0, 0.0) else None


class LiteLLMMedia:
    """Media through LiteLLM. Each kind is a pair of methods, ``_<kind>`` and ``_a<kind>``."""

    def __init__(self, ref: str, params: dict[str, Any], keys: list[str]) -> None:
        self.ref, self.params, self.keys = ref, dict(params), list(keys)

    def _kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        kwargs = {**self.params, **params}
        if self.keys:
            kwargs.setdefault("api_key", random.choice(self.keys))
        return kwargs

    def _op(self, kind: str, prefix: str = "_") -> Callable[..., Any]:
        op = getattr(self, f"{prefix}{kind}", None)
        if op is None:
            raise ConfigError(f"The litellm media backend doesn't make {kind!r}; give the touchpoint a media_backend.")
        return op

    def generate(self, kind: str, request: dict[str, Any], **params: Any) -> MediaResult:
        return self._op(kind)(request, self._kwargs(params))

    async def agenerate(self, kind: str, request: dict[str, Any], **params: Any) -> MediaResult:
        return await self._op(kind, "_a")(request, self._kwargs(params))


_factories: dict[str, MediaFactory] = {}
_lock = threading.Lock()


def get_media_factory(name: str | MediaFactory | None) -> MediaFactory:
    """A built-in backend by name, your factory by dotted path, or your factory itself."""
    if callable(name):
        return name
    name = name or config.get("MEDIA_BACKEND") or "auto"
    if callable(name):
        return name
    if name in ("auto", "litellm"):
        return LiteLLMMedia
    with _lock:
        if name not in _factories:
            factory = import_target(name)
            if not callable(factory):
                raise ConfigError(f"Media backend {name!r} is not callable.")
            _factories[name] = factory
    return _factories[name]
