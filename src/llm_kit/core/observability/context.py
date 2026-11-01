"""Who/what a call belongs to, for code outside a request (workers, scripts).

with llm_kit.observability.context(user=request.user.pk, tenant="acme", tags={"job": "nightly"}):
    summarise.complete(...)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class ObsContext:
    user: str | None = None
    tenant: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)


_current: ContextVar[ObsContext | None] = ContextVar("llm_kit_obs_context", default=None)


def current() -> ObsContext:
    return _current.get() or ObsContext()


@contextmanager
def context(*, user: Any = None, tenant: Any = None, tags: dict[str, Any] | None = None) -> Iterator[ObsContext]:
    """Nested contexts inherit and override the outer one."""
    outer = current()
    ctx = replace(
        outer,
        user=str(user) if user is not None else outer.user,
        tenant=str(tenant) if tenant is not None else outer.tenant,
        tags={**outer.tags, **(tags or {})},
    )
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)
