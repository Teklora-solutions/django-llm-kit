"""Who a tool acts for. Always set by the server, never chosen by the model.

Runs started by llm-kit's views put a ``Caller`` in ``configurable["__llm_kit_caller"]``.
LangGraph never copies ``"__"`` keys into checkpoints, and ``build_config`` drops any
``"__"`` key a client sends, so a client can't pretend to be someone else.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Any

from llm_kit import conf
from llm_kit.core.graphs import import_target

CONFIG_KEY = "__llm_kit_caller"

# Permission meaning "any signed-in user" (for hand-written tools).
AUTHENTICATED = "__authenticated__"
# Permission meaning "anyone": anonymous visitors and runs with no user (a Celery task). Only for
# hand-written tools that read public data (site search, the time); never for tools on your models.
PUBLIC = "__public__"

_UNSET: Any = object()


@dataclass(frozen=True)
class Caller:
    user: Any = None
    tenant: Any = None  # an instance or an id; scoping uses tenant_id

    @property
    def is_authenticated(self) -> bool:
        return self.user is not None and bool(getattr(self.user, "is_authenticated", False))

    @property
    def user_id(self) -> Any:
        return self.user.pk if self.is_authenticated else None

    @property
    def tenant_id(self) -> Any:
        return getattr(self.tenant, "pk", self.tenant)

    def has_perm(self, permission) -> bool:
        if permission == PUBLIC:
            return True
        if not self.is_authenticated or getattr(self.user, "is_active", True) is False:
            return False
        if permission == AUTHENTICATED:
            return True
        if callable(permission):
            return bool(permission(self.user))
        return bool(self.user.has_perm(permission))


@functools.lru_cache(maxsize=32)
def _context_params(fn) -> tuple[str, ...]:
    """Which of ``request`` / ``scope`` a resolver takes (by name, or through ``**kwargs``)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return ()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return ("request", "scope")
    return tuple(name for name in ("request", "scope") if name in params)


def resolve_tenant(user, *, request=None, scope=None) -> Any:
    """``TENANT_RESOLVER(user)``, or ``(user, request=..., scope=...)`` when it takes them.

    ``request`` is the Django (or DRF) request, ``scope`` the Channels scope; both are None
    outside a request (``agents.run`` from a task). A resolver that raises Django's
    ``PermissionDenied`` (a tenant the user doesn't belong to) refuses the request:
    ``NotAllowed``, a 403 from the views and a 4403 close from the WebSocket consumer.
    """
    path = conf.get("TENANT_RESOLVER")
    if not path or user is None or not getattr(user, "is_authenticated", False):
        return None
    fn = import_target(path)
    context = {"request": request, "scope": scope}
    kwargs = {name: context[name] for name in _context_params(fn)}
    from django.core.exceptions import PermissionDenied

    try:
        return fn(user, **kwargs)
    except PermissionDenied as e:
        from llm_kit.runtime.access import NotAllowed

        raise NotAllowed(str(e) or "Not allowed for this tenant.") from e


def caller_for(user, tenant: Any = _UNSET) -> Caller:
    """A caller for ``user``; the tenant comes from ``TENANT_RESOLVER`` unless given."""
    if isinstance(user, Caller):
        return user
    return Caller(user=user, tenant=resolve_tenant(user) if tenant is _UNSET else tenant)


def caller_from(config) -> Caller | None:
    """The caller the server put in a run's config, or None."""
    value = ((config or {}).get("configurable") or {}).get(CONFIG_KEY)
    return value if isinstance(value, Caller) else None
