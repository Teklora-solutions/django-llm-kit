"""Who a tool acts for. Always set by the server, never chosen by the model.

Runs started by llm-kit's views put a ``Caller`` in ``configurable["__llm_kit_caller"]``.
LangGraph never copies ``"__"`` keys into checkpoints, and ``build_config`` drops any
``"__"`` key a client sends, so a client can't pretend to be someone else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_kit import conf
from llm_kit.core.graphs import import_target

CONFIG_KEY = "__llm_kit_caller"

# Permission meaning "any signed-in user" (for hand-written tools). Anonymous callers never get tools.
AUTHENTICATED = "__authenticated__"

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
        if not self.is_authenticated or getattr(self.user, "is_active", True) is False:
            return False
        if permission == AUTHENTICATED:
            return True
        if callable(permission):
            return bool(permission(self.user))
        return bool(self.user.has_perm(permission))


def resolve_tenant(user) -> Any:
    path = conf.get("TENANT_RESOLVER")
    if not path or user is None or not getattr(user, "is_authenticated", False):
        return None
    return import_target(path)(user)


def caller_for(user, tenant: Any = _UNSET) -> Caller:
    """A caller for ``user``; the tenant comes from ``TENANT_RESOLVER`` unless given."""
    if isinstance(user, Caller):
        return user
    return Caller(user=user, tenant=resolve_tenant(user) if tenant is _UNSET else tenant)


def caller_from(config) -> Caller | None:
    """The caller the server put in a run's config, or None."""
    value = ((config or {}).get("configurable") or {}).get(CONFIG_KEY)
    return value if isinstance(value, Caller) else None
