"""Which rows a caller may see and change. Applied to every query and every write."""

from __future__ import annotations

from django.core.exceptions import ImproperlyConfigured
from django.db.models import QuerySet

from llm_kit.tools.caller import Caller


class OutOfScope(Exception):
    """A write would leave the row outside the caller's rows (rolled back)."""


def scoped(spec, caller: Caller) -> QuerySet:
    qs = spec.model._default_manager.all()
    if spec.unscoped is not None:
        return qs
    if spec.owner_field:
        if not caller.is_authenticated:
            return qs.none()
        qs = qs.filter(**{spec.owner_field: caller.user_id})
    if spec.tenant_field:
        if caller.tenant_id is None:
            return qs.none()  # tenant-scoped data and no tenant: nothing, never everything
        qs = qs.filter(**{spec.tenant_field: caller.tenant_id})
    if spec.queryset_fn is not None:
        result = spec.queryset_fn(qs, caller)
        if not isinstance(result, QuerySet) or result.model is not spec.model:
            raise ImproperlyConfigured(f"{spec.label} agent_queryset must return a {spec.model.__name__} queryset")
        qs = result
    return qs


def inject(spec, obj, caller: Caller) -> None:
    """On create, the server sets the owner and tenant (direct fields only; paths are checked after saving)."""
    if spec.owner_field and "__" not in spec.owner_field and spec.owner_field != "pk":
        setattr(obj, spec.field(spec.owner_field).attname, caller.user_id)
    if spec.tenant_field and "__" not in spec.tenant_field:
        setattr(obj, spec.field(spec.tenant_field).attname, caller.tenant_id)
