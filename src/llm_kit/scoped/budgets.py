"""Spending limits a tenant, a user, or each member of a tenant has, set by whoever you allow.

Which periods exist, and who may set each kind of budget, is yours to decide::

    LLM_KIT = {
        "BUDGET_PERIODS": ["day", "month"],               # or {"quarter": "myapp.periods:quarter_start"}
        "BUDGET_EDITORS": {                               # budget scope -> scopes that may set it
            "tenant": ["platform"],                       # your admins set each tenant's limit
            "member": ["platform", "tenant"],             # tenant managers set a per-member limit
            "user": ["user"],                             # users cap themselves
        },
    }

Budgets are checked before every model call (``Touchpoint.model``/``complete``/``embed``):
over one, the call raises ``BudgetExceeded`` and an agent stops with ``stopped="budget"``.
"""

from __future__ import annotations

import threading
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from llm_kit import conf
from llm_kit.core.observability.spend import SpendKey, check
from llm_kit.runtime.versioned_cache import VersionedCache
from llm_kit.scoped.service import NotPermitted, can_edit, scope_ids

BUDGET_SCOPES = ("tenant", "member", "user")
COUNTS = ("all", "platform_keys")

_caches: dict[tuple[str, str], VersionedCache] = {}
_lock = threading.Lock()


def periods() -> list[str]:
    configured = conf.get("BUDGET_PERIODS")
    return list(configured.keys() if isinstance(configured, dict) else configured or ())


def editors(scope: str) -> tuple[str, ...]:
    return tuple((conf.get("BUDGET_EDITORS") or {}).get(scope) or ())


def _target(who, scope: str) -> str | None:
    """The scope id a budget of ``scope`` has for ``who`` (member budgets live on the tenant)."""
    ids = scope_ids(who)
    return ids["tenant"] if scope in ("tenant", "member") else ids["user"]


def can_set(who, scope: str) -> bool:
    return any(can_edit(who, editor) for editor in editors(scope))


# ------------------------------------------------------------------ reading


def _cache(scope: str, scope_id: str) -> VersionedCache:
    with _lock:
        if (scope, scope_id) not in _caches:
            if len(_caches) > 10_000:
                _caches.clear()
            _caches[(scope, scope_id)] = VersionedCache(f"budgets:{scope}:{scope_id}")
        return _caches[(scope, scope_id)]


def stored(scope: str, scope_id: str) -> list[dict[str, Any]]:
    def load():
        from llm_kit.models import ScopedBudget

        return [
            {"touchpoint": b.touchpoint, "period": b.period, "amount": b.amount, "counts": b.counts}
            for b in ScopedBudget.objects.filter(scope=scope, scope_id=scope_id)
        ]

    return _cache(scope, scope_id).get(load)


def invalidate(scope: str, scope_id: str) -> None:
    _cache(scope, scope_id).bump()


def key_for(scope: str, who, budget: dict[str, Any]) -> SpendKey:
    ids = scope_ids(who)
    scope_id = f"{ids['tenant']}:{ids['user']}" if scope == "member" else ids["tenant" if scope == "tenant" else "user"]
    return SpendKey(scope, scope_id, budget["touchpoint"], budget["period"], budget["counts"])


def applying(who, touchpoint: str | None = None) -> list[tuple[str, dict[str, Any]]]:
    """``(scope, budget)`` for every budget that limits ``who`` (and ``touchpoint``, if given)."""
    ids = scope_ids(who)
    out = []
    for scope in BUDGET_SCOPES:
        scope_id = _target(who, scope)
        if scope_id is None or (scope == "member" and ids["user"] is None):
            continue
        for budget in stored(scope, scope_id):
            if touchpoint is None or budget["touchpoint"] in ("", touchpoint):
                out.append((scope, budget))
    return out


def check_all(who, tp) -> None:
    """Raise ``BudgetExceeded`` when any budget limiting ``who``'s calls of ``tp`` is used up."""
    for scope, budget in applying(who, tp.name):
        label = {"tenant": "workspace", "member": "per-member", "user": "personal"}[scope]
        if budget["touchpoint"]:
            label += f" {budget['touchpoint']!r}"
        check(key_for(scope, who, budget), Decimal(budget["amount"]), label)


# ------------------------------------------------------------------ for UIs


def resets_at(period: str):
    from llm_kit.observability.budgets import period_start

    start = period_start(period)
    if period == "day":
        return start + timedelta(days=1)
    if period == "week":
        return start + timedelta(days=7)
    if period == "month":
        return (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    return None


def describe(scope: str, who, budget: dict[str, Any]) -> dict[str, Any]:
    from llm_kit.core.observability.spend import spend

    spent = spend.spent(key_for(scope, who, budget))
    amount = Decimal(budget["amount"])
    ends = resets_at(budget["period"])
    return {
        "scope": scope,
        "touchpoint": budget["touchpoint"],
        "period": budget["period"],
        "amount": str(amount.normalize()),
        "counts": budget["counts"],
        "spent": str(spent.quantize(Decimal("0.0001"))),
        "remaining": str(max(amount - spent, Decimal(0)).quantize(Decimal("0.0001"))),
        "resets_at": ends.isoformat() if ends else None,
    }


def list_for(who, scope: str) -> list[dict[str, Any]]:
    scope_id = _target(who, scope)
    if scope_id is None:
        raise NotPermitted(f"You have no {scope}.")
    return [describe(scope, who, b) for b in stored(scope, scope_id)]


def _checked(who, scope: str, touchpoint: str, period: str) -> str:
    if scope not in BUDGET_SCOPES:
        raise ValueError(f"scope must be one of {', '.join(BUDGET_SCOPES)}.")
    if period not in periods():
        raise ValueError(f"period must be one of {', '.join(periods()) or '(none: set BUDGET_PERIODS)'}.")
    if touchpoint:
        from llm_kit.core.touchpoints import get_touchpoint

        try:
            get_touchpoint(touchpoint)
        except LookupError:
            raise ValueError(f"There is no touchpoint {touchpoint!r}.") from None
    scope_id = _target(who, scope)
    if scope_id is None:
        raise NotPermitted(f"You have no {scope}.")
    if not can_set(who, scope):
        raise NotPermitted(f"You can't set {scope} budgets.")
    return scope_id


def set_budget(who, scope: str, period: str, amount: Any, *, touchpoint: str = "", counts: str = "all"):
    from llm_kit.models import ScopedBudget

    scope_id = _checked(who, scope, touchpoint, period)
    try:
        amount = Decimal(str(amount))
    except InvalidOperation:
        raise ValueError("amount must be a number (USD).") from None
    if amount < 0:
        raise ValueError("amount can't be negative.")
    if counts not in COUNTS:
        raise ValueError(f"counts must be one of {', '.join(COUNTS)}.")
    user = getattr(who, "user", who)
    ScopedBudget.objects.update_or_create(
        scope=scope,
        scope_id=scope_id,
        touchpoint=touchpoint,
        period=period,
        defaults={"amount": amount, "counts": counts, "updated_by": user if getattr(user, "pk", None) else None},
    )
    invalidate(scope, scope_id)
    return describe(scope, who, {"touchpoint": touchpoint, "period": period, "amount": amount, "counts": counts})


def remove_budget(who, scope: str, period: str, *, touchpoint: str = "") -> bool:
    from llm_kit.models import ScopedBudget

    scope_id = _checked(who, scope, touchpoint, period)
    deleted, _ = ScopedBudget.objects.filter(
        scope=scope, scope_id=scope_id, touchpoint=touchpoint, period=period
    ).delete()
    invalidate(scope, scope_id)
    return bool(deleted)
