"""Daily spend limits per user, tenant and touchpoint (``LLM_KIT["DAILY_BUDGETS"]``), from recorded calls.

    LLM_KIT = {"DAILY_BUDGETS": {"user": 5, "tenant": 50}}   # USD since local midnight
    LLM_KIT = {"DAILY_BUDGETS": {"touchpoints": {"crm.digest": 20}}}   # per touchpoint

Users and tenants are checked when a request starts (llm-kit's views, ``agents.run``);
touchpoints before each call, from a running total (``llm_kit.core.observability.spend``).
Calls are recorded in the background, so totals can trail the last few seconds of spend;
a request already running is limited by ``AGENT_BUDGET`` instead.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from llm_kit import conf
from llm_kit.core.errors import BudgetExceeded

SCOPES = ("user", "tenant")
KEYS = (*SCOPES, "touchpoints")  # touchpoints: checked on each call (llm_kit.core.observability.spend)


def spent_since(start, **where: Any) -> Decimal:
    """USD recorded since ``start`` for the given ``LLMCall`` filters."""
    from django.db.models import Sum

    from llm_kit.models import LLMCall

    total = LLMCall.objects.filter(started_at__gte=start, **where).aggregate(total=Sum("cost"))["total"]
    return total or Decimal(0)


def seed(key, start) -> Decimal:
    """Recorded spend for a ``SpendKey`` since ``start`` (the tracker's database read)."""
    where: dict[str, Any] = {}
    if key.touchpoint:
        where["touchpoint"] = key.touchpoint
    if key.scope in ("tenant", "user"):
        where[key.scope] = key.scope_id
    elif key.scope == "member":
        where["tenant"], _, where["user"] = key.scope_id.partition(":")
    if key.counts == "platform_keys":
        where["key_owner__in"] = ["", "platform"]
    return spent_since(start, **where)


def period_start(period: str):
    """When the current ``period`` started, in local time: ``day``, ``week`` (Monday), ``month``,
    or a period from ``BUDGET_PERIODS`` given as ``{"name": "dotted.path:fn(now) -> start"}``."""
    from django.utils import timezone

    from llm_kit.core.observability.spend import default_period_start

    now = timezone.localtime()
    custom = conf.get("BUDGET_PERIODS")
    if isinstance(custom, dict) and custom.get(period):
        from llm_kit.core.graphs import import_target

        fn = custom[period]
        return (import_target(fn) if isinstance(fn, str) else fn)(now)
    return default_period_start(period, now)


def spent_today(**where: Any) -> Decimal:
    """USD recorded since local midnight for ``user=`` or ``tenant=`` (as recorded: strings)."""
    from django.db.models import Sum
    from django.utils import timezone

    from llm_kit.models import LLMCall

    midnight = timezone.localtime().replace(hour=0, minute=0, second=0, microsecond=0)
    total = LLMCall.objects.filter(started_at__gte=midnight, **where).aggregate(total=Sum("cost"))["total"]
    return total or Decimal(0)


def check_daily(caller) -> None:
    """Raise ``BudgetExceeded`` when ``caller``'s user or tenant is over today's budget."""
    limits = conf.get("DAILY_BUDGETS") or {}
    if not limits or caller is None:
        return
    unknown = set(limits) - set(KEYS)
    if unknown:
        from llm_kit.core.errors import ConfigError

        raise ConfigError(f"DAILY_BUDGETS: unknown key(s) {sorted(unknown)}; use {', '.join(KEYS)}.")
    ids = {"user": caller.user_id, "tenant": caller.tenant_id}
    for scope in SCOPES:
        limit, who = limits.get(scope), ids[scope]
        if limit is None or who is None:
            continue
        spent = spent_today(**{scope: str(who)})
        if spent >= Decimal(str(limit)):
            raise BudgetExceeded(scope, spent.quantize(Decimal("0.01")), limit)
