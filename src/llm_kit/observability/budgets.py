"""Daily spend limits per user and tenant (``LLM_KIT["DAILY_BUDGETS"]``), from recorded calls.

    LLM_KIT = {"DAILY_BUDGETS": {"user": 5, "tenant": 50}}   # USD since local midnight

Checked when a request starts (llm-kit's views, ``agents.run``). Calls are recorded in the
background, so the total can trail the last few seconds of spend; a request already running
is limited by ``AGENT_BUDGET`` instead.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from llm_kit import conf
from llm_kit.core.errors import BudgetExceeded

SCOPES = ("user", "tenant")


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
    unknown = set(limits) - set(SCOPES)
    if unknown:
        from llm_kit.core.errors import ConfigError

        raise ConfigError(f"DAILY_BUDGETS: unknown key(s) {sorted(unknown)}; use {', '.join(SCOPES)}.")
    ids = {"user": caller.user_id, "tenant": caller.tenant_id}
    for scope in SCOPES:
        limit, who = limits.get(scope), ids[scope]
        if limit is None or who is None:
            continue
        spent = spent_today(**{scope: str(who)})
        if spent >= Decimal(str(limit)):
            raise BudgetExceeded(scope, spent.quantize(Decimal("0.01")), limit)
