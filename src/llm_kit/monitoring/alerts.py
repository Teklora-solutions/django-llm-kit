"""Evaluate alert rules and deliver what fires (signal, email, webhook).

Run every few minutes: ``manage.py llm_kit_check_alerts``, or schedule
``llm_kit.tasks.run_alert_checks`` (Celery: ``llm_kit.check_alerts``).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import timedelta
from decimal import Decimal
from typing import Any

from django.core.mail import send_mail
from django.dispatch import Signal
from django.utils import timezone

from llm_kit.models import AlertEvent, AlertRule, CatalogueChange

logger = logging.getLogger("llm_kit.alerts")

alert_fired = Signal()
"""Sent with ``rule`` and ``event`` whenever an alert fires. Hook your own delivery here."""


def _window(rule: AlertRule) -> dict[str, Any]:
    return {"period": f"{rule.window_minutes}m", **(rule.filters or {})}


def evaluate(rule: AlertRule) -> tuple[bool, str, Any]:
    """Whether ``rule`` is breached now, a human message, and the measured value."""
    from llm_kit import monitoring

    t = rule.threshold
    if rule.kind == "spend":
        cost = monitoring.summary(**_window(rule))["cost"]
        return (
            t is not None and cost > t,
            f"Spend ${cost.normalize():f} in the last {rule.window_minutes} min (threshold ${t})",
            str(cost),
        )
    if rule.kind in ("error_rate", "unpriced_share", "latency_p95"):
        s = monitoring.summary(**_window(rule))
        if s["calls"] < rule.min_calls:
            return False, f"only {s['calls']} calls (min {rule.min_calls})", None
        if rule.kind == "error_rate":
            v = Decimal(s["errors"]) / s["calls"]  # exact; summary's error_rate is rounded
            return t is not None and v > t, f"Error rate {v:.1%} over {s['calls']} calls (threshold {t:.1%})", float(v)
        if rule.kind == "unpriced_share":
            v = Decimal(s["unpriced"]) / s["calls"]
            return t is not None and v > t, f"{v:.1%} of calls unpriced (threshold {t:.1%})", float(v)
        p95 = s["latency_ms"]["p95"]
        return t is not None and p95 is not None and p95 > t, f"p95 latency {p95} ms (threshold {t} ms)", p95
    if rule.kind == "provider_down":
        from llm_kit.monitoring.breaker import breaker

        wanted = (rule.filters or {}).get("provider")
        down = {p: s for p, s in breaker.states().items() if s["state"] == "open" and (not wanted or p == wanted)}
        msg = "; ".join(f"{p} ({s['reason']}, retry in {s['retry_in_s']} s)" for p, s in down.items())
        return bool(down), f"Provider circuit open: {msg}" if down else "all circuits closed", sorted(down)
    if rule.kind == "catalogue_change":
        return _catalogue_change(rule)
    raise ValueError(f"Unknown alert kind {rule.kind!r}")


def _touchpoint_models() -> dict[str, list[str]]:
    """model ref -> touchpoints that use it as default or fallback."""
    from llm_kit.core.touchpoints import all_touchpoints

    used: dict[str, list[str]] = {}
    for tp in all_touchpoints():
        try:
            refs = tp.resolve().candidates + [tp.default, *tp.fallbacks]
        except Exception:
            refs = [tp.default, *tp.fallbacks]
        for ref in set(refs):
            used.setdefault(ref, []).append(tp.name)
    return used


def _catalogue_change(rule: AlertRule):
    since = rule.last_checked_at or timezone.now() - timedelta(minutes=rule.window_minutes)
    used = _touchpoint_models()
    hits = []
    for c in CatalogueChange.objects.filter(
        synced_at__gt=since, kind__in=("removed", "deprecated", "price_changed")
    ).order_by("synced_at"):
        ref = f"{c.provider}/{c.model_id}"
        if ref in used:
            detail = f" {c.field}: {c.old} → {c.new}" if c.field else ""
            hits.append(f"{ref} {c.kind}{detail} (used by {', '.join(sorted(used[ref]))})")
    return bool(hits), "Catalogue changes affecting touchpoints: " + "; ".join(hits) if hits else "none", hits


def _deliver(rule: AlertRule, event: AlertEvent) -> dict[str, str]:
    results: dict[str, str] = {}
    subject = f"[llm-kit] {rule.name}"
    if rule.emails:
        try:
            send_mail(subject, event.message, None, rule.emails)
            results["email"] = "ok"
        except Exception as e:
            results["email"] = f"{type(e).__name__}: {e}"[:300]
    if rule.webhook_url:
        body = json.dumps(
            {
                "text": f"{subject}: {event.message}",
                "rule": rule.name,
                "kind": rule.kind,
                "value": event.value,
                "fired_at": event.fired_at.isoformat(),
            },
            default=str,
        ).encode()
        req = urllib.request.Request(rule.webhook_url, body, {"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                results["webhook"] = "ok" if resp.status < 300 else f"HTTP {resp.status}"
        except Exception as e:
            results["webhook"] = f"{type(e).__name__}: {e}"[:300]
    return results


def check(rule: AlertRule, *, now=None) -> AlertEvent | None:
    now = now or timezone.now()
    fired = None
    try:
        breached, message, value = evaluate(rule)
    except Exception:
        logger.exception("alert rule %s failed to evaluate", rule.pk)
        breached, message, value = False, "", None
    cooling = rule.last_fired_at and now - rule.last_fired_at < timedelta(minutes=rule.cooldown_minutes)
    if breached and not cooling:
        fired = AlertEvent.objects.create(rule=rule, fired_at=now, message=message, value=value)
        fired.delivered = _deliver(rule, fired)
        fired.save(update_fields=["delivered"])
        rule.last_fired_at = now
        alert_fired.send(sender=AlertRule, rule=rule, event=fired)
        logger.warning("llm-kit alert %s: %s", rule.name, message)
    rule.last_checked_at = now
    rule.save(update_fields=["last_fired_at", "last_checked_at"])
    return fired


def check_all() -> list[AlertEvent]:
    return [e for rule in AlertRule.objects.filter(enabled=True) if (e := check(rule)) is not None]
