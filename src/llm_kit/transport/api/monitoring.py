"""Monitoring JSON for custom dashboards. Same numbers as ``llm_kit.monitoring`` and the admin.

    GET monitoring/summary/?period=24h&touchpoint=chat.assistant
    GET monitoring/timeseries/?metric=cost&bucket=day&period=30d&group_by=model
    GET monitoring/breakdown/?by=touchpoint&period=7d
    GET monitoring/errors/?period=24h
    GET monitoring/top/?by=user&metric=cost&n=10
    GET monitoring/providers/?period=15m
    GET monitoring/alerts/

Common filters: period | since & until, touchpoint, model, provider, user, tenant, graph, status, chosen_by.
Staff only unless ``LLM_KIT["MONITORING_PERMISSION"]`` names another DRF permission class.
"""

from __future__ import annotations

from rest_framework.exceptions import ValidationError
from rest_framework.permissions import IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView

from llm_kit import conf, monitoring
from llm_kit.core.graphs import import_target
from llm_kit.monitoring.filters import FILTER_FIELDS

_COMMON = (*FILTER_FIELDS, "period", "since", "until")


def _permission_classes():
    path = conf.get("MONITORING_PERMISSION")
    return [import_target(path)] if path else [IsAdminUser]


class _MonitoringView(APIView):
    extra: tuple[str, ...] = ()

    def get_permissions(self):
        return [cls() for cls in _permission_classes()]

    def params(self, request) -> dict:
        allowed = (*_COMMON, *self.extra)
        unknown = set(request.query_params) - set(allowed)
        if unknown:
            raise ValidationError({"detail": f"Unknown parameter(s) {sorted(unknown)}. Allowed: {', '.join(allowed)}."})
        return {k: v for k, v in request.query_params.items() if v != ""}

    def respond(self, fn, request, **fixed):
        try:
            return Response(fn(**{**self.params(request), **fixed}))
        except ValueError as e:
            raise ValidationError({"detail": str(e)}) from e


class SummaryView(_MonitoringView):
    def get(self, request):
        return self.respond(monitoring.summary, request)


class TimeseriesView(_MonitoringView):
    extra = ("metric", "bucket", "group_by")

    def get(self, request):
        return self.respond(monitoring.timeseries, request)


class BreakdownView(_MonitoringView):
    extra = ("by", "limit")

    def get(self, request):
        p = self.params(request)
        if "limit" in p:
            p["limit"] = _int(p["limit"], "limit")
        try:
            return Response(monitoring.breakdown(**p))
        except ValueError as e:
            raise ValidationError({"detail": str(e)}) from e


class ErrorsView(_MonitoringView):
    def get(self, request):
        return self.respond(monitoring.errors, request)


class TopView(_MonitoringView):
    extra = ("by", "metric", "n")

    def get(self, request):
        p = self.params(request)
        if "n" in p:
            p["n"] = min(_int(p["n"], "n"), 100)
        try:
            return Response(monitoring.top(**p))
        except ValueError as e:
            raise ValidationError({"detail": str(e)}) from e


class ProvidersView(_MonitoringView):
    def get(self, request):
        return self.respond(monitoring.provider_health, request)


class AlertsView(_MonitoringView):
    def get(self, request):
        from llm_kit.models import AlertEvent, AlertRule

        rules = list(
            AlertRule.objects.values("id", "name", "kind", "threshold", "enabled", "last_fired_at", "last_checked_at")
        )
        events = list(
            AlertEvent.objects.select_related("rule").values("id", "rule__name", "fired_at", "message", "value")[:50]
        )
        return Response({"rules": rules, "events": events})


def _int(value, name):
    try:
        return int(value)
    except ValueError:
        raise ValidationError({name: "Must be an integer."}) from None
