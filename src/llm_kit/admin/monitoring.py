from datetime import timedelta

from django.contrib import admin
from django.template.response import TemplateResponse
from django.utils import timezone

from llm_kit.models import AlertEvent, AlertRule, MonitoringDashboard

PERIODS = {"24h": 1, "7d": 7, "30d": 30}


@admin.register(MonitoringDashboard)
class MonitoringDashboardAdmin(admin.ModelAdmin):
    """The numbers come from ``llm_kit.monitoring``, the same functions your own dashboards can call."""

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def changelist_view(self, request, extra_context=None):
        from llm_kit import monitoring
        from llm_kit.models import CatalogueChange
        from llm_kit.monitoring.alerts import _touchpoint_models
        from llm_kit.monitoring.chart import daily_columns
        from llm_kit.monitoring.health import report

        period = request.GET.get("period", "7d")
        if period not in PERIODS:
            period = "7d"
        days = PERIODS[period]
        summary = monitoring.summary(period=period)
        chart_days = max(days, 7)
        spend = monitoring.timeseries("cost", bucket="day", period=f"{chart_days}d")
        used = _touchpoint_models()
        changes = [
            c
            for c in CatalogueChange.objects.filter(synced_at__gte=timezone.now() - timedelta(days=30)).order_by(
                "-synced_at"
            )[:500]
            if f"{c.provider}/{c.model_id}" in used
        ][:20]
        context = {
            **self.admin_site.each_context(request),
            "title": "LLM monitoring",
            "opts": self.model._meta,
            "period": period,
            "periods": list(PERIODS),
            "s": summary,
            "chart": daily_columns(spend["points"], chart_days, end=timezone.localdate()),
            "by_touchpoint": monitoring.breakdown("touchpoint", period=period, limit=15)["rows"],
            "by_model": monitoring.breakdown("model", period=period, limit=15)["rows"],
            "errors": monitoring.errors(period=period)["by_class"],
            "providers": monitoring.provider_health(period="15m")["providers"],
            "health": report(),
            "alerts": AlertEvent.objects.select_related("rule")[:10],
            "rules": AlertRule.objects.count(),
            "changes": changes,
            "used": used,
        }
        return TemplateResponse(request, "admin/llm_kit/dashboard.html", context)


@admin.register(AlertRule)
class AlertRuleAdmin(admin.ModelAdmin):
    list_display = ("name", "kind", "threshold", "window_minutes", "enabled", "last_fired_at", "last_checked_at")
    list_filter = ("kind", "enabled")
    list_editable = ("enabled",)


@admin.register(AlertEvent)
class AlertEventAdmin(admin.ModelAdmin):
    list_display = ("fired_at", "rule", "message", "delivered")
    list_filter = ("rule",)
    date_hierarchy = "fired_at"

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
