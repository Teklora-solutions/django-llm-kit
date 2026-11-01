"""Alert rules and the alerts they fired."""

from __future__ import annotations

from django.db import models

ALERT_KINDS = [
    ("spend", "Spend over threshold (USD in window)"),
    ("error_rate", "Error rate over threshold (0-1, in window)"),
    ("latency_p95", "p95 latency over threshold (ms, in window)"),
    ("unpriced_share", "Share of unpriced calls over threshold (0-1)"),
    ("provider_down", "A provider's circuit is open"),
    ("catalogue_change", "A touchpoint's model was removed, deprecated or changed price"),
]


class AlertRule(models.Model):
    name = models.CharField(max_length=200)
    kind = models.CharField(max_length=30, choices=ALERT_KINDS)
    threshold = models.DecimalField(max_digits=20, decimal_places=6, null=True, blank=True)
    window_minutes = models.PositiveIntegerField(default=60)
    min_calls = models.PositiveIntegerField(default=20, help_text="For rates: don't fire on too few calls.")
    filters = models.JSONField(
        default=dict,
        blank=True,
        help_text='Narrow the rule, e.g. {"touchpoint": "chat.assistant"} or {"provider": "gemini"}',
    )
    emails = models.JSONField(default=list, blank=True, help_text='["ops@example.com"]')
    webhook_url = models.URLField(blank=True, help_text="POSTed a JSON body (Slack/Discord-compatible 'text' field).")
    cooldown_minutes = models.PositiveIntegerField(
        default=60, help_text="Minimum time between two alerts of this rule."
    )
    enabled = models.BooleanField(default=True)
    last_fired_at = models.DateTimeField(null=True, blank=True, editable=False)
    last_checked_at = models.DateTimeField(null=True, blank=True, editable=False)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class AlertEvent(models.Model):
    rule = models.ForeignKey(AlertRule, on_delete=models.CASCADE, related_name="events")
    fired_at = models.DateTimeField(db_index=True)
    message = models.TextField()
    value = models.JSONField(null=True, blank=True)
    delivered = models.JSONField(default=dict, blank=True, help_text="Per channel: ok or the error.")

    class Meta:
        ordering = ["-fired_at"]

    def __str__(self) -> str:
        return f"{self.rule} @ {self.fired_at:%Y-%m-%d %H:%M}"


class MonitoringDashboard(models.Model):
    """Admin entry point for the monitoring dashboard (no table of its own)."""

    class Meta:
        managed = False
        default_permissions = ("view",)
        verbose_name = "monitoring dashboard"
        verbose_name_plural = "monitoring dashboard"
