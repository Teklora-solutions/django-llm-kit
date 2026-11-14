"""Settings saved by a scope (the platform, a tenant, a user): see ``llm_kit.scoped``."""

from __future__ import annotations

from django.conf import settings
from django.db import models

SCOPE_CHOICES = [("platform", "Platform"), ("tenant", "Tenant"), ("user", "User")]


class ScopedSetting(models.Model):
    """One saved value. ``scope_id`` is the tenant's or user's id as text ("" for the platform);
    ``touchpoint`` is "" for settings that aren't tied to one."""

    scope = models.CharField(max_length=10, choices=SCOPE_CHOICES)
    scope_id = models.CharField(max_length=255, blank=True)
    touchpoint = models.CharField(max_length=200, blank=True)
    key = models.CharField(max_length=200)
    value = models.JSONField()
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["scope", "scope_id", "touchpoint", "key"], name="llm_kit_scoped_uniq")
        ]
        indexes = [models.Index(fields=["scope", "scope_id"])]
        ordering = ["scope", "scope_id", "touchpoint", "key"]

    def __str__(self) -> str:
        where = f"{self.scope}:{self.scope_id}" if self.scope_id else self.scope
        return f"{where} {self.touchpoint + ' ' if self.touchpoint else ''}{self.key}"


class ScopedBudget(models.Model):
    """A spending limit for a tenant, a user, or each member of a tenant (``member``), per period."""

    scope = models.CharField(
        max_length=10, choices=[("tenant", "Tenant"), ("member", "Each member of a tenant"), ("user", "User")]
    )
    scope_id = models.CharField(max_length=255, help_text="The tenant's id (tenant, member) or the user's (user).")
    touchpoint = models.CharField(max_length=200, blank=True, help_text="Blank: every touchpoint.")
    period = models.CharField(max_length=20, help_text="day, week, month, or one of BUDGET_PERIODS.")
    amount = models.DecimalField(max_digits=14, decimal_places=4, help_text="USD")
    counts = models.CharField(
        max_length=20,
        default="all",
        choices=[("all", "All calls"), ("platform_keys", "Only calls on the platform's keys")],
    )
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["scope", "scope_id", "touchpoint", "period"], name="llm_kit_scoped_budget_uniq"
            )
        ]
        ordering = ["scope", "scope_id", "touchpoint", "period"]

    def __str__(self) -> str:
        return f"{self.scope}:{self.scope_id} {self.touchpoint or 'all'} ${self.amount}/{self.period}"
