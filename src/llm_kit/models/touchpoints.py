"""Admin overrides for touchpoints declared in code."""

from __future__ import annotations

from django.db import models


class TouchpointOverride(models.Model):
    """Change a touchpoint's model without a deploy. Code stays the source of the policy:
    ``allow`` can only narrow what the touchpoint's ``ClientChoice`` permits."""

    name = models.CharField(max_length=200, unique=True, help_text="Touchpoint name, e.g. crm.summarise_lead")
    default = models.CharField(max_length=255, blank=True, help_text="provider/model_id; blank = code default")
    fallbacks = models.JSONField(null=True, blank=True, help_text="List of provider/model_id; null = code fallbacks")
    params = models.JSONField(default=dict, blank=True, help_text='Merged over code params, e.g. {"temperature": 0}')
    allow = models.JSONField(null=True, blank=True, help_text="Glob patterns narrowing client choice; null = code")
    disabled = models.BooleanField(default=False, help_text="Kill switch: calls fail with TouchpointDisabled.")
    note = models.CharField(max_length=255, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name
