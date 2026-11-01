"""Changes an agent proposed through a model tool, waiting for a person."""

from __future__ import annotations

from django.conf import settings
from django.db import models


class ToolProposal(models.Model):
    PENDING, APPROVED, REJECTED, FAILED = "pending", "approved", "rejected", "failed"
    STATUSES = [
        (PENDING, "Waiting for approval"),
        (APPROVED, "Approved and applied"),
        (REJECTED, "Rejected"),
        (FAILED, "Approved but could not be applied"),
    ]

    tool = models.CharField(max_length=100)
    model = models.CharField(max_length=100, help_text="app_label.ModelName")
    object_pk = models.CharField(max_length=64, blank=True, help_text="Empty when the proposal creates a row.")
    object_repr = models.CharField(max_length=200, blank=True)
    data = models.JSONField(help_text="The values the agent gave.")
    before = models.JSONField(default=dict, blank=True, help_text="Those fields' values when proposed.")
    after = models.JSONField(default=dict, blank=True, help_text="The values after validation.")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL, related_name="+", help_text="Acting for."
    )
    tenant = models.CharField(max_length=64, blank=True)
    thread_id = models.CharField(max_length=64, blank=True)
    status = models.CharField(max_length=10, choices=STATUSES, default=PENDING)
    created_at = models.DateTimeField(auto_now_add=True)
    decided_at = models.DateTimeField(null=True, blank=True)
    decided_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    reason = models.TextField(blank=True, help_text="Why it was rejected, or why applying it failed.")
    result_pk = models.CharField(max_length=64, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["status", "created_at"]), models.Index(fields=["model", "object_pk"])]

    @property
    def action(self) -> str:
        return "update" if self.object_pk else "create"

    def __str__(self) -> str:
        target = f"{self.model} {self.object_pk}" if self.object_pk else f"new {self.model}"
        return f"#{self.pk} {self.action} {target} ({self.status})"
