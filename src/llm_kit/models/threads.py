from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models


class Thread(models.Model):
    """A conversation / run history. Its ``id`` is the LangGraph ``thread_id``."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="llm_threads",
    )
    # Anonymous visitors: a random id kept in their session data (survives the key rotation at login).
    session_key = models.CharField(max_length=64, blank=True, default="", db_index=True)
    # Chosen by your code, never by a client: e.g. "whatsapp:254700000000". Unique per tenant.
    external_key = models.CharField(max_length=255, blank=True, default="")
    tenant = models.CharField(max_length=64, blank=True, default="", help_text="Tenant id (string).")
    graph = models.CharField(max_length=100, help_text="Graph or agent name.")
    title = models.CharField(max_length=255, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [models.Index(fields=["user", "-updated_at"])]
        constraints = [
            models.UniqueConstraint(
                fields=["tenant", "external_key"],
                condition=~models.Q(external_key=""),
                name="llm_kit_thread_external_key_per_tenant",
            )
        ]

    def __str__(self) -> str:
        return self.title or str(self.id)

    @property
    def owner_label(self) -> str:
        if self.user_id:
            return f"user {self.user_id}"
        if self.external_key:
            return self.external_key
        if self.session_key:
            return "visitor"
        return "nobody (code only)"

    @property
    def thread_id(self) -> str:
        return str(self.id)

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}
