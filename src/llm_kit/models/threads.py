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
    graph = models.CharField(max_length=100, help_text="Name from LLM_KIT['GRAPHS'].")
    title = models.CharField(max_length=255, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [models.Index(fields=["user", "-updated_at"])]

    def __str__(self) -> str:
        return self.title or str(self.id)

    @property
    def thread_id(self) -> str:
        return str(self.id)

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}
