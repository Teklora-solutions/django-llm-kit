"""Admin overrides for agents declared in code."""

from __future__ import annotations

from django.core.exceptions import ValidationError
from django.db import models


class AgentOverride(models.Model):
    """Change an agent without a deploy. Code stays the source: an override can narrow an
    agent (fewer tools, fewer steps, paused), never widen it. Its model is changed through
    the agent's touchpoint override."""

    PROMPT_MODES = [("append", "Append to the code prompt"), ("replace", "Replace the code prompt")]

    agent = models.CharField(max_length=200, unique=True, help_text="Agent name, e.g. crm.assistant")
    prompt = models.TextField(blank=True, help_text="Same {placeholders} as in code: {date}, {user}, {tenant}…")
    prompt_mode = models.CharField(max_length=10, choices=PROMPT_MODES, default="append")
    max_steps = models.PositiveSmallIntegerField(null=True, blank=True, help_text="Only lowers the code limit.")
    tools = models.JSONField(
        null=True, blank=True, help_text='Only these of its tools, e.g. ["find_invoices"]; empty = all of them.'
    )
    paused = models.CharField(
        max_length=255, blank=True, help_text="Why it is paused. It then answers with this instead of working."
    )
    note = models.CharField(max_length=255, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["agent"]

    def __str__(self) -> str:
        return self.agent

    def clean(self):
        if self.tools is not None and (
            not isinstance(self.tools, list) or not all(isinstance(t, str) and t for t in self.tools)
        ):
            raise ValidationError({"tools": "A list of tool names."})
        if self.max_steps == 0:
            raise ValidationError({"max_steps": "At least 1."})
