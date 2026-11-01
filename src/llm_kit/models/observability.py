"""Recorded model calls, runs and tool calls (written by ``llm_kit.observability``)."""

from __future__ import annotations

from django.db import models

STATUS = [("ok", "ok"), ("error", "error"), ("cancelled", "cancelled"), ("interrupted", "interrupted")]


class LLMCall(models.Model):
    """One model call attempt. A failed primary and the fallback that answered are two rows."""

    call_id = models.UUIDField(unique=True)
    run_id = models.UUIDField(null=True, blank=True, db_index=True)
    started_at = models.DateTimeField(db_index=True)
    touchpoint = models.CharField(max_length=200, blank=True)
    graph = models.CharField(max_length=100, blank=True)
    node = models.CharField(max_length=200, blank=True)
    thread_id = models.CharField(max_length=255, blank=True, db_index=True)
    user = models.CharField(max_length=255, blank=True)
    tenant = models.CharField(max_length=255, blank=True)
    tags = models.JSONField(default=dict, blank=True)
    requested_model = models.CharField(max_length=255, blank=True)
    resolved_model = models.CharField(max_length=255, blank=True)
    model = models.CharField(max_length=255, blank=True, help_text="The candidate this attempt used.")
    served_model = models.CharField(max_length=255, blank=True, help_text="What the provider reported.")
    chosen_by = models.CharField(max_length=10, blank=True)
    fallback_index = models.PositiveSmallIntegerField(default=0)
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)
    cache_read_tokens = models.PositiveIntegerField(default=0)
    cache_write_tokens = models.PositiveIntegerField(default=0)
    reasoning_tokens = models.PositiveIntegerField(default=0)
    usage_estimated = models.BooleanField(default=False)
    cost = models.DecimalField(max_digits=20, decimal_places=10, null=True, blank=True)
    unpriced = models.BooleanField(default=False)
    latency_ms = models.PositiveIntegerField(default=0)
    time_to_first_token_ms = models.PositiveIntegerField(null=True, blank=True)
    streamed = models.BooleanField(default=False)
    status = models.CharField(max_length=12, choices=STATUS, default="ok")
    error_class = models.CharField(max_length=30, blank=True)
    error_message = models.CharField(max_length=500, blank=True)
    prompt_hash = models.CharField(max_length=16, blank=True)
    tools_hash = models.CharField(max_length=16, blank=True)
    provider_request_id = models.CharField(max_length=255, blank=True)
    prompt = models.JSONField(null=True, blank=True, help_text="Only when content capture is on.")
    completion = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at"]
        verbose_name = "LLM call"
        indexes = [
            models.Index(fields=["touchpoint", "-started_at"]),
            models.Index(fields=["model", "-started_at"]),
            models.Index(fields=["user", "-started_at"]),
            models.Index(fields=["tenant", "-started_at"]),
            models.Index(fields=["status", "-started_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.touchpoint or self.model} @ {self.started_at:%Y-%m-%d %H:%M:%S}"


class LLMRun(models.Model):
    """One top-level invocation: a graph run or a touchpoint ``complete()``."""

    run_id = models.UUIDField(unique=True)
    name = models.CharField(max_length=200)
    started_at = models.DateTimeField(db_index=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    graph = models.CharField(max_length=100, blank=True)
    thread_id = models.CharField(max_length=255, blank=True, db_index=True)
    user = models.CharField(max_length=255, blank=True)
    tenant = models.CharField(max_length=255, blank=True)
    tags = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=12, choices=STATUS, default="ok")
    error_class = models.CharField(max_length=30, blank=True)
    calls = models.PositiveIntegerField(default=0)
    tool_calls = models.PositiveIntegerField(default=0)
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)
    cost = models.DecimalField(max_digits=20, decimal_places=10, default=0)
    unpriced_calls = models.PositiveIntegerField(default=0)
    duration_ms = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-started_at"]
        verbose_name = "LLM run"
        indexes = [models.Index(fields=["graph", "-started_at"]), models.Index(fields=["user", "-started_at"])]

    def __str__(self) -> str:
        return f"{self.name} @ {self.started_at:%Y-%m-%d %H:%M:%S}"


class ToolCall(models.Model):
    tool_call_id = models.UUIDField(unique=True)
    run_id = models.UUIDField(null=True, blank=True, db_index=True)
    name = models.CharField(max_length=200, db_index=True)
    started_at = models.DateTimeField(db_index=True)
    duration_ms = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=12, choices=STATUS, default="ok")
    error_class = models.CharField(max_length=30, blank=True)
    input_hash = models.CharField(max_length=16, blank=True)
    output_hash = models.CharField(max_length=16, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return self.name
