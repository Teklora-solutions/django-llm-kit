"""Runs started in the background (``llm_kit.background``), detached from any request."""

from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models

WAITING = "waiting"  # queued behind the thread's current run (start(..., queue=True))
ACTIVE = ("queued", "running")  # at most one per thread
FINISHED = ("interrupted", "done", "error", "cancelled", "lost")
STATUS = [(s, s) for s in (WAITING, *ACTIVE, *FINISHED)]


GRAPH, MEDIA = "graph", "media"
KINDS = [(GRAPH, "graph"), (MEDIA, "media")]


class BackgroundRun(models.Model):
    """One run that keeps going without the request that started it: a run of a thread's graph
    (``kind="graph"``), or a media job such as a video (``kind="media"``, no thread).

    Its ``id`` is also the recorded ``LLMRun.run_id`` (a media job's calls have it as ``run_id``).
    At most one is queued or running per thread; others may be ``waiting`` behind it."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    kind = models.CharField(max_length=10, choices=KINDS, default=GRAPH, db_index=True)
    thread = models.ForeignKey(
        "llm_kit.Thread", null=True, blank=True, on_delete=models.CASCADE, related_name="background_runs"
    )
    graph = models.CharField(max_length=100, blank=True)
    status = models.CharField(max_length=12, choices=STATUS, default="queued", db_index=True)
    # waiting: submitted to the executor once the thread is free; queued/running: submitted.
    payload = models.JSONField(default=dict, blank=True, help_text="The run payload (message, input, resume...).")
    # Server-set facts for run hooks (start(..., context=)); never from a request.
    context = models.JSONField(default=dict, blank=True, help_text="Server-set context for run hooks.")
    # Who the run acts for, as when it was started.
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.CASCADE, related_name="llm_background_runs"
    )
    session_key = models.CharField(max_length=64, blank=True, default="")
    tenant = models.CharField(max_length=64, blank=True, default="")
    executor = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    heartbeat_at = models.DateTimeField(null=True, blank=True)
    cancel_requested = models.BooleanField(default=False)
    # A media job's provider said it changed (a webhook): check it now instead of at the next poll.
    notified_at = models.DateTimeField(null=True, blank=True)
    error = models.CharField(max_length=500, blank=True)
    result = models.JSONField(null=True, blank=True, help_text="The end event's data: next, interrupts.")
    # A subagent started in the background (delegate(..., background=true)): where its answer goes,
    # the agent run that started it, the agents above it and the limits it inherited.
    parent_thread = models.ForeignKey(
        "llm_kit.Thread", null=True, blank=True, on_delete=models.SET_NULL, related_name="background_tasks"
    )
    parent_run_id = models.UUIDField(null=True, blank=True, db_index=True)
    agents = models.JSONField(default=list, blank=True, help_text="Agent names from the root agent down.")
    limits = models.JSONField(null=True, blank=True, help_text="budget (USD left), delegations, max_depth.")
    # The run that tells the parent agent a background task's result.
    reports_on = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL, related_name="reports")

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "background run"
        indexes = [models.Index(fields=["thread", "-created_at"])]
        constraints = [
            models.UniqueConstraint(
                fields=["thread"],
                condition=models.Q(status__in=ACTIVE),
                name="llm_kit_one_active_background_run_per_thread",
            )
        ]

    def __str__(self) -> str:
        what = self.graph if self.kind == GRAPH else f"media {(self.payload or {}).get('kind', '')}"
        return f"{what} {self.status} @ {self.created_at:%Y-%m-%d %H:%M:%S}"

    @property
    def run_id(self) -> str:
        return str(self.id)

    @property
    def active(self) -> bool:
        """Not finished yet: waiting, queued or running."""
        return self.status == WAITING or self.status in ACTIVE

    def cancel(self) -> bool:
        """Ask the run to stop (it stops within a couple of seconds wherever it runs)."""
        from llm_kit.background import service

        return service.cancel(self)

    def to_json(self) -> dict:
        return {
            "id": self.run_id,
            "kind": self.kind,
            "thread": self.thread_id and str(self.thread_id),
            "graph": self.graph,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "cancel_requested": self.cancel_requested,
            "error": self.error,
            "result": self.result,
            "parent_thread": str(self.parent_thread_id) if self.parent_thread_id else None,
            "reports_on": str(self.reports_on_id) if self.reports_on_id else None,
        }
