"""LangGraph checkpoint storage, used by ``llm_kit.persistence.DjangoSaver``."""

from __future__ import annotations

from django.db import models


class Checkpoint(models.Model):
    thread_id = models.CharField(max_length=255)
    checkpoint_ns = models.CharField(max_length=255, default="", blank=True)
    checkpoint_id = models.CharField(max_length=255)
    parent_checkpoint_id = models.CharField(max_length=255, null=True, blank=True)
    type = models.CharField(max_length=50)
    checkpoint = models.BinaryField()
    metadata_type = models.CharField(max_length=50)
    metadata_blob = models.BinaryField()
    # JSON copy of the metadata, for filtering (`list(filter=...)`) and the admin.
    metadata = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["thread_id", "checkpoint_ns", "checkpoint_id"],
                name="llm_kit_checkpoint_uniq",
            )
        ]
        indexes = [models.Index(fields=["thread_id", "checkpoint_ns", "-checkpoint_id"])]

    def __str__(self) -> str:
        return f"{self.thread_id}/{self.checkpoint_ns}/{self.checkpoint_id}"


class CheckpointBlob(models.Model):
    """One channel value at one version. Shared across checkpoints of a thread."""

    thread_id = models.CharField(max_length=255)
    checkpoint_ns = models.CharField(max_length=255, default="", blank=True)
    channel = models.CharField(max_length=255)
    version = models.CharField(max_length=255)
    type = models.CharField(max_length=50)
    blob = models.BinaryField(null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["thread_id", "checkpoint_ns", "channel", "version"],
                name="llm_kit_blob_uniq",
            )
        ]


class CheckpointWrite(models.Model):
    """Pending writes from a task, linked to the checkpoint they were made on."""

    thread_id = models.CharField(max_length=255)
    checkpoint_ns = models.CharField(max_length=255, default="", blank=True)
    checkpoint_id = models.CharField(max_length=255)
    task_id = models.CharField(max_length=255)
    task_path = models.CharField(max_length=1024, default="", blank=True)
    idx = models.IntegerField()
    channel = models.CharField(max_length=255)
    type = models.CharField(max_length=50)
    blob = models.BinaryField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"],
                name="llm_kit_write_uniq",
            )
        ]
        indexes = [models.Index(fields=["thread_id", "checkpoint_ns", "checkpoint_id"])]
