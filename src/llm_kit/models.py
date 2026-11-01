from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models, transaction


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


class LLMModel(models.Model):
    """Catalogue entry: a named, admin-editable chat model configuration.

    Resolved with ``llm_kit.get_chat_model(name)`` through LangChain's
    ``init_chat_model``. API keys are never stored here; set ``api_key_env``
    to the name of an environment variable instead.
    """

    name = models.SlugField(max_length=100, unique=True, help_text="Alias used in code and API, e.g. 'fast'.")
    model = models.CharField(
        max_length=200,
        help_text="init_chat_model id, e.g. 'anthropic:claude-sonnet-5-5' or 'openai:gpt-5'.",
    )
    provider = models.CharField(
        max_length=100,
        blank=True,
        help_text="Optional model_provider, if 'model' has no 'provider:' prefix.",
    )
    params = models.JSONField(
        default=dict,
        blank=True,
        help_text='Extra kwargs for init_chat_model, e.g. {"temperature": 0, "max_tokens": 1024}.',
    )
    api_key_env = models.CharField(
        max_length=100,
        blank=True,
        help_text="Environment variable holding the API key (passed as api_key).",
    )
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    is_default = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "LLM model"
        verbose_name_plural = "LLM models"

    def __str__(self) -> str:
        return f"{self.name} ({self.model})"

    def save(self, *args, **kwargs):
        with transaction.atomic():
            if self.is_default:
                LLMModel.objects.exclude(pk=self.pk).filter(is_default=True).update(is_default=False)
            super().save(*args, **kwargs)


# --- LangGraph checkpoint storage (used by llm_kit.checkpoint.DjangoSaver) ---


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
