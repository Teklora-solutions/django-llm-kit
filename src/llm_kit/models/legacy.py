"""v0.1 hand-edited model list. Replaced by touchpoints + the catalogue in phase 3."""

from __future__ import annotations

from django.db import models, transaction


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
