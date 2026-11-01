"""Model catalogue tables, synced from online sources (see ``llm_kit.catalogue``)."""

from __future__ import annotations

from django.db import models


class CatalogueModel(models.Model):
    """One model at one provider, merged from catalogue sources plus overrides.

    ``provider`` uses LiteLLM's provider names, so ``ref`` (``provider/model_id``)
    is a valid LiteLLM model string. Prices are USD per 1M tokens.
    """

    provider = models.CharField(max_length=100)
    model_id = models.CharField(max_length=255)
    name = models.CharField(max_length=255, blank=True)
    family = models.CharField(max_length=100, blank=True)
    kind = models.CharField(max_length=20, blank=True, help_text="chat, embedding, image, audio, rerank, other")
    canonical_id = models.CharField(max_length=255, blank=True)
    # Comma-delimited with leading/trailing commas (",text,image,") so filters
    # like `input_modalities__contains=",image,"` work on every database.
    input_modalities = models.CharField(max_length=200, blank=True)
    output_modalities = models.CharField(max_length=200, blank=True)
    context_window = models.PositiveBigIntegerField(null=True, blank=True)
    max_input = models.PositiveBigIntegerField(null=True, blank=True)
    max_output = models.PositiveBigIntegerField(null=True, blank=True)
    tool_call = models.BooleanField(null=True)
    structured_output = models.BooleanField(null=True)
    reasoning = models.BooleanField(null=True)
    attachments = models.BooleanField(null=True)
    temperature = models.BooleanField(null=True)
    prompt_caching = models.BooleanField(null=True)
    open_weights = models.BooleanField(null=True)
    price_input = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    price_output = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    price_cache_read = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    price_cache_write = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    price_reasoning = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    price_tiers = models.JSONField(default=list, blank=True)
    release_date = models.CharField(max_length=20, blank=True)
    knowledge_cutoff = models.CharField(max_length=20, blank=True)
    status = models.CharField(max_length=20, blank=True)
    deprecation_date = models.CharField(max_length=20, blank=True)
    # Exact effective values (as merged) and what each source said.
    values = models.JSONField(default=dict)
    source_values = models.JSONField(default=dict)
    provenance = models.JSONField(default=dict)
    conflicts = models.JSONField(default=dict, blank=True)
    hidden = models.BooleanField(default=False)
    removed = models.BooleanField(default=False, help_text="No source lists this model any more.")
    first_seen = models.DateTimeField(null=True)
    last_seen = models.DateTimeField(null=True)
    removed_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["provider", "model_id"], name="llm_kit_catmodel_uniq")]
        indexes = [models.Index(fields=["kind", "removed", "hidden"]), models.Index(fields=["model_id"])]
        ordering = ["provider", "model_id"]
        verbose_name = "catalogue model"

    def __str__(self) -> str:
        return self.ref

    @property
    def ref(self) -> str:
        return f"{self.provider}/{self.model_id}"

    litellm_id = ref

    @property
    def input_modality_list(self) -> list[str]:
        return [m for m in self.input_modalities.split(",") if m]

    @property
    def output_modality_list(self) -> list[str]:
        return [m for m in self.output_modalities.split(",") if m]


class CatalogueProvider(models.Model):
    provider = models.CharField(max_length=100, primary_key=True)
    name = models.CharField(max_length=255, blank=True)
    env = models.JSONField(default=list, blank=True, help_text="Environment variables that hold this provider's key.")
    doc_url = models.URLField(max_length=500, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["provider"]

    def __str__(self) -> str:
        return self.provider

    @property
    def has_credentials(self) -> bool:
        import os

        return any(os.environ.get(e) for e in self.env)


class CatalogueOverride(models.Model):
    """Admin corrections for one model: any catalogue field, plus ``hidden``.

    A row for a model no source lists creates a private model (e.g. self-hosted).
    """

    provider = models.CharField(max_length=100)
    model_id = models.CharField(max_length=255)
    values = models.JSONField(
        default=dict,
        help_text='e.g. {"price_input": "2.5", "tool_call": true, "hidden": false}. Prices: USD per 1M tokens.',
    )
    note = models.CharField(max_length=255, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["provider", "model_id"], name="llm_kit_catoverride_uniq")]

    def __str__(self) -> str:
        return f"{self.provider}/{self.model_id}"

    def clean(self):
        from django.core.exceptions import ValidationError

        from llm_kit.core.catalogue.fields import FIELDS

        bad = [k for k in (self.values or {}) if k not in FIELDS and k != "hidden"]
        if bad:
            raise ValidationError({"values": f"Unknown fields: {', '.join(bad)}"})


class CatalogueChange(models.Model):
    synced_at = models.DateTimeField(db_index=True)
    provider = models.CharField(max_length=100)
    model_id = models.CharField(max_length=255)
    kind = models.CharField(max_length=30, db_index=True)
    field = models.CharField(max_length=50, blank=True)
    old = models.JSONField(null=True, blank=True)
    new = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ["-synced_at", "provider", "model_id"]
        indexes = [models.Index(fields=["provider", "model_id"])]

    def __str__(self) -> str:
        return f"{self.kind} {self.provider}/{self.model_id} {self.field}".strip()


class CatalogueSource(models.Model):
    name = models.CharField(max_length=100, primary_key=True)
    url = models.URLField(max_length=500)
    etag = models.CharField(max_length=255, blank=True)
    last_modified = models.CharField(max_length=100, blank=True)
    content_hash = models.CharField(max_length=64, blank=True)
    last_fetched_at = models.DateTimeField(null=True, blank=True)
    last_success_at = models.DateTimeField(null=True, blank=True)
    last_status = models.CharField(max_length=20, blank=True)
    last_error = models.TextField(blank=True)
    record_count = models.PositiveIntegerField(default=0)

    def __str__(self) -> str:
        return self.name
