"""Per-model rate limits of your provider accounts, set by people (``llm_kit.ratelimits``)."""

from __future__ import annotations

from django.db import models


class RateLimit(models.Model):
    """Limits for one model, or for each model matching a pattern (``gemini/*``).

    They describe your account's tier with the provider, so nothing syncs them: set them here or
    in ``LLM_KIT["RATE_LIMITS"]`` (a row wins over the setting for the same pattern)."""

    model = models.CharField(
        max_length=255, unique=True, help_text="provider/model, or a pattern each matching model gets, e.g. gemini/*"
    )
    rpm = models.PositiveIntegerField(default=0, help_text="Requests per minute; 0 = no limit.")
    tpm = models.PositiveIntegerField(default=0, help_text="Tokens per minute; 0 = no limit.")
    rpd = models.PositiveIntegerField(default=0, help_text="Requests per day (UTC); 0 = no limit.")
    note = models.CharField(max_length=255, blank=True, help_text="e.g. which account tier this is.")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["model"]
        verbose_name = "rate limit"

    def __str__(self) -> str:
        parts = [f"{n} {k}" for k, n in (("rpm", self.rpm), ("tpm", self.tpm), ("rpd", self.rpd)) if n]
        return f"{self.model}: {', '.join(parts) or 'no limits'}"
