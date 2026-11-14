"""Provider API keys stored encrypted in the database (optional; env vars work without this)."""

from __future__ import annotations

from django.db import models


class ProviderKey(models.Model):
    """A provider key: the platform's (``scope`` blank), or one a tenant or user brought (BYOK)."""

    provider = models.CharField(max_length=100, help_text="LiteLLM provider name, e.g. anthropic, gemini, openrouter")
    scope = models.CharField(
        max_length=10,
        blank=True,
        choices=[("", "Platform"), ("tenant", "Tenant"), ("user", "User")],
        help_text="Blank: yours, used for everyone. Tenant/user: brought by them (bring your own key).",
    )
    scope_id = models.CharField(max_length=255, blank=True, help_text="The tenant's or user's id.")
    label = models.CharField(max_length=100, blank=True)
    key_encrypted = models.TextField(editable=False)
    fingerprint = models.CharField(max_length=40, editable=False, help_text="Last 4 characters and a hash prefix")
    active = models.BooleanField(default=True)
    last_checked_at = models.DateTimeField(null=True, blank=True, editable=False)
    last_check_ok = models.BooleanField(null=True, editable=False, help_text="Result of the last test call.")
    last_check_error = models.CharField(max_length=255, blank=True, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["provider", "label"]

    def __str__(self) -> str:
        return f"{self.provider} {self.label or self.fingerprint}".strip()

    def set_key(self, plaintext: str) -> None:
        import hashlib

        from llm_kit.credentials.crypto import encrypt

        plaintext = plaintext.strip()
        self.key_encrypted = encrypt(plaintext)
        self.fingerprint = f"…{plaintext[-4:]} {hashlib.sha256(plaintext.encode()).hexdigest()[:8]}"

    def get_key(self) -> str:
        from llm_kit.credentials.crypto import decrypt

        return decrypt(self.key_encrypted)
