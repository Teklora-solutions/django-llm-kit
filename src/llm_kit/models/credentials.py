"""Provider API keys stored encrypted in the database (optional; env vars work without this)."""

from __future__ import annotations

from django.db import models


class ProviderKey(models.Model):
    provider = models.CharField(max_length=100, help_text="LiteLLM provider name, e.g. anthropic, gemini, openrouter")
    label = models.CharField(max_length=100, blank=True)
    key_encrypted = models.TextField(editable=False)
    fingerprint = models.CharField(max_length=40, editable=False, help_text="Last 4 characters and a hash prefix")
    active = models.BooleanField(default=True)
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
