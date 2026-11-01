"""Fernet encryption for ``ProviderKey`` rows (``pip install django-llm-kit[crypto]``).

Keys come from ``LLM_KIT["ENCRYPTION_KEYS"]`` or the ``LLM_KIT_ENCRYPTION_KEYS``
environment variable (comma-separated), newest first. There is deliberately no
fallback to ``SECRET_KEY``: rotating that would make stored keys unreadable.
"""

from __future__ import annotations

import base64
import hashlib
import os

from llm_kit import conf
from llm_kit.core.errors import ConfigError


def _fernet_key(raw: str) -> bytes:
    try:
        if len(base64.urlsafe_b64decode(raw.encode())) == 32:
            return raw.encode()
    except (ValueError, TypeError):
        pass
    return base64.urlsafe_b64encode(hashlib.sha256(raw.encode()).digest())


def _cipher():
    try:
        from cryptography.fernet import Fernet, MultiFernet
    except ImportError as e:
        raise ConfigError("Encrypted provider keys need 'cryptography': pip install django-llm-kit[crypto]") from e
    keys = conf.get("ENCRYPTION_KEYS") or [k for k in os.environ.get("LLM_KIT_ENCRYPTION_KEYS", "").split(",") if k]
    if isinstance(keys, str):
        keys = [keys]
    if not keys:
        raise ConfigError(
            "Set LLM_KIT['ENCRYPTION_KEYS'] (or LLM_KIT_ENCRYPTION_KEYS) to store provider keys in the DB."
        )
    return MultiFernet([Fernet(_fernet_key(k)) for k in keys])


def encrypt(plaintext: str) -> str:
    return _cipher().encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    return _cipher().decrypt(token.encode()).decode()


def rotate(token: str) -> str:
    """Re-encrypt with the newest key."""
    return _cipher().rotate(token.encode()).decode()
