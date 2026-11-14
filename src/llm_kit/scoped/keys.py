"""Provider keys a tenant or user brings (BYOK): list, add, remove. Secrets are write-only."""

from __future__ import annotations

from typing import Any

from llm_kit import conf
from llm_kit.scoped.service import NotPermitted, can_edit, scope_ids


def byok_scopes() -> tuple[str, ...]:
    return tuple(conf.get("BYOK_SCOPES") or ())


def _scope_id(who, scope: str, editing: bool) -> str:
    if scope not in byok_scopes():
        raise NotPermitted(f"A {scope} can't bring its own keys here.")
    scope_id = scope_ids(who).get(scope)
    if scope_id is None:
        raise NotPermitted(f"You have no {scope}.")
    if editing and not can_edit(who, scope):
        raise NotPermitted(f"You can't change {scope} keys.")
    return scope_id


def describe(row) -> dict[str, Any]:
    return {
        "id": row.pk,
        "provider": row.provider,
        "label": row.label,
        "fingerprint": row.fingerprint,
        "active": row.active,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def list_keys(who, scope: str) -> list[dict[str, Any]]:
    from llm_kit.models import ProviderKey

    scope_id = _scope_id(who, scope, editing=False)
    return [describe(r) for r in ProviderKey.objects.filter(scope=scope, scope_id=scope_id)]


def add_key(who, scope: str, provider: str, key: str, label: str = "") -> dict[str, Any]:
    from llm_kit.core.catalogue.fields import normalise_provider
    from llm_kit.models import ProviderKey

    scope_id = _scope_id(who, scope, editing=True)
    provider = normalise_provider((provider or "").strip())
    if not provider:
        raise ValueError("Give the provider, e.g. anthropic.")
    if not isinstance(key, str) or len(key.strip()) < 8:
        raise ValueError("That doesn't look like an API key.")
    row = ProviderKey(provider=provider, label=(label or "")[:100], scope=scope, scope_id=scope_id)
    row.set_key(key)
    row.save()
    return describe(row)


def remove_key(who, scope: str, key_id: int) -> bool:
    from llm_kit.models import ProviderKey

    scope_id = _scope_id(who, scope, editing=True)
    row = ProviderKey.objects.filter(pk=key_id, scope=scope, scope_id=scope_id).first()
    if row is None:
        return False
    row.delete()
    return True


def suggested_providers() -> list[str]:
    """Providers your touchpoints may call (their defaults and fallbacks), for a key form."""
    from llm_kit.core.touchpoints import all_touchpoints
    from llm_kit.core.touchpoints.env import KEYLESS_PROVIDERS, provider_of

    found = {provider_of(ref) for tp in all_touchpoints() for ref in (tp.default, *tp.fallbacks)}
    return sorted(p for p in found if p and p not in KEYLESS_PROVIDERS)
