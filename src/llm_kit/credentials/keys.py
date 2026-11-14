"""Which API keys exist for a provider: your ``CREDENTIALS`` source, then active DB keys,
then environment variables."""

from __future__ import annotations

import logging

from llm_kit.core.catalogue.fields import normalise_provider
from llm_kit.core.touchpoints.env import env_api_keys
from llm_kit.runtime.versioned_cache import VersionedCache

logger = logging.getLogger("llm_kit")

_db_keys = VersionedCache("provider_keys")


def _load_db_keys() -> dict[str, list[str]]:
    from llm_kit.models import ProviderKey

    out: dict[str, list[str]] = {}
    for row in ProviderKey.objects.filter(active=True, scope=""):  # the platform's (BYOK keys: scoped_keys)
        try:
            out.setdefault(row.provider, []).append(row.get_key())
        except Exception as e:  # undecryptable row: skip, but say so
            logger.error("llm_kit: cannot decrypt ProviderKey %s (%s): %s", row.pk, row.provider, type(e).__name__)
    return out


def _from_source(provider: str) -> list[str]:
    """``LLM_KIT["CREDENTIALS"]``: ``fn(provider) -> secret | [secrets] | None``, for projects that
    keep secrets elsewhere (a settings object, a vault, a service-account file path, OAuth tokens)."""
    from llm_kit import conf
    from llm_kit.core.graphs import import_target

    path = conf.get("CREDENTIALS")
    if not path:
        return []
    source = import_target(path) if isinstance(path, str) else path
    found = source(provider)
    if not found:
        return []
    return [found] if isinstance(found, str) else [k for k in found if k]


def api_keys(provider: str) -> list[str]:
    from llm_kit.models import CatalogueProvider

    provider = normalise_provider(provider)
    keys = _from_source(provider)
    for k in _db_keys.get(_load_db_keys).get(provider, []):
        if k not in keys:
            keys.append(k)
    env_names = CatalogueProvider.objects.filter(provider=provider).values_list("env", flat=True).first() or []
    for k in env_api_keys(provider, env_names):
        if k not in keys:
            keys.append(k)
    return keys


_scoped: dict[tuple[str, str], VersionedCache] = {}


def _scoped_cache(scope: str, scope_id: str) -> VersionedCache:
    key = (scope, scope_id)
    if key not in _scoped:
        if len(_scoped) > 10_000:
            _scoped.clear()
        _scoped[key] = VersionedCache(f"provider_keys:{scope}:{scope_id}")
    return _scoped[key]


def scoped_keys(scope: str, scope_id: str) -> dict[str, list[str]]:
    """``{provider: [keys]}`` a tenant or user brought (bring your own key)."""
    from llm_kit.models import ProviderKey

    def load():
        out: dict[str, list[str]] = {}
        for row in ProviderKey.objects.filter(active=True, scope=scope, scope_id=scope_id):
            try:
                out.setdefault(normalise_provider(row.provider), []).append(row.get_key())
            except Exception as e:
                logger.error("llm_kit: cannot decrypt ProviderKey %s (%s): %s", row.pk, row.provider, type(e).__name__)
        return out

    return _scoped_cache(scope, scope_id).get(load)


def invalidate(scope: str = "", scope_id: str = "") -> None:
    if scope:
        _scoped_cache(scope, scope_id).bump()
    else:
        _db_keys.bump()
