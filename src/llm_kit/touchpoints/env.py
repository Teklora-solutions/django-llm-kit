"""``ResolutionEnv`` backed by the catalogue, ``TouchpointOverride`` rows and stored keys."""

from __future__ import annotations

from llm_kit.core.touchpoints.env import ModelInfo, Override
from llm_kit.runtime.versioned_cache import VersionedCache

_overrides = VersionedCache("touchpoint_overrides")


def _load_overrides() -> dict[str, Override]:
    from llm_kit.models import TouchpointOverride

    return {
        o.name: Override(
            default=o.default or None,
            fallbacks=o.fallbacks,
            params=o.params or {},
            allow=o.allow,
            disabled=o.disabled,
        )
        for o in TouchpointOverride.objects.all()
    }


class DjangoEnv:
    def lookup(self, ref: str) -> ModelInfo | None:
        from llm_kit import catalogue

        row = catalogue.get(ref)
        return ModelInfo(ref=row.ref, values=row.values or {}, hidden=row.hidden) if row else None

    def override(self, touchpoint: str) -> Override | None:
        return _overrides.get(_load_overrides).get(touchpoint)

    def api_keys(self, provider: str) -> list[str]:
        from llm_kit.credentials import api_keys

        return api_keys(provider)


def invalidate_overrides() -> None:
    _overrides.bump()
