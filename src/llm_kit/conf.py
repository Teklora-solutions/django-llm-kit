"""Django settings adapter: ``settings.LLM_KIT`` feeds ``llm_kit.core.config``.

See ``llm_kit.core.config.DEFAULTS`` for every key. ``EXPOSE_ERRORS`` defaults
to ``settings.DEBUG``.
"""

from __future__ import annotations

from typing import Any

from django.conf import settings

from llm_kit.core import config as core_config


def _source() -> dict[str, Any]:
    user = dict(getattr(settings, "LLM_KIT", {}) or {})
    user.setdefault("EXPOSE_ERRORS", settings.DEBUG)
    return user


core_config.configure(_source)

DEFAULTS = core_config.DEFAULTS


def get(key: str) -> Any:
    return core_config.get(key)
