"""What a settings page shows: the data behind ``llm/api/settings/`` and ``{% llm_kit_settings %}``."""

from __future__ import annotations

from typing import Any

from llm_kit.core.scoped import Setting
from llm_kit.scoped import service
from llm_kit.scoped.registry import definitions


def describe(setting: Setting, owner, scope: str) -> dict[str, Any]:
    here = service.scope_ids(owner).get(scope)
    own = service.stored(scope, here).get((setting.touchpoint, setting.key)) if here is not None else None
    inherited = service.inherited(setting.key, owner, scope, setting.touchpoint, scopes=setting.scopes)
    entry: dict[str, Any] = {
        "key": setting.key,
        "touchpoint": setting.touchpoint,
        "type": setting.type,
        "label": setting.label or setting.key,
        "help": setting.help,
        "value": own,
        "inherited": None if inherited is None else {"value": inherited[0], "scope": inherited[1]},
        "default": setting.default,
        "editable": service.can_edit(owner, scope),
    }
    if setting.type in ("choice", "model"):
        entry["options"] = setting.options(owner)
    if setting.type in ("number", "integer"):
        entry["min"], entry["max"] = setting.min, setting.max
    return entry


def sections(owner, scope: str, touchpoint: str | None = None) -> list[dict[str, Any]]:
    """Settings ``scope`` may save, grouped for a settings page (also used by the template tag)."""
    from llm_kit.core.touchpoints import get_touchpoint

    groups: dict[str, list[Setting]] = {}
    for setting in definitions(touchpoint, scope=scope):
        groups.setdefault(setting.touchpoint, []).append(setting)
    out = []
    for name, settings_ in groups.items():
        if name:
            tp = get_touchpoint(name)
            title, description = name, tp.description
        else:
            title, description = "General", ""
        out.append(
            {
                "touchpoint": name,
                "title": title,
                "description": description,
                "settings": [describe(s, owner, scope) for s in settings_],
            }
        )
    return out
