"""Which settings exist: ones you register, and the ones touchpoints' ``Settable`` declare."""

from __future__ import annotations

import dataclasses
import threading

from llm_kit.core.errors import ConfigError
from llm_kit.core.scoped import Setting, touchpoint_settings

_registered: dict[str, Setting] = {}
_lock = threading.Lock()


def register(setting: Setting) -> Setting:
    """Add a setting not tied to a touchpoint (re-registering a key replaces it)."""
    if setting.touchpoint:
        raise ConfigError(f"Setting {setting.key!r}: touchpoint settings come from touchpoint(settable=...).")
    if setting.key == "model" or setting.key.startswith("param."):
        raise ConfigError(f"Setting key {setting.key!r} is reserved for touchpoint settings.")
    with _lock:
        _registered[setting.key] = setting
    return setting


def unregister(key: str) -> None:
    with _lock:
        _registered.pop(key, None)


def _with_model_options(setting: Setting, tp) -> Setting:
    if setting.type != "model":
        return setting

    def options(owner=None):
        from llm_kit.scoped.options import model_options

        return model_options(tp)

    return dataclasses.replace(setting, choices=options)


def definitions(touchpoint: str | None = None, scope: str | None = None) -> list[Setting]:
    """Every setting (or those of one touchpoint; ``""`` for the ones tied to none), optionally
    only those ``scope`` may save. Registered ones first, then each touchpoint's."""
    from llm_kit.core.touchpoints import all_touchpoints

    out: list[Setting] = []
    if touchpoint in (None, ""):
        with _lock:
            out += list(_registered.values())
    for tp in all_touchpoints():
        if touchpoint not in (None, tp.name):
            continue
        out += [_with_model_options(s, tp) for s in touchpoint_settings(tp)]
    if scope is not None:
        out = [s for s in out if scope in s.scopes]
    return out


def definition(key: str, touchpoint: str = "") -> Setting | None:
    for setting in definitions(touchpoint or ""):
        if setting.key == key and setting.touchpoint == (touchpoint or ""):
            return setting
    return None
