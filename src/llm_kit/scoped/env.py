"""The resolution env for one caller: what their user and tenant saved is applied."""

from __future__ import annotations

from typing import Any

from llm_kit.scoped import service


class ScopedEnv:
    """Wraps an env (``DjangoEnv``); used by touchpoints with ``settable`` (``Touchpoint.env_for``)."""

    def __init__(self, base, caller) -> None:
        self.base, self.caller = base, caller

    def __getattr__(self, name: str) -> Any:  # lookup, override, api_keys, circuit_open, ...
        return getattr(self.base, name)

    def for_caller(self, caller) -> ScopedEnv:
        return ScopedEnv(self.base, caller)

    def saved_model(self, tp) -> tuple[str, str] | None:
        settable = tp.settable
        if settable is None or not settable.model:
            return None
        found = service.lookup("model", self.caller, tp.name, scopes=settable.model)
        if found is None or not isinstance(found[0], str) or not found[0]:
            return None
        return found[0], found[1]

    def saved_params(self, tp) -> dict[str, Any]:
        settable = tp.settable
        if settable is None or not settable.params:
            return {}
        from llm_kit.core.scoped import touchpoint_settings

        out = {}
        for setting in touchpoint_settings(tp):
            if not setting.key.startswith("param."):
                continue
            found = service.lookup(setting.key, self.caller, tp.name, scopes=setting.scopes)
            if found is None:
                continue
            try:
                out[setting.key[len("param.") :]] = setting.clean(found[0])
            except ValueError:  # saved before the declaration changed: ignore it
                continue
        return out
