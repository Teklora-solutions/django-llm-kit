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

    def check_budgets(self, tp) -> None:
        """``BudgetExceeded`` when a budget limiting this caller's calls of ``tp`` is used up."""
        from llm_kit.scoped.budgets import check_all

        check_all(self.caller, tp)

    # ------------------------------------------------------------ keys (BYOK)

    def key_policy(self, tp) -> str:
        from llm_kit import conf
        from llm_kit.core.scoped import KeyPolicy

        settable = getattr(tp, "settable", None)
        policy = (settable.keys if settable is not None else None) or conf.get("KEY_POLICY") or KeyPolicy.PLATFORM_ONLY
        if policy == KeyPolicy.TENANT_CHOOSES:
            found = service.lookup("keys.policy", self.caller, scopes=("tenant",))
            chosen = found[0] if found else None
            choices = (KeyPolicy.OWN_ONLY, KeyPolicy.OWN_THEN_PLATFORM)
            policy = chosen if chosen in choices else KeyPolicy.OWN_THEN_PLATFORM
        return policy

    def own_key_sets(self, provider: str) -> list[tuple[str, list[str]]]:
        """The caller's own keys for ``provider``: the user's, then the tenant's."""
        from llm_kit import conf
        from llm_kit.core.catalogue.fields import normalise_provider
        from llm_kit.credentials.keys import scoped_keys

        allowed = conf.get("BYOK_SCOPES") or ()
        ids = service.scope_ids(self.caller)
        provider = normalise_provider(provider)
        out = []
        for scope in ("user", "tenant"):
            if scope in allowed and ids.get(scope) is not None:
                keys = scoped_keys(scope, ids[scope]).get(provider)
                if keys:
                    out.append((scope, list(keys)))
        return out

    def key_sets(self, provider: str, tp=None) -> list[tuple[str, list[str]]]:
        from llm_kit.core.scoped import KeyPolicy
        from llm_kit.core.touchpoints.env import KEYLESS_PROVIDERS

        if provider in KEYLESS_PROVIDERS:  # local models: nobody's key
            return [("platform", self.base.api_keys(provider))]
        policy = self.key_policy(tp) if tp is not None else KeyPolicy.PLATFORM_ONLY
        if policy == KeyPolicy.PLATFORM_ONLY:
            return [("platform", self.base.api_keys(provider))]
        own = self.own_key_sets(provider)
        if policy == KeyPolicy.OWN_ONLY:
            return own
        platform = self.base.api_keys(provider)
        return [*own, ("platform", platform)] if platform or not own else own

    def api_keys(self, provider: str) -> list[str]:
        """Every key a call may use (for "is there a key for this provider?" during resolution)."""
        return [k for _, keys in self.key_sets(provider, self._tp) for k in keys]

    _tp = None  # set by for_touchpoint: api_keys answers for that touchpoint's key policy

    def for_touchpoint(self, tp) -> ScopedEnv:
        env = ScopedEnv(self.base, self.caller)
        env._tp = tp
        return env

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
