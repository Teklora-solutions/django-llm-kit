"""Reading, saving and clearing scoped values, and who may change them."""

from __future__ import annotations

import threading
from typing import Any

from llm_kit import conf
from llm_kit.core.scoped import SCOPES, Setting
from llm_kit.runtime.versioned_cache import VersionedCache

_caches: dict[tuple[str, str], VersionedCache] = {}
_caches_lock = threading.Lock()


class NotPermitted(PermissionError):
    pass


class UnknownSetting(LookupError):
    pass


# ------------------------------------------------------------------ who


class Target:
    """A platform admin acting on another tenant's or user's values (``target``)."""

    def __init__(self, user, ids: dict[str, str | None]) -> None:
        self.user, self.ids = user, ids

    @property
    def tenant_id(self) -> str | None:
        return self.ids.get("tenant")

    tenant = None


def target(who, scope: str, scope_id: Any = None):
    """``who`` itself, or, with ``scope_id``, the tenant or user with that id. Only someone who may
    edit the platform scope may act on another tenant or user (``NotPermitted`` otherwise)."""
    if scope_id in (None, ""):
        return who
    if not can_edit(who, "platform"):
        raise NotPermitted("Only platform admins can manage another tenant's or user's settings.")
    ids: dict[str, str | None] = {"platform": "", "tenant": None, "user": None}
    if scope in ("tenant", "member"):
        ids["tenant"] = str(scope_id)
    elif scope == "user":
        ids["user"] = str(scope_id)
    else:
        raise ValueError(f"scope_id only applies to tenant, member and user scopes, not {scope!r}.")
    return Target(getattr(who, "user", who), ids)


def scope_ids(who) -> dict[str, str | None]:
    """The id each scope has for ``who`` (a ``Caller``, an ``Owner`` or a user): None when it has none."""
    if isinstance(who, Target):
        return dict(who.ids)
    user = getattr(who, "user", who)
    tenant = getattr(who, "tenant_id", None)
    signed_in = user is not None and bool(getattr(user, "is_authenticated", False))
    return {
        "platform": "",
        "tenant": None if tenant in (None, "") else str(tenant),
        "user": str(user.pk) if signed_in else None,
    }


def can_edit(who, scope: str) -> bool:
    """Whether ``who`` may change ``scope``'s values: ``SCOPE_PERMISSIONS[scope]`` decides
    (``fn(user, tenant) -> bool``). Without one: the platform needs the
    ``llm_kit.change_scopedsetting`` permission, a tenant nobody (configure it), a user themselves."""
    user = getattr(who, "user", who)
    if user is None or not getattr(user, "is_authenticated", False) or not getattr(user, "is_active", True):
        return False
    if scope_ids(who).get(scope) is None:
        return False
    if scope != "platform" and _platform_editor(user):  # your admins may change every scope
        return True
    hook = (conf.get("SCOPE_PERMISSIONS") or {}).get(scope)
    if hook:
        from llm_kit.core.graphs import import_target

        fn = import_target(hook) if isinstance(hook, str) else hook
        return bool(fn(user, getattr(who, "tenant", None)))
    if scope == "platform":
        return bool(user.has_perm("llm_kit.change_scopedsetting"))
    return scope == "user"


def _platform_editor(user) -> bool:
    hook = (conf.get("SCOPE_PERMISSIONS") or {}).get("platform")
    if hook:
        from llm_kit.core.graphs import import_target

        return bool((import_target(hook) if isinstance(hook, str) else hook)(user, None))
    return bool(user.has_perm("llm_kit.change_scopedsetting"))


def members(tenant_id: Any) -> list[str] | None:
    """The user ids of a tenant's members, from ``LLM_KIT["TENANT_MEMBERS"]`` (``fn(tenant_id) ->
    users or ids``), or None when it isn't set."""
    hook = conf.get("TENANT_MEMBERS")
    if not hook or tenant_id in (None, ""):
        return None
    from llm_kit.core.graphs import import_target

    found = (import_target(hook) if isinstance(hook, str) else hook)(tenant_id)
    return [str(getattr(m, "pk", m)) for m in found]


# ------------------------------------------------------------------ values


def _cache(scope: str, scope_id: str) -> VersionedCache:
    key = (scope, scope_id)
    with _caches_lock:
        if key not in _caches:
            if len(_caches) > 10_000:
                _caches.clear()
            _caches[key] = VersionedCache(f"scoped:{scope}:{scope_id}")
        return _caches[key]


def stored(scope: str, scope_id: str) -> dict[tuple[str, str], Any]:
    """``{(touchpoint, key): value}`` saved by one scope (cached; reloaded when it changes)."""

    def load():
        from llm_kit.models import ScopedSetting

        rows = ScopedSetting.objects.filter(scope=scope, scope_id=scope_id)
        return {(r.touchpoint, r.key): r.value for r in rows}

    return _cache(scope, scope_id).get(load)


def lookup(key: str, who, touchpoint: str = "", scopes=SCOPES) -> tuple[Any, str] | None:
    """The narrowest saved value for ``who`` among ``scopes``, with the scope that saved it."""
    ids = scope_ids(who)
    for scope in reversed(SCOPES):
        if scope not in scopes or ids.get(scope) is None:
            continue
        values = stored(scope, ids[scope])
        if (touchpoint, key) in values:
            return values[(touchpoint, key)], scope
    return None


def inherited(key: str, who, scope: str, touchpoint: str = "", scopes=SCOPES) -> tuple[Any, str] | None:
    """What ``scope`` would get if it saved nothing: the value of a broader scope."""
    broader = SCOPES[: SCOPES.index(scope)]
    return lookup(key, who, touchpoint, scopes=[s for s in scopes if s in broader])


def value(key: str, who=None, *, touchpoint: str = "", default: Any = None) -> Any:
    """The value ``who`` gets for a setting: their own, else their tenant's, else the platform's,
    else the setting's default (or ``default``)."""
    from llm_kit.scoped.registry import definition

    setting = definition(key, touchpoint)
    found = lookup(key, who, touchpoint, scopes=setting.scopes if setting else SCOPES) if who is not None else None
    if found is None and who is None:
        found = lookup(key, None, touchpoint, scopes=("platform",))
    if found is not None:
        return found[0]
    if setting is not None and setting.default is not None:
        return setting.default
    return default


def _setting(key: str, touchpoint: str) -> Setting:
    from llm_kit.scoped.registry import definition

    setting = definition(key, touchpoint)
    if setting is None:
        where = f" for touchpoint {touchpoint!r}" if touchpoint else ""
        raise UnknownSetting(f"There is no setting {key!r}{where}.")
    return setting


def _check(who, scope: str, setting: Setting) -> str:
    if scope not in setting.scopes:
        raise NotPermitted(f"{setting.key!r} can't be set by a {scope}.")
    if not can_edit(who, scope):
        raise NotPermitted(f"You can't change {scope} settings.")
    return scope_ids(who)[scope] or ""


def save(who, scope: str, key: str, new_value: Any, *, touchpoint: str = "") -> Any:
    """Save ``new_value`` for ``who``'s ``scope``. Raises ``UnknownSetting``, ``NotPermitted`` or
    ``ValueError`` (the value isn't allowed). Returns the stored value."""
    from llm_kit.models import ScopedSetting

    setting = _setting(key, touchpoint)
    scope_id = _check(who, scope, setting)
    cleaned = setting.clean(new_value, who)
    user = getattr(who, "user", who)
    ScopedSetting.objects.update_or_create(
        scope=scope,
        scope_id=scope_id,
        touchpoint=touchpoint or "",
        key=key,
        defaults={"value": cleaned, "updated_by": user if getattr(user, "pk", None) else None},
    )
    _cache(scope, scope_id).bump()
    return cleaned


def clear(who, scope: str, key: str, *, touchpoint: str = "") -> bool:
    """Remove ``scope``'s value so it inherits again. True if there was one."""
    from llm_kit.models import ScopedSetting

    setting = _setting(key, touchpoint)
    scope_id = _check(who, scope, setting)
    deleted, _ = ScopedSetting.objects.filter(
        scope=scope, scope_id=scope_id, touchpoint=touchpoint or "", key=key
    ).delete()
    _cache(scope, scope_id).bump()
    return bool(deleted)


def invalidate(scope: str, scope_id: str) -> None:
    _cache(scope, scope_id).bump()
