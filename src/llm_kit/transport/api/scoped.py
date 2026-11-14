"""Settings a UI shows and saves: ``GET/PATCH/DELETE llm/api/settings/``.

``GET ?scope=tenant|user|platform[&touchpoint=name]`` describes every setting that scope may
save, grouped in sections (one for settings tied to no touchpoint, one per touchpoint), each with
its current value, what it would inherit, whether the requester may edit it, and its options.

``PATCH {"scope", "key", "value", "touchpoint"?}`` saves; ``DELETE`` (same body, no value)
removes the value so it inherits again. The scope's id always comes from the request: the
signed-in user, and the tenant from ``TENANT_RESOLVER``.
"""

from __future__ import annotations

from typing import Any

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from llm_kit.core.scoped import SCOPES, Setting
from llm_kit.runtime import access
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


def _scope(value) -> str | None:
    return value if value in SCOPES else None


class SettingsView(APIView):
    permission_classes = [IsAuthenticated]

    def _owner(self, request):
        try:
            return access.owner_from_request(request)
        except access.NotAllowed as e:
            from rest_framework.exceptions import PermissionDenied

            raise PermissionDenied(str(e)) from e

    def get(self, request):
        scope = _scope(request.query_params.get("scope", "user"))
        if scope is None:
            return Response({"detail": f"scope must be one of {', '.join(SCOPES)}."}, status=400)
        owner = self._owner(request)
        if service.scope_ids(owner).get(scope) is None:
            return Response({"detail": f"You have no {scope} to configure."}, status=404)
        touchpoint = request.query_params.get("touchpoint")
        return Response(
            {
                "scope": scope,
                "editable": service.can_edit(owner, scope),
                "sections": sections(owner, scope, touchpoint),
            }
        )

    def _change(self, request, saving: bool):
        data = request.data if isinstance(request.data, dict) else {}
        scope, key, touchpoint = _scope(data.get("scope")), data.get("key"), data.get("touchpoint") or ""
        if scope is None or not isinstance(key, str):
            return Response({"detail": "Give scope and key."}, status=400)
        if saving and "value" not in data:
            return Response({"detail": "Give a value (DELETE removes one)."}, status=400)
        owner = self._owner(request)
        try:
            if saving:
                service.save(owner, scope, key, data["value"], touchpoint=touchpoint)
            else:
                service.clear(owner, scope, key, touchpoint=touchpoint)
        except service.UnknownSetting as e:
            return Response({"detail": str(e)}, status=404)
        except service.NotPermitted as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e), "key": key}, status=400)
        from llm_kit.scoped.registry import definition

        return Response(describe(definition(key, touchpoint), owner, scope), status=status.HTTP_200_OK)

    def patch(self, request):
        return self._change(request, saving=True)

    def delete(self, request):
        return self._change(request, saving=False)
