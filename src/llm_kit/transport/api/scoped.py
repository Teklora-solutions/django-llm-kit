"""Settings a UI shows and saves: ``GET/PATCH/DELETE llm/api/settings/``.

``GET ?scope=tenant|user|platform[&touchpoint=name]`` describes every setting that scope may
save, grouped in sections (one for settings tied to no touchpoint, one per touchpoint), each with
its current value, what it would inherit, whether the requester may edit it, and its options.

``PATCH {"scope", "key", "value", "touchpoint"?}`` saves; ``DELETE`` (same body, no value)
removes the value so it inherits again. The scope's id always comes from the request: the
signed-in user, and the tenant from ``TENANT_RESOLVER``.
"""

from __future__ import annotations

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from llm_kit.core.scoped import SCOPES
from llm_kit.runtime import access
from llm_kit.scoped import service
from llm_kit.scoped.ui import describe, sections


def _scope(value) -> str | None:
    return value if value in SCOPES else None


def _who(request, scope: str | None):
    """The requester, or with ``scope_id`` (query or body) the tenant or user a platform admin is
    acting on. Refusals become 403/400 responses."""
    from rest_framework.exceptions import PermissionDenied, ValidationError

    try:
        owner = access.owner_from_request(request)
    except access.NotAllowed as e:
        raise PermissionDenied(str(e)) from e
    data = request.data if isinstance(request.data, dict) else {}
    scope_id = data.get("scope_id") or request.query_params.get("scope_id")
    try:
        return service.target(owner, scope or "", scope_id)
    except service.NotPermitted as e:
        raise PermissionDenied(str(e)) from e
    except ValueError as e:
        raise ValidationError({"detail": str(e)}) from e


class SettingsView(APIView):
    """Add ``scope_id`` (query or body) to manage another tenant's or user's values (platform admins)."""

    permission_classes = [IsAuthenticated]

    def get(self, request):
        scope = _scope(request.query_params.get("scope", "user"))
        if scope is None:
            return Response({"detail": f"scope must be one of {', '.join(SCOPES)}."}, status=400)
        owner = _who(request, scope)
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
        owner = _who(request, scope)
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


class KeysView(APIView):
    """``GET llm/api/keys/?scope=tenant`` lists the scope's own provider keys (fingerprints only);
    ``POST {"scope", "provider", "key", "label"?}`` adds one. Needs ``BYOK_SCOPES``."""

    permission_classes = [IsAuthenticated]

    def get(self, request):
        from llm_kit.scoped import keys

        scope = request.query_params.get("scope", "user")
        owner = _who(request, scope)
        try:
            listed = keys.list_keys(owner, scope)
        except service.NotPermitted as e:
            return Response({"detail": str(e)}, status=403)
        return Response(
            {
                "scope": scope,
                "editable": service.can_edit(owner, scope),
                "providers": keys.suggested_providers(),
                "keys": listed,
            }
        )

    def post(self, request):
        from llm_kit.scoped import keys

        data = request.data if isinstance(request.data, dict) else {}
        owner = _who(request, str(data.get("scope") or ""))
        try:
            row = keys.add_key(
                owner,
                str(data.get("scope") or ""),
                data.get("provider", ""),
                data.get("key"),
                data.get("label", ""),
                verify=bool(data.get("verify")),
            )
        except service.NotPermitted as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=400)
        return Response(row, status=201)


class KeyDetailView(APIView):
    """``DELETE llm/api/keys/<id>/?scope=tenant`` removes one of the scope's keys."""

    permission_classes = [IsAuthenticated]

    def delete(self, request, pk: int):
        from llm_kit.scoped import keys

        scope = request.query_params.get("scope", "user")
        owner = _who(request, scope)
        try:
            removed = keys.remove_key(owner, scope, pk)
        except service.NotPermitted as e:
            return Response({"detail": str(e)}, status=403)
        return Response(status=204 if removed else 404)


class BudgetsView(APIView):
    """Spending limits, with what has been spent: ``GET llm/api/budgets/?scope=tenant|member|user``;
    ``PUT {"scope", "period", "amount", "touchpoint"?, "counts"?, "member"?}`` sets one; ``DELETE
    {"scope", "period", "touchpoint"?, "member"?}`` removes it. Who may set each scope's budgets:
    ``BUDGET_EDITORS``. ``member`` (a user id) is one member's own budget; ``scope_id`` lets a
    platform admin act on another tenant or user."""

    permission_classes = [IsAuthenticated]

    def get(self, request):
        from llm_kit.scoped import budgets

        scope = request.query_params.get("scope", "user")
        if scope not in budgets.BUDGET_SCOPES:
            return Response({"detail": f"scope must be one of {', '.join(budgets.BUDGET_SCOPES)}."}, status=400)
        owner = _who(request, scope)
        member = request.query_params.get("member")
        try:
            listed = budgets.list_for(owner, scope, member)
        except service.NotPermitted as e:
            return Response({"detail": str(e)}, status=404)
        except ValueError as e:
            return Response({"detail": str(e)}, status=400)
        body = {
            "scope": scope,
            "periods": budgets.periods(),
            "editable": budgets.can_set(owner, scope),
            "budgets": listed,
        }
        if scope == "member" and not member and body["editable"]:
            body["members"] = service.members(service.scope_ids(owner)["tenant"])  # None: no TENANT_MEMBERS
            body["member_budgets"] = budgets.member_overrides(owner)
        return Response(body)

    def _change(self, request, saving: bool):
        from llm_kit.scoped import budgets

        data = request.data if isinstance(request.data, dict) else {}
        owner = _who(request, str(data.get("scope") or ""))
        member = data.get("member")
        scope, period, touchpoint = (
            str(data.get("scope") or ""),
            str(data.get("period") or ""),
            data.get("touchpoint") or "",
        )
        try:
            if saving:
                if "amount" not in data:
                    return Response({"detail": "Give an amount (USD)."}, status=400)
                return Response(
                    budgets.set_budget(
                        owner,
                        scope,
                        period,
                        data["amount"],
                        touchpoint=touchpoint,
                        counts=data.get("counts") or "all",
                        member=member,
                    )
                )
            removed = budgets.remove_budget(owner, scope, period, touchpoint=touchpoint, member=member)
        except service.NotPermitted as e:
            return Response({"detail": str(e)}, status=403)
        except ValueError as e:
            return Response({"detail": str(e)}, status=400)
        return Response(status=204 if removed else 404)

    def put(self, request):
        return self._change(request, saving=True)

    def delete(self, request):
        return self._change(request, saving=False)


class KeyTestView(APIView):
    """``POST llm/api/keys/<id>/test/?scope=tenant`` tries the key with one tiny call and records the result."""

    permission_classes = [IsAuthenticated]

    def post(self, request, pk: int):
        from llm_kit.scoped import keys

        scope = request.query_params.get("scope", "user")
        owner = _who(request, scope)
        try:
            result = keys.test_key(owner, scope, pk)
        except service.NotPermitted as e:
            return Response({"detail": str(e)}, status=403)
        return Response(result) if result is not None else Response(status=404)
