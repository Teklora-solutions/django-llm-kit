"""Form posts from the ``{% llm_kit_settings %}``, ``{% llm_kit_keys %}`` and ``{% llm_kit_budgets %}``
templates. Each saves through the same service as the API, then redirects back to ``next``
with a message (``django.contrib.messages``, when installed)."""

from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from llm_kit.runtime import access
from llm_kit.scoped import budgets, keys, service


def _back(request):
    target = request.POST.get("next") or request.META.get("HTTP_REFERER") or "/"
    if not url_has_allowed_host_and_scheme(target, {request.get_host()}, require_https=request.is_secure()):
        target = "/"
    return HttpResponseRedirect(target)


def _say(request, level: str, text: str) -> None:
    try:
        from django.contrib import messages

        getattr(messages, level)(request, text, fail_silently=True)
    except Exception:  # messages not installed
        pass


def _run(request, action, done: str):
    try:
        action(access.owner_from_request(request))
    except (service.NotPermitted, access.NotAllowed) as e:
        _say(request, "error", str(e))
    except service.UnknownSetting as e:
        _say(request, "error", str(e))
    except ValueError as e:
        _say(request, "error", str(e))
    else:
        _say(request, "success", done)
    return _back(request)


@require_POST
@login_required
def save_setting(request):
    p = request.POST
    scope, key, touchpoint = p.get("scope", ""), p.get("key", ""), p.get("touchpoint", "")
    raw = p.get("value", "")

    def action(owner):
        if "clear" in p or raw == "":
            service.clear(owner, scope, key, touchpoint=touchpoint)
            return
        from llm_kit.scoped.registry import definition

        setting = definition(key, touchpoint)
        value = {"true": True, "false": False}.get(raw, raw) if setting and setting.type == "boolean" else raw
        service.save(owner, scope, key, value, touchpoint=touchpoint)

    return _run(request, action, "Saved.")


@require_POST
@login_required
def add_key(request):
    p = request.POST
    return _run(
        request,
        lambda owner: keys.add_key(
            owner, p.get("scope", ""), p.get("provider", ""), p.get("key", ""), p.get("label", "")
        ),
        "Key added.",
    )


@require_POST
@login_required
def remove_key(request, pk: int):
    return _run(request, lambda owner: keys.remove_key(owner, request.POST.get("scope", ""), pk), "Key removed.")


@require_POST
@login_required
def save_budget(request):
    p = request.POST

    def action(owner):
        scope, period, touchpoint = p.get("scope", ""), p.get("period", ""), p.get("touchpoint", "")
        if "remove" in p:
            budgets.remove_budget(owner, scope, period, touchpoint=touchpoint)
        else:
            budgets.set_budget(
                owner, scope, period, p.get("amount", ""), touchpoint=touchpoint, counts=p.get("counts") or "all"
            )

    return _run(request, action, "Budget saved." if "remove" not in request.POST else "Budget removed.")
