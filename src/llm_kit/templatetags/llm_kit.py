from django import template

register = template.Library()


@register.filter
def pct(value, digits=1):
    """0.0455 -> "4.5%"."""
    try:
        return f"{float(value) * 100:.{int(digits)}f}%"
    except (TypeError, ValueError):
        return "—"


@register.filter
def num(value):
    """2586505 -> "2,586,505"."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return value


# ------------------------------------------------------------------ settings pages


def _owner(context):
    from llm_kit.runtime import access

    request = context.get("request")
    if request is None:
        raise template.TemplateSyntaxError("The llm-kit settings tags need 'request' in the context.")
    return request, access.owner_from_request(request)


def _base(context, scope, scope_id=None):
    """The requester, or (``scope_id``, platform admins) the tenant or user they manage."""
    from llm_kit.scoped import service

    request, owner = _owner(context)
    allowed = True
    try:
        owner = service.target(owner, scope, scope_id)
    except (service.NotPermitted, ValueError):
        allowed = False
    out = {
        "scope": scope,
        "scope_id": "" if scope_id in (None, "") else scope_id,
        "allowed": allowed,
        "csrf_token": context.get("csrf_token"),
        "next": request.get_full_path(),
        "request": request,
    }
    return request, owner, out


@register.inclusion_tag("llm_kit/settings/sections.html", takes_context=True)
def llm_kit_settings(context, scope="user", touchpoint=None, scope_id=None):
    """The settings ``scope`` may save, as forms: ``{% llm_kit_settings scope="tenant" %}``.
    Platform admins can pass ``scope_id`` to manage a given tenant or user."""
    from llm_kit.scoped import service
    from llm_kit.scoped.ui import sections

    request, owner, out = _base(context, scope, scope_id)
    available = out["allowed"] and service.scope_ids(owner).get(scope) is not None
    out.update(
        available=available,
        editable=available and service.can_edit(owner, scope),
        sections=sections(owner, scope, touchpoint) if available else [],
    )
    return out


@register.inclusion_tag("llm_kit/settings/keys.html", takes_context=True)
def llm_kit_keys(context, scope="tenant", scope_id=None):
    """The scope's own provider keys, with forms to add, test and remove: ``{% llm_kit_keys scope="tenant" %}``."""
    from llm_kit.scoped import keys, service

    request, owner, out = _base(context, scope, scope_id)
    listed, available = [], False
    if out["allowed"]:
        try:
            listed, available = keys.list_keys(owner, scope), True
        except service.NotPermitted:
            pass
    out.update(
        available=available,
        editable=available and service.can_edit(owner, scope),
        keys=listed,
        providers=keys.suggested_providers(),
    )
    return out


@register.inclusion_tag("llm_kit/settings/budgets.html", takes_context=True)
def llm_kit_budgets(context, scope="tenant", scope_id=None):
    """The scope's budgets with what was spent, and a form to set one: ``{% llm_kit_budgets scope="member" %}``.
    For ``member`` budgets, managers also see members' own budgets and (with ``TENANT_MEMBERS``)
    can pick a member for a new one."""
    from datetime import datetime

    from llm_kit.core.touchpoints import all_touchpoints
    from llm_kit.scoped import budgets, service

    request, owner, out = _base(context, scope, scope_id)
    listed, available, member_budgets, member_choices = [], False, [], None
    if out["allowed"]:
        try:
            listed, available = budgets.list_for(owner, scope), True
            if scope == "member" and budgets.can_set(owner, scope):
                for group in budgets.member_overrides(owner):
                    member_budgets += group["budgets"]
                member_choices = service.members(service.scope_ids(owner)["tenant"])
        except service.NotPermitted:
            pass
    for b in [*listed, *member_budgets]:  # for the template's date filter
        b["resets"] = datetime.fromisoformat(b["resets_at"]) if b.get("resets_at") else None
    out.update(
        available=available,
        editable=available and budgets.can_set(owner, scope),
        budgets=listed,
        member_budgets=member_budgets,
        members=_member_labels(member_choices),
        periods=budgets.periods(),
        touchpoints=[t.name for t in all_touchpoints()],
    )
    return out


def _member_labels(ids):
    """``[(id, name)]`` for a member picker (None when ``TENANT_MEMBERS`` isn't set)."""
    if ids is None:
        return None
    from django.contrib.auth import get_user_model

    users = get_user_model().objects.filter(pk__in=ids)
    return [(str(u.pk), u.get_full_name() or u.get_username()) for u in users]
