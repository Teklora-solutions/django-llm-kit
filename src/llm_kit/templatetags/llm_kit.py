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


def _base(context, scope):
    request, owner = _owner(context)
    return (
        request,
        owner,
        {
            "scope": scope,
            "csrf_token": context.get("csrf_token"),
            "next": request.get_full_path(),
            "request": request,
        },
    )


@register.inclusion_tag("llm_kit/settings/sections.html", takes_context=True)
def llm_kit_settings(context, scope="user", touchpoint=None):
    """The settings ``scope`` may save, as forms: ``{% llm_kit_settings scope="tenant" %}``."""
    from llm_kit.scoped import service
    from llm_kit.scoped.ui import sections

    request, owner, out = _base(context, scope)
    available = service.scope_ids(owner).get(scope) is not None
    out.update(
        available=available,
        editable=available and service.can_edit(owner, scope),
        sections=sections(owner, scope, touchpoint) if available else [],
    )
    return out


@register.inclusion_tag("llm_kit/settings/keys.html", takes_context=True)
def llm_kit_keys(context, scope="tenant"):
    """The scope's own provider keys, with forms to add and remove: ``{% llm_kit_keys scope="tenant" %}``."""
    from llm_kit.scoped import keys, service

    request, owner, out = _base(context, scope)
    try:
        listed, available = keys.list_keys(owner, scope), True
    except service.NotPermitted:
        listed, available = [], False
    out.update(
        available=available,
        editable=available and service.can_edit(owner, scope),
        keys=listed,
        providers=keys.suggested_providers(),
    )
    return out


@register.inclusion_tag("llm_kit/settings/budgets.html", takes_context=True)
def llm_kit_budgets(context, scope="tenant"):
    """The scope's budgets with what was spent, and a form to set one: ``{% llm_kit_budgets scope="member" %}``."""
    from llm_kit.core.touchpoints import all_touchpoints
    from llm_kit.scoped import budgets, service

    request, owner, out = _base(context, scope)
    try:
        listed, available = budgets.list_for(owner, scope), True
    except service.NotPermitted:
        listed, available = [], False
    out.update(
        available=available,
        editable=available and budgets.can_set(owner, scope),
        budgets=listed,
        periods=budgets.periods(),
        touchpoints=[t.name for t in all_touchpoints()],
    )
    return out
