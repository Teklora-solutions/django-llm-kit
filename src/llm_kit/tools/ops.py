"""What generated tools do: find rows, save one directly, or propose a change for approval.

Every function takes the server-set ``Caller`` and works only on that caller's rows.
Replies are short text for the model, not exceptions: a refusal the model can read
is one it can explain to the user.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from django.core.exceptions import ValidationError
from django.core.serializers.json import DjangoJSONEncoder
from django.db import models, transaction
from django.db.models import Q
from django.forms import modelform_factory
from django.forms.models import model_to_dict

from llm_kit.tools.caller import Caller
from llm_kit.tools.schemas import is_date
from llm_kit.tools.scope import OutOfScope, inject, scoped

logger = logging.getLogger("llm_kit.tools")

TEXT_PREVIEW_CHARS = 400


class _Encoder(DjangoJSONEncoder):
    def default(self, o):
        if isinstance(o, models.Model):
            return o.pk
        return super().default(o)


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, cls=_Encoder))


# ------------------------------------------------------------------ reading


def _value(obj, field):
    if field.is_relation:
        related = getattr(obj, field.name)
        return f"{related} (id {related.pk})" if related is not None else None
    raw = getattr(obj, field.attname)
    if field.choices:
        label = getattr(obj, f"get_{field.name}_display")()
        return raw if str(label) == str(raw) else f"{raw} ({label})"
    if hasattr(raw, "isoformat"):
        return raw.isoformat()
    return raw


def describe(spec, obj, full: bool = False) -> str:
    lines = [f"- id: {obj.pk}"]
    for name in spec.fields:
        value = _value(obj, spec.field(name))
        if value in (None, ""):
            continue
        text = str(value)
        if not full and len(text) > TEXT_PREVIEW_CHARS:
            text = text[:TEXT_PREVIEW_CHARS] + "…"
        lines.append(f"  {name}: {text}")
    return "\n".join(lines)


def _rows(spec, caller):
    related = [n for n in spec.fields if spec.field(n).many_to_one or spec.field(n).one_to_one]
    return scoped(spec, caller).select_related(*related).order_by(*spec.ordering)


def _get(qs, id):
    try:
        return qs.get(pk=id)
    except (qs.model.DoesNotExist, ValidationError, ValueError, TypeError):
        return None


def find(spec, caller: Caller, id=None, query: str = "", limit=None, offset: int = 0, **filters) -> str:
    plural = spec.model._meta.verbose_name_plural
    if id not in (None, ""):
        obj = _get(_rows(spec, caller), id)
        return describe(spec, obj, full=True) if obj else f"No {spec.model._meta.verbose_name} has id {id}."

    rows = _rows(spec, caller)
    if query and spec.search_fields:
        match = Q()
        for name in spec.search_fields:
            match |= Q(**{f"{name}__icontains": query})
        rows = rows.filter(match)
        if any("__" in name for name in spec.search_fields):
            rows = rows.distinct()
    for name in spec.filter_fields:
        field = spec.field(name)
        try:
            if is_date(field):
                lookup = f"{name}__date" if field.get_internal_type() == "DateTimeField" else name
                if (start := filters.get(f"{name}_from")) not in (None, ""):
                    rows = rows.filter(**{f"{lookup}__gte": start})
                if (end := filters.get(f"{name}_to")) not in (None, ""):
                    rows = rows.filter(**{f"{lookup}__lte": end})
            elif (value := filters.get(name)) not in (None, ""):
                rows = rows.filter(**{name: value})
        except (ValidationError, ValueError, TypeError) as e:
            return f"Bad value for {name}: {e}"

    try:
        total = rows.count()
    except (ValidationError, ValueError, TypeError) as e:
        return f"Bad filter value: {e}"
    if not total:
        return f"No {plural} match."
    limit = max(1, min(int(limit or spec.limit), spec.max_limit))
    offset = max(0, int(offset or 0))
    shown = list(rows[offset : offset + limit])
    if not shown:
        return f"{total} match, none after offset {offset}."
    if total > len(shown):
        end = offset + len(shown)
        head = f"{total} match; showing {offset + 1}-{end}." + (f" Use offset={end} for more." if end < total else "")
    else:
        head = f"{total} match."
    return head + "\n" + "\n".join(describe(spec, obj) for obj in shown)


# ------------------------------------------------------------------ writing


def build_form(spec, caller: Caller, instance, data: dict):
    """A ModelForm over the fields being saved; link choices limited to the caller's rows."""
    from llm_kit.tools import registry

    # Only the fields given, plus (on create) the required ones so a missing one is reported.
    # Fields left out keep their value, or their model default on create.
    names = [
        n
        for n in spec.saved_fields
        if n in data or (instance is None and not spec.field(n).blank and not spec.field(n).has_default())
    ]
    form_class = modelform_factory(spec.model, fields=names)
    form = form_class(data={**(model_to_dict(instance, fields=names) if instance else {}), **data}, instance=instance)
    for name, form_field in form.fields.items():
        field = spec.field(name)
        if field.is_relation:
            form_field.queryset = registry.scoped_rows(field.related_model, caller)
    return form


def form_errors(form) -> str:
    return "; ".join(f"{field}: {' '.join(errors)}" for field, errors in form.errors.items())


def apply(spec, caller: Caller, form, creating: bool):
    """Save in one transaction; roll back if the row ends up outside the caller's rows."""
    with transaction.atomic():
        obj = form.save(commit=False)
        if creating:
            inject(spec, obj, caller)
        obj.save()
        form.save_m2m()
        if not scoped(spec, caller).filter(pk=obj.pk).exists():
            raise OutOfScope
    return obj


def save(spec, caller: Caller, config=None, id=None, **values) -> str:
    label = spec.model._meta.verbose_name
    given = {k: v for k, v in values.items() if v is not None and k in spec.saved_fields}
    creating = id in (None, "")
    if creating and not spec.creates:
        return f"Agents can't create a {label}; give the id of one to change."
    if not creating and not spec.updates:
        return f"Agents can't change an existing {label}, only create one."
    if not caller.has_perm(spec.add_permission if creating else spec.change_permission):
        return f"Refused: you don't have permission to {'create' if creating else 'change'} a {label}."

    instance = None
    if not creating:
        instance = _get(scoped(spec, caller), id)
        if instance is None:
            return f"No {label} has id {id}; nothing was changed."
        if not given:
            return "Nothing to change: give at least one field."

    # Snapshot first: validating a ModelForm writes the new values onto the instance.
    before = jsonable(model_to_dict(instance, fields=list(given))) if instance is not None else {}
    before_repr = str(instance)[:200] if instance is not None else ""
    form = build_form(spec, caller, instance, given)
    if not form.is_valid():
        return f"Not saved. {form_errors(form)}"

    if spec.needs_approval(given):
        from llm_kit.tools.proposals import propose

        proposal = propose(spec, caller, config, instance, given, form, before, before_repr)
        changes = ", ".join(f"{k} → {proposal.after[k]}" for k in given)
        what = f"{label} {instance.pk}" if instance else f"new {label}"
        return (
            f"Proposed #{proposal.pk} for {what} ({changes}). Nothing has changed yet: a person must approve it. "
            "Tell the user it is waiting for approval."
        )

    try:
        obj = apply(spec, caller, form, creating)
    except OutOfScope:
        return f"Not saved: that would put the {label} outside the records you can see."
    logger.info("Agent %s %s %s for user %s", "created" if creating else "changed", spec.label, obj.pk, caller.user_id)
    return f"{'Created' if creating else 'Changed'} {label}.\n" + describe(spec, obj, full=True)
