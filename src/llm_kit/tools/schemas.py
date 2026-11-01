"""Argument schemas the model sees for generated tools. Kept small: they are sent every turn."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, create_model

Id = int | str

_INTS = {
    "IntegerField",
    "PositiveIntegerField",
    "PositiveSmallIntegerField",
    "BigIntegerField",
    "PositiveBigIntegerField",
    "SmallIntegerField",
}
DATE_TYPES = {"DateField", "DateTimeField"}


def python_type(field):
    if field.choices:
        return Literal[tuple(value for value, _ in field.flatchoices)]
    kind = field.get_internal_type()
    if kind == "BooleanField":
        return bool
    if kind in _INTS:
        return int
    if kind == "FloatField":
        return float
    if kind == "DecimalField":
        return float | str
    if field.is_relation:
        return Id
    return str


def hint(field) -> str:
    text = str(field.help_text or field.verbose_name)
    if field.is_relation:
        text += f" (the id of a {field.related_model._meta.verbose_name})"
    elif field.get_internal_type() == "DateField":
        text += " (YYYY-MM-DD)"
    elif field.get_internal_type() == "DateTimeField":
        text += " (ISO 8601)"
    return text


def is_date(field) -> bool:
    return field.get_internal_type() in DATE_TYPES


def find_schema(spec):
    fields = {
        "id": (Id | None, Field(None, description="Look one up by its id and show it in full.")),
        "query": (str, Field("", description="Words to look for in " + ", ".join(spec.search_fields))),
        "limit": (int, Field(spec.limit, description=f"How many to show, at most {spec.max_limit}.")),
        "offset": (int, Field(0, description="Skip this many, to see the next page.")),
    }
    if not spec.search_fields:
        del fields["query"]
    for name in spec.filter_fields:
        field = spec.field(name)
        if is_date(field):
            fields[f"{name}_from"] = (str | None, Field(None, description=f"{hint(field)}: on or after"))
            fields[f"{name}_to"] = (str | None, Field(None, description=f"{hint(field)}: on or before"))
        else:
            fields[name] = (python_type(field) | None, Field(None, description=hint(field)))
    return create_model(f"Find{spec.model.__name__}Args", **fields)


def save_schema(spec):
    fields = {}
    if spec.updates:
        fields["id"] = (
            Id | None,
            Field(
                None, description="Give an id to change that one." + (" Leave empty to create." if spec.creates else "")
            ),
        )
    for name in spec.saved_fields:
        field = spec.field(name)
        required = spec.creates and not field.blank and not field.has_default()
        text = hint(field) + (" (required when creating)" if required else "")
        if name in spec.approval_fields and spec.writes != "approval":
            text += " (needs a person's approval)"
        fields[name] = (python_type(field) | None, Field(None, description=text))
    return create_model(f"Save{spec.model.__name__}Args", **fields)
