"""Reading and checking a model's agent declaration.

A declaration is an inner ``class AgentAccess:`` on the model, or the keyword
arguments given to ``llm_kit.tools.expose(Model, ...)``. Both become an
``AccessSpec``, checked once when the app starts: a bad declaration stops start-up
instead of failing at the first tool call.
"""

from __future__ import annotations

import re
from typing import Any

from django.conf import settings
from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured

WRITE_MODES = ("none", "direct", "approval")

SECRET_NAME_PARTS = ("password", "passwd", "secret", "token", "api_key", "apikey", "key_hash", "private_key", "otp")
SECRET_FIELD_TYPES = {"BinaryField", "VectorField", "HalfVectorField", "SparseVectorField", "BitField"}
FILE_FIELD_TYPES = {"FileField", "ImageField", "FilePathField"}

# Decisions a person makes. They can be proposed (approval_fields), never written directly.
PROTECTED_FIELD_NAMES = {
    "status",
    "stage",
    "state",
    "visibility",
    "approved",
    "approved_by",
    "approved_at",
    "enabled",
    "is_active",
    "is_default",
    "is_staff",
    "is_superuser",
    "is_verified",
    "verified",
    "published",
    "is_published",
}

OPTIONS = {
    "name",
    "plural",
    "description",
    "guidance",
    "fields",
    "search_fields",
    "filter_fields",
    "ordering",
    "limit",
    "max_limit",
    "writes",
    "writable_fields",
    "approval_fields",
    "creates",
    "updates",
    "permission",
    "write_permission",
    "approve_permission",
    "owner_field",
    "tenant_field",
    "agent_queryset",
    "unscoped",
    "paused",
}

_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,50}$")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _is_user_model(model) -> bool:
    return model is not None and model._meta.label == settings.AUTH_USER_MODEL


def read_declaration(declaration: Any) -> dict[str, Any]:
    """Options from an ``AgentAccess`` class or a dict, unknown names refused."""
    if isinstance(declaration, dict):
        options = dict(declaration)
    else:
        options = {k: v for k, v in vars(declaration).items() if not k.startswith("_")}
        # An inner class inherited from an abstract base declares through its parents too.
        for base in getattr(declaration, "__mro__", ())[1:]:
            for k, v in vars(base).items():
                if not k.startswith("_"):
                    options.setdefault(k, v)
    return options


class AccessSpec:
    """One model's declaration, read once and checked."""

    def __init__(self, model, declaration: Any, *, source: str = "AgentAccess"):
        self.model = model
        self.source = source
        options = read_declaration(declaration)
        unknown = set(options) - OPTIONS
        if unknown:
            self._refuse(f"unknown option(s) {sorted(unknown)}; allowed: {', '.join(sorted(OPTIONS))}")
        meta = model._meta
        get = options.get

        self.name = get("name") or meta.model_name
        self.plural = get("plural") or _slug(meta.verbose_name_plural)
        self.description = get("description") or str(meta.verbose_name_plural)
        self.guidance_text = get("guidance")
        self.fields = list(get("fields") or [])
        self.search_fields = list(get("search_fields") or [])
        self.filter_fields = list(get("filter_fields") or [])
        self.ordering = list(get("ordering") or meta.ordering or ["-pk"])
        self.limit = int(get("limit") or 10)
        self.max_limit = int(get("max_limit") or 25)
        self.writable_fields = list(get("writable_fields") or [])
        self.approval_fields = list(get("approval_fields") or [])
        default_writes = "direct" if self.writable_fields else ("approval" if self.approval_fields else "none")
        self.writes = get("writes") or default_writes
        self.creates = bool(get("creates", True))
        self.updates = bool(get("updates", True))

        label = meta.app_label
        self.permission = get("permission") or f"{label}.view_{meta.model_name}"
        write = get("write_permission")
        self.add_permission = write or f"{label}.add_{meta.model_name}"
        self.change_permission = write or f"{label}.change_{meta.model_name}"
        self.approve_permission = get("approve_permission") or self.change_permission

        self.owner_field = get("owner_field")
        self.tenant_field = get("tenant_field")
        queryset = get("agent_queryset")
        self.queryset_fn = getattr(queryset, "__func__", queryset)  # staticmethod/classmethod in a class body
        self.unscoped = get("unscoped")
        self.paused = get("paused") or ""
        self._check()

    # ------------------------------------------------------------------ helpers

    @property
    def label(self) -> str:
        return self.model._meta.label

    @property
    def find_name(self) -> str:
        return f"find_{self.plural}"

    @property
    def save_name(self) -> str:
        return f"save_{self.name}"

    @property
    def saved_fields(self) -> list[str]:
        """Every field a save may touch: direct first, then approval-only."""
        return [*self.writable_fields, *(f for f in self.approval_fields if f not in self.writable_fields)]

    def needs_approval(self, names) -> bool:
        return self.writes == "approval" or any(n in self.approval_fields for n in names)

    def field(self, name: str):
        return self.model._meta.get_field(name)

    def scope_text(self) -> str:
        if self.unscoped:
            return f"unscoped ({self.unscoped})"
        parts = []
        if self.owner_field:
            parts.append(f"owner={self.owner_field}")
        if self.tenant_field:
            parts.append(f"tenant={self.tenant_field}")
        if self.queryset_fn:
            parts.append("agent_queryset()")
        return ", ".join(parts)

    def _refuse(self, why: str):
        raise ImproperlyConfigured(f"{self.model._meta.label} {self.source}: {why}")

    def _resolve(self, path: str, *, into_users: bool = False):
        """The field at the end of ``path``, following ``__`` through relations."""
        model, field = self.model, None
        for part in path.split("__"):
            if field is not None:
                if not field.is_relation or field.many_to_many:
                    self._refuse(f"{path!r} does not follow a relation")
                model = field.related_model
                if _is_user_model(model) and not into_users:
                    self._refuse(f"{path!r} reaches into user accounts")
            try:
                field = model._meta.pk if part == "pk" else model._meta.get_field(part)
            except FieldDoesNotExist:
                self._refuse(f"{path!r} is not a field")
        return field

    # ------------------------------------------------------------------ checks

    def _check(self):
        if not _TOOL_NAME.match(self.find_name) or not _TOOL_NAME.match(self.save_name):
            self._refuse(f"tool names {self.find_name!r}/{self.save_name!r} are not valid; set name= and plural=")
        for perm in (self.permission, self.add_permission, self.change_permission, self.approve_permission):
            if isinstance(perm, str) and "." not in perm:
                self._refuse(f"permission {perm!r} must be 'app_label.codename' (or a callable taking the user)")
        if not self.fields:
            self._refuse("list the fields to expose; nothing is exposed by default")
        if self.writes not in WRITE_MODES:
            self._refuse(f"writes must be one of {WRITE_MODES}, not {self.writes!r}")
        if self.writes == "none" and (self.writable_fields or self.approval_fields):
            self._refuse('writes = "none" but writable_fields/approval_fields are listed')
        if self.writes != "none" and not self.saved_fields:
            self._refuse(f'writes = "{self.writes}" needs writable_fields or approval_fields')
        if self.limit < 1 or self.max_limit < self.limit:
            self._refuse("limit must be at least 1 and not above max_limit")
        self._check_scope()
        self._check_fields()
        if self.writes != "none":
            self._check_writes()

    def _check_scope(self):
        scoped = [self.owner_field, self.tenant_field, self.queryset_fn]
        if self.unscoped is not None:
            if any(scoped):
                self._refuse("unscoped cannot be combined with owner_field, tenant_field or agent_queryset")
            if not isinstance(self.unscoped, str) or not self.unscoped.strip():
                self._refuse(
                    'unscoped must say why every caller may see every row, e.g. unscoped = "public price list"'
                )
            return
        if not any(scoped):
            self._refuse(
                "declare whose rows a caller sees: owner_field, tenant_field, agent_queryset(qs, caller), "
                'or unscoped = "<why everyone may see every row>"'
            )
        if self.queryset_fn is not None and not callable(self.queryset_fn):
            self._refuse("agent_queryset must be a function (qs, caller) -> queryset")
        if self.owner_field:
            field = self._resolve(self.owner_field, into_users=True)
            if field.primary_key and _is_user_model(field.model):
                pass  # expose(User, owner_field="pk"): a user sees only themselves
            elif not (field.is_relation and _is_user_model(field.related_model)):
                self._refuse(f"owner_field {self.owner_field!r} must lead to a link to {settings.AUTH_USER_MODEL}")
        if self.tenant_field:
            self._resolve(self.tenant_field)

    def _check_fields(self):
        for name in {*self.fields, *self.filter_fields, *self.writable_fields, *self.approval_fields}:
            if "__" in name:
                self._refuse(f"{name!r}: only search_fields may follow a relation")
        for name in {*self.fields, *self.search_fields, *self.filter_fields, *self.saved_fields}:
            field = self._resolve(name)
            if not field.concrete or field.many_to_many:
                self._refuse(f"{name!r} is not a column (many-to-many and reverse relations are not supported)")
            kind = field.get_internal_type()
            looks_secret = any(p in part for part in name.lower().split("__") for p in SECRET_NAME_PARTS)
            if kind in SECRET_FIELD_TYPES or "encrypted" in type(field).__name__.lower() or looks_secret:
                self._refuse(f"{name!r} looks like a secret, binary data or an embedding and cannot be exposed")
        for name in self.search_fields:
            if self._resolve(name).is_relation:
                self._refuse(f"search field {name!r} is a link; search a text field across it, e.g. {name}__name")

    def _check_writes(self):
        server_set = {f for f in (self.owner_field, self.tenant_field) if f}
        for name in self.saved_fields:
            field = self.field(name)
            if name in server_set:
                self._refuse(f"{name!r} is the scope field; the server sets it, so it cannot be writable")
            if name in PROTECTED_FIELD_NAMES and not self.needs_approval([name]):
                self._refuse(f"{name!r} is a decision for a person: list it in approval_fields, not writable_fields")
            if field.is_relation and _is_user_model(field.related_model):
                self._refuse(f"{name!r} links to a user and cannot be writable")
            if not field.editable or field.primary_key:
                self._refuse(f"{name!r} is not editable")
            if field.get_internal_type() in FILE_FIELD_TYPES:
                self._refuse(f"{name!r} is a file field; agents cannot upload files")
        if self.creates:
            injected = {f for f in server_set if "__" not in f}
            missing = [
                f.name
                for f in self.model._meta.concrete_fields
                if f.editable
                and not f.primary_key
                and not f.blank
                and not f.has_default()
                and f.name not in self.saved_fields
                and f.name not in injected
            ]
            if missing:
                self._refuse(
                    f"agents cannot create one: required field(s) {missing} are not writable. "
                    "Make them writable, give them defaults, or set creates = False"
                )
        if not (self.creates or self.updates):
            self._refuse('creates and updates are both False; use writes = "none"')

    def check_links(self, exposed: dict) -> None:
        """Writable links must point at exposed models, so their choices can be scoped (run after loading)."""
        for name in self.saved_fields:
            field = self.field(name)
            if field.is_relation and field.related_model not in exposed:
                self._refuse(
                    f"{name!r} links to {field.related_model._meta.label}, which agents can't see. Expose that "
                    "model too, so the choices are limited to the caller's rows"
                )
