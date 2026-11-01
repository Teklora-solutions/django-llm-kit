"""The tool catalogue: generated model tools plus hand-written ones, filtered per caller.

Loaded once at start-up (``LLMKitConfig.ready``): every model with an inner
``AgentAccess`` class, every ``expose()`` call and every ``register()`` call made in
an app's ``agent_tools.py``. A bad declaration raises ``ImproperlyConfigured`` there.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from asgiref.sync import sync_to_async
from django.apps import apps
from django.core.exceptions import ImproperlyConfigured
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools import tool as as_tool

from llm_kit.tools import ops
from llm_kit.tools.caller import _UNSET, CONFIG_KEY, PUBLIC, Caller, caller_for, caller_from
from llm_kit.tools.schemas import find_schema, save_schema
from llm_kit.tools.scope import scoped
from llm_kit.tools.spec import PROTECTED_FIELD_NAMES, AccessSpec

NO_CALLER = "Refused: no signed-in user was given for this tool call, so it can't run."


@dataclass
class ToolEntry:
    name: str
    description: str
    category: str
    kind: str  # "find" | "save" | "custom"
    permissions: tuple  # allowed if the caller holds any of them
    args_schema: Any
    guidance: str = ""
    spec: AccessSpec | None = None
    inner: BaseTool | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def writes(self) -> str:
        return self.spec.writes if self.kind == "save" else "none"

    def allowed(self, caller: Caller | None) -> bool:
        return caller is not None and any(caller.has_perm(p) for p in self.permissions)

    def refusal(self, caller: Caller | None) -> str | None:
        if caller is None:
            return NO_CALLER
        if self.allowed(caller):
            return None
        if not caller.is_authenticated:
            return NO_CALLER
        return f"Refused: you don't have permission to use {self.name}."

    def _inner_config(self, caller, config):
        config = dict(config or {})
        # Hand-written tools can read the caller with tools.caller_from(config); no second ToolCall row.
        config["configurable"] = {**(config.get("configurable") or {}), CONFIG_KEY: caller}
        config["callbacks"] = []
        return config

    def execute(self, caller: Caller, config, kwargs: dict) -> Any:
        if self.kind == "find":
            return ops.find(self.spec, caller, **kwargs)
        if self.kind == "save":
            return ops.save(self.spec, caller, config, **kwargs)
        return self.inner.invoke(kwargs, self._inner_config(caller, config))

    def tool(self, caller: Caller | None = None) -> StructuredTool:
        """A LangChain tool. Bound to ``caller``, or reading the caller from the run's config."""
        entry = self

        def run(config: RunnableConfig, **kwargs):
            who = caller or caller_from(config)
            return entry.refusal(who) or entry.execute(who, config, kwargs)

        async def arun(config: RunnableConfig, **kwargs):
            who = caller or caller_from(config)
            if entry.kind == "custom":
                refusal = await sync_to_async(entry.refusal, thread_sensitive=True)(who)
                return refusal or await entry.inner.ainvoke(kwargs, entry._inner_config(who, config))
            return await sync_to_async(run, thread_sensitive=True)(config, **kwargs)

        return StructuredTool(
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
            func=run,
            coroutine=arun,
            metadata={"llm_kit_category": self.category, "llm_kit_kind": self.kind, **self.metadata},
        )


_specs: dict[type, AccessSpec] = {}
_entries: dict[str, ToolEntry] = {}
_queued: list[Callable[[], None]] = []
_loaded = False


# ------------------------------------------------------------------ declaring


def expose(model, declaration: Any = None, **options) -> None:
    """Declare agent access for a model you can't edit (``User``, a third-party app's model).

    Takes the same options as an inner ``AgentAccess`` class, as keywords or as a class.
    """
    if declaration is not None and options:
        raise TypeError("pass a declaration class or keyword options, not both")

    def add():
        if getattr(model, "AgentAccess", None) is not None:
            raise ImproperlyConfigured(f"{model._meta.label} already declares AgentAccess; don't also expose() it")
        _add_spec(AccessSpec(model, declaration if declaration is not None else options, source="expose()"))

    _when_loaded(add)


def register(
    tool: BaseTool | Callable,
    *,
    permission: Any,
    category: str = "custom",
    guidance: str = "",
) -> BaseTool:
    """Add a hand-written tool to the catalogue, behind a permission (``"app.codename"``,
    ``tools.AUTHENTICATED``, ``tools.PUBLIC`` (anyone, including visitors and runs with no
    user), or a callable taking the user). Usable as a decorator::

        @functools.partial(tools.register, permission="crm.view_lead", category="crm")
        @langchain_core.tools.tool
        def lead_score(lead_id: int, config: RunnableConfig) -> str: ...
    """
    inner = tool if isinstance(tool, BaseTool) else as_tool(tool)
    if permission is None:
        raise ImproperlyConfigured(
            f"tool {inner.name!r}: give a permission (tools.AUTHENTICATED for any signed-in user, "
            "tools.PUBLIC for anyone)"
        )

    def add():
        _add_entry(
            ToolEntry(
                name=inner.name,
                description=inner.description,
                category=category,
                kind="custom",
                permissions=(permission,),
                args_schema=inner.args_schema,
                guidance=guidance,
                inner=inner,
            )
        )

    _when_loaded(add)
    return inner


def _when_loaded(fn: Callable[[], None]) -> None:
    if _loaded:
        fn()
        _check_links()
    else:
        _queued.append(fn)


def _add_entry(entry: ToolEntry) -> None:
    if entry.kind != "custom" and PUBLIC in entry.permissions:
        raise ImproperlyConfigured(f"tool {entry.name!r}: tools on your models can't be PUBLIC; give a permission")
    if entry.name in _entries:
        raise ImproperlyConfigured(
            f"two tools are named {entry.name!r}; set name= or plural= in one of the declarations"
        )
    _entries[entry.name] = entry


def _add_spec(spec: AccessSpec) -> None:
    if spec.model in _specs:
        raise ImproperlyConfigured(f"{spec.label} is exposed to agents twice")
    _specs[spec.model] = spec
    if spec.paused:
        return
    category = spec.model._meta.app_label
    plural = spec.model._meta.verbose_name_plural
    label = spec.model._meta.verbose_name
    _add_entry(
        ToolEntry(
            name=spec.find_name,
            description=(
                f"Look up {plural}: {spec.description}. Shows {', '.join(spec.fields)}. Give an id to see one in full."
            ),
            category=category,
            kind="find",
            permissions=(spec.permission,),
            args_schema=find_schema(spec),
            guidance=spec.guidance_text
            or f"- For questions about {spec.description}, use `{spec.find_name}`; look one up by id before "
            "describing it in detail.",
            spec=spec,
            metadata={"llm_kit_model": spec.label},
        )
    )
    if spec.writes == "none":
        return
    verbs = " or ".join(v for v, ok in (("create one", spec.creates), ("change one by id", spec.updates)) if ok)
    read_only = [n for n in spec.fields if n not in spec.saved_fields and n in PROTECTED_FIELD_NAMES]
    notes = ""
    if spec.writes == "approval":
        notes += " Every save is a proposal: nothing changes until a person approves it."
    elif spec.approval_fields:
        notes += f" Changing {' or '.join(spec.approval_fields)} is a proposal a person must approve."
    if read_only:
        notes += f" It cannot change {' or '.join(read_only)}: a person does that, so say so instead of saving."
    _add_entry(
        ToolEntry(
            name=spec.save_name,
            description=(
                f"Save {_a(label)} ({spec.description}): {verbs}. Can set {', '.join(spec.saved_fields)}; "
                f"fields left out are not changed.{notes}"
            ),
            category=category,
            kind="save",
            permissions=tuple(
                p for p, ok in ((spec.add_permission, spec.creates), (spec.change_permission, spec.updates)) if ok
            ),
            args_schema=save_schema(spec),
            guidance=f"- Only call `{spec.save_name}` when the person asked you to record or change {_a(label)}; "
            "find it first so you change the right one rather than creating a duplicate, and say what you saved.",
            spec=spec,
            metadata={"llm_kit_model": spec.label, "llm_kit_writes": spec.writes},
        )
    )


def _a(noun: str) -> str:
    return f"{'an' if noun[:1].lower() in 'aeiou' else 'a'} {noun}"


def _check_links() -> None:
    for spec in _specs.values():
        spec.check_links(_specs)


def load() -> None:
    """Read every declaration (idempotent). Raises ImproperlyConfigured on a bad one."""
    global _loaded
    if _loaded:
        return
    for model in apps.get_models():
        declaration = getattr(model, "AgentAccess", None)
        if declaration is not None:
            _add_spec(AccessSpec(model, declaration))
    queued, _queued[:] = list(_queued), []
    for fn in queued:
        fn()
    _check_links()
    _loaded = True


def reset() -> None:
    """Forget everything (tests). The next use reloads model declarations; expose()/register() calls
    made in agent_tools modules are not repeated."""
    global _loaded
    _specs.clear()
    _entries.clear()
    _queued.clear()
    _loaded = False


# ------------------------------------------------------------------ querying


def specs() -> dict[type, AccessSpec]:
    load()
    return dict(_specs)


def spec_for_label(label: str) -> AccessSpec | None:
    load()
    return next((s for s in _specs.values() if s.label == label), None)


def scoped_rows(model, caller: Caller):
    """The caller's rows of an exposed model (none for a model agents can't see)."""
    load()
    spec = _specs.get(model)
    return scoped(spec, caller) if spec is not None else model._default_manager.none()


def catalogue() -> list[ToolEntry]:
    """Every tool, whoever may use it."""
    load()
    return list(_entries.values())


def _select(names, categories, kinds) -> Iterable[ToolEntry]:
    for entry in catalogue():
        if names is not None and entry.name not in names:
            continue
        if categories is not None and entry.category not in categories:
            continue
        if kinds is not None and entry.kind not in kinds:
            continue
        yield entry


def entries_for(user, tenant: Any = _UNSET, *, names=None, categories=None, kinds=None) -> list[ToolEntry]:
    caller = caller_for(user, tenant)
    return [e for e in _select(names, categories, kinds) if e.allowed(caller)]


def for_user(user, tenant: Any = _UNSET, *, names=None, categories=None, kinds=None) -> list[StructuredTool]:
    """The tools ``user`` may use, bound to them: ``llm.bind_tools(tools.for_user(request.user))``.

    ``tenant`` defaults to ``TENANT_RESOLVER(user)``. ``kinds``: any of "find", "save", "custom".
    """
    caller = caller_for(user, tenant)
    return [e.tool(caller) for e in entries_for(caller, names=names, categories=categories, kinds=kinds)]


def for_config(config, **filters) -> list[StructuredTool]:
    """``for_user`` for the caller the server put in a run's config (none: no tools)."""
    caller = caller_from(config)
    return for_user(caller, **filters) if caller is not None else []


@dataclass(frozen=True)
class ToolSelection:
    """Part of the catalogue, for an agent's ``tools=``: called with a run's config, it returns
    the selected tools that run's caller may use."""

    names: frozenset | None = None
    categories: frozenset | None = None
    kinds: frozenset | None = None

    def __call__(self, config) -> list[StructuredTool]:
        return for_config(config, names=self.names, categories=self.categories, kinds=self.kinds)

    def entries(self) -> list[ToolEntry]:
        """Every catalogue entry it selects, whoever the caller."""
        return list(_select(self.names, self.categories, self.kinds))

    def unknown_names(self) -> list[str]:
        known = {e.name for e in catalogue()}
        return sorted(n for n in self.names or () if n not in known)


def select(*, names=None, categories=None, kinds=None) -> ToolSelection:
    """Choose catalogue tools for an agent: ``agent(..., tools=tools.select(kinds=["find"]))``.

    ``kinds``: any of "find", "save", "custom". Each run gets the selected tools its caller may use.
    """

    def frozen(value):
        return None if value is None else frozenset([value] if isinstance(value, str) else value)

    return ToolSelection(names=frozen(names), categories=frozen(categories), kinds=frozen(kinds))


def all_tools(*, names=None, categories=None, kinds=None) -> list[StructuredTool]:
    """Every tool, unbound: each call reads the caller from the run's config and checks its
    permission. For a ``ToolNode`` built once at import; bind the model with ``for_config``."""
    return [e.tool() for e in _select(names, categories, kinds)]


def guidance(tools: Iterable[BaseTool]) -> str:
    """System-prompt lines for these tools (one per tool that has guidance)."""
    names = [t.name for t in tools]
    lines = [_entries[n].guidance for n in names if n in _entries and _entries[n].guidance]
    return "\n".join(lines)
