"""Settings that tenants and users may save, declared by the developer.

Scopes, broadest first: ``platform`` (your admins), ``tenant`` (a workspace, from
``TENANT_RESOLVER``), ``user``. A narrower scope's saved value wins over a broader one's,
and every saved value stays inside what the code allows.

Per touchpoint, ``touchpoint(..., settable=Settable(...))`` says what may be saved there::

    Settable(
        model=("tenant", "user"),             # scopes that may save a model choice
        params={"temperature": Param("number", min=0, max=1, scopes=("user",))},
    )

Settings not tied to a touchpoint are registered with ``llm_kit.scoped.register(Setting(...))``
and read with ``llm_kit.scoped.value(key, caller=...)``. The Django side (``llm_kit.scoped``)
stores the values, checks who may change them, and applies them when a model is resolved.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from llm_kit.core.errors import ConfigError

SCOPES = ("platform", "tenant", "user")  # broadest first
TYPES = ("text", "number", "integer", "boolean", "choice", "model")


def _scopes(value: Any, what: str) -> tuple[str, ...]:
    scopes = (value,) if isinstance(value, str) else tuple(value or ())
    bad = [s for s in scopes if s not in SCOPES]
    if bad:
        raise ConfigError(f"{what}: unknown scope(s) {bad}; use {', '.join(SCOPES)}.")
    return scopes


@dataclass(frozen=True)
class Param:
    """A model parameter a scope may set (``temperature``, ``max_tokens``, ...)."""

    type: str = "number"  # number | integer | boolean | choice | text
    min: float | None = None
    max: float | None = None
    choices: tuple[Any, ...] = ()
    scopes: tuple[str, ...] = ("tenant", "user")
    label: str = ""
    help: str = ""

    def __post_init__(self) -> None:
        if self.type not in TYPES or self.type == "model":
            raise ConfigError(f"Param type {self.type!r}: use number, integer, boolean, choice or text.")
        object.__setattr__(self, "scopes", _scopes(self.scopes, "Param"))
        object.__setattr__(self, "choices", tuple(self.choices))


@dataclass(frozen=True)
class Settable:
    """What tenants and users may save for one touchpoint."""

    model: tuple[str, ...] = ()  # scopes that may save a model choice
    # Models a saved choice may name when the touchpoint uses ServerChoice (a ClientChoice
    # touchpoint bounds saved choices by its own allow, requires and price caps).
    model_allow: tuple[str, ...] = ("*",)
    params: dict[str, Param] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _scopes(self.model, "Settable.model"))
        allow = (self.model_allow,) if isinstance(self.model_allow, str) else tuple(self.model_allow)
        object.__setattr__(self, "model_allow", allow)
        params = {}
        for name, spec in dict(self.params or {}).items():
            params[name] = spec if isinstance(spec, Param) else Param(**spec)
        object.__setattr__(self, "params", params)


@dataclass(frozen=True)
class Setting:
    """One setting a UI can show and a scope can save.

    ``choices`` may be a callable ``fn(owner) -> [values]`` for options that depend on who
    is asking. ``touchpoint`` is set on the settings generated from a touchpoint's ``Settable``."""

    key: str
    type: str = "text"
    label: str = ""
    help: str = ""
    scopes: tuple[str, ...] = ("tenant", "user")
    choices: Sequence[Any] | Callable[..., Sequence[Any]] = ()
    min: float | None = None
    max: float | None = None
    default: Any = None
    touchpoint: str = ""

    def __post_init__(self) -> None:
        if self.type not in TYPES:
            raise ConfigError(f"Setting {self.key!r}: type must be one of {', '.join(TYPES)}.")
        object.__setattr__(self, "scopes", _scopes(self.scopes, f"Setting {self.key!r}"))
        if not callable(self.choices):
            object.__setattr__(self, "choices", tuple(self.choices))

    def options(self, owner: Any = None) -> list[Any]:
        return list(self.choices(owner) if callable(self.choices) else self.choices)

    def clean(self, value: Any, owner: Any = None) -> Any:
        """The value to store, or ``ValueError`` saying why it is refused."""
        if self.type == "boolean":
            if isinstance(value, bool):
                return value
            raise ValueError("Must be true or false.")
        if self.type in ("number", "integer"):
            if isinstance(value, bool):
                raise ValueError("Must be a number.")
            try:
                number = Decimal(str(value))
            except InvalidOperation:
                raise ValueError("Must be a number.") from None
            if self.type == "integer" and number != number.to_integral_value():
                raise ValueError("Must be a whole number.")
            if self.min is not None and number < Decimal(str(self.min)):
                raise ValueError(f"Must be at least {self.min}.")
            if self.max is not None and number > Decimal(str(self.max)):
                raise ValueError(f"Must be at most {self.max}.")
            return int(number) if self.type == "integer" else float(number)
        if not isinstance(value, str):
            raise ValueError("Must be text.")
        value = value.strip()
        if self.type in ("choice", "model"):
            options = self.options(owner)
            allowed = [o["value"] if isinstance(o, dict) else o for o in options]
            if value not in allowed:
                raise ValueError("Not one of the allowed options.")
        return value


def touchpoint_settings(tp) -> list[Setting]:
    """The settings a touchpoint's ``Settable`` declares (``model``, ``param.<name>``)."""
    settable = getattr(tp, "settable", None)
    if settable is None:
        return []
    out = []
    if settable.model:
        out.append(
            Setting(
                key="model",
                type="model",
                label="Model",
                help=f"The model {tp.name} uses.",
                scopes=settable.model,
                touchpoint=tp.name,
            )
        )
    for name, p in settable.params.items():
        out.append(
            Setting(
                key=f"param.{name}",
                type=p.type,
                label=p.label or name.replace("_", " ").capitalize(),
                help=p.help,
                scopes=p.scopes,
                choices=p.choices,
                min=p.min,
                max=p.max,
                touchpoint=tp.name,
            )
        )
    return out


__all__ = ["SCOPES", "Param", "Settable", "Setting", "touchpoint_settings"]
