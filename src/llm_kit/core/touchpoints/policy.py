"""Who may choose a touchpoint's model, declared in code."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal


@dataclass(frozen=True)
class ServerChoice:
    """The server picks the model; a client's request is ignored or rejected.

    Args:
        on_requested: ``"ignore"`` (default) runs with the server's model and
            records why; ``"reject"`` fails the run with HTTP 400.
    """

    on_requested: Literal["ignore", "reject"] = "ignore"


@dataclass(frozen=True)
class ClientChoice:
    """Clients may pick the model, within limits checked against the catalogue.

    Args:
        allow: glob patterns on ``provider/model_id`` refs, e.g. ``"anthropic/*"``.
        requires: capabilities the model must have (same keys as
            ``catalogue.search``), on top of the touchpoint's own ``requires``.
        max_price_input / max_price_output: USD per 1M tokens.
        on_disallowed: ``"reject"`` (HTTP 400) or ``"ignore"`` (use the default).
        use_fallbacks: keep the touchpoint's fallbacks behind a client's pick.
    """

    allow: tuple[str, ...] = ("*",)
    requires: dict[str, Any] = field(default_factory=dict)
    max_price_input: Decimal | float | str | None = None
    max_price_output: Decimal | float | str | None = None
    on_disallowed: Literal["reject", "ignore"] = "reject"
    use_fallbacks: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.allow, str):
            object.__setattr__(self, "allow", (self.allow,))


Choice = ServerChoice | ClientChoice


def matches_any(ref: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(fnmatch.fnmatchcase(ref, p) for p in patterns)
