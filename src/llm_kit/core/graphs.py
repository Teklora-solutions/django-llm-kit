"""Graph registry: name -> compiled LangGraph graph with a checkpointer attached.

Each registered target is a dotted path (``"pkg.mod:attr"`` or
``"pkg.mod.attr"``), or the object itself, and is one of:

* an uncompiled ``StateGraph`` builder -> compiled with the checkpointer
* a compiled graph -> copied with the checkpointer attached (unless it has one)
* a callable ``fn(checkpointer) -> compiled graph`` (e.g. wrapping ``create_agent``)
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Mapping
from typing import Any

from langgraph.graph import StateGraph
from langgraph.pregel import Pregel

from llm_kit.core.errors import ConfigError, UnknownGraph


def import_target(path: str) -> Any:
    """Import ``"pkg.mod:attr"`` or ``"pkg.mod.attr"``."""
    module_path, sep, attr = path.partition(":")
    if not sep:
        module_path, _, attr = path.rpartition(".")
    if not module_path or not attr:
        raise ConfigError(f"{path!r} is not a dotted path like 'pkg.module:attr'.")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ConfigError(f"Cannot import {module_path!r} for {path!r}: {e}") from e
    obj = module
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as e:
            raise ConfigError(f"{module_path!r} has no attribute {attr!r}.") from e
    return obj


def compile_target(name: str, target: Any, checkpointer: Any) -> Pregel:
    if isinstance(target, str):
        target = import_target(target)
    if isinstance(target, StateGraph):
        return target.compile(checkpointer=checkpointer)
    if isinstance(target, Pregel):
        return target if target.checkpointer else target.copy(update={"checkpointer": checkpointer})
    if callable(target):
        graph = target(checkpointer)
        if not isinstance(graph, Pregel):
            raise ConfigError(f"Graph factory for {name!r} returned {type(graph).__name__}, not a compiled graph.")
        return graph
    raise ConfigError(f"Graph {name!r} must be a StateGraph, a compiled graph, or a factory.")


class GraphRegistry:
    """Compiles and caches graphs on first use.

    Args:
        targets: zero-argument callable returning the current ``{name: target}``
            mapping (read each time, so settings overrides take effect).
        checkpointer: zero-argument callable returning the checkpointer to attach.
        default: zero-argument callable returning the default graph name, if any.
    """

    def __init__(
        self,
        targets: Callable[[], Mapping[str, Any]],
        checkpointer: Callable[[], Any],
        default: Callable[[], str | None] = lambda: None,
    ) -> None:
        self._targets = targets
        self._checkpointer = checkpointer
        self._default = default
        self._cache: dict[str, Pregel] = {}
        self._lock = threading.Lock()

    def names(self) -> list[str]:
        return list(self._targets())

    def default_name(self) -> str | None:
        names = self.names()
        return self._default() or (names[0] if len(names) == 1 else None)

    def get(self, name: str | None = None) -> Pregel:
        name = name or self.default_name()
        targets = self._targets()
        if not name or name not in targets:
            raise UnknownGraph(f"Unknown graph {name!r}. Registered: {', '.join(targets) or 'none'}.")
        if name not in self._cache:
            with self._lock:
                if name not in self._cache:
                    self._cache[name] = compile_target(name, targets[name], self._checkpointer())
        return self._cache[name]

    def clear(self) -> None:
        self._cache.clear()
