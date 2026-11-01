"""Graph registry: name -> compiled LangGraph graph wired to the Django checkpointer.

``LLM_KIT["GRAPHS"]`` maps a name to a dotted path (``"pkg.mod:attr"`` or
``"pkg.mod.attr"``) pointing at one of:

* an uncompiled ``StateGraph`` builder -> compiled with ``DjangoSaver``
* a compiled graph -> copied with ``DjangoSaver`` attached (unless it has one)
* a callable ``fn(checkpointer) -> compiled graph`` (e.g. wrapping ``create_agent``)
"""

from __future__ import annotations

import threading

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string
from langgraph.graph import StateGraph
from langgraph.pregel import Pregel

from llm_kit import conf
from llm_kit.checkpoint import DjangoSaver


class UnknownGraph(LookupError):
    pass


_cache: dict[str, Pregel] = {}
_lock = threading.Lock()
_saver: DjangoSaver | None = None


def get_checkpointer() -> DjangoSaver:
    global _saver
    if _saver is None:
        _saver = DjangoSaver()
    return _saver


def _load(path: str):
    return import_string(path.replace(":", "."))


def _build(name: str, target) -> Pregel:
    saver = get_checkpointer()
    if isinstance(target, str):
        target = _load(target)
    if isinstance(target, StateGraph):
        return target.compile(checkpointer=saver)
    if isinstance(target, Pregel):
        return target if target.checkpointer else target.copy(update={"checkpointer": saver})
    if callable(target):
        graph = target(saver)
        if not isinstance(graph, Pregel):
            raise ImproperlyConfigured(
                f"Graph factory for {name!r} returned {type(graph).__name__}, not a compiled graph."
            )
        return graph
    raise ImproperlyConfigured(f"LLM_KIT['GRAPHS'][{name!r}] must be a StateGraph, compiled graph, or factory.")


def graph_names() -> list[str]:
    return list(conf.get("GRAPHS"))


def default_graph_name() -> str | None:
    return conf.get("DEFAULT_GRAPH") or (graph_names()[0] if len(graph_names()) == 1 else None)


def get_graph(name: str | None = None) -> Pregel:
    """Return the compiled graph registered as ``name`` (cached per process)."""
    name = name or default_graph_name()
    graphs = conf.get("GRAPHS")
    if not name or name not in graphs:
        raise UnknownGraph(f"Unknown graph {name!r}. Registered: {', '.join(graphs) or 'none'}.")
    if name not in _cache:
        with _lock:
            if name not in _cache:
                _cache[name] = _build(name, graphs[name])
    return _cache[name]


def clear_cache() -> None:
    """Drop compiled graphs, e.g. after changing settings in tests."""
    _cache.clear()
