"""Django adapter for the graph registry (``LLM_KIT["GRAPHS"]``), using ``DjangoSaver``.

See ``llm_kit.core.graphs`` for what a registered target may be.
"""

from __future__ import annotations

from langgraph.pregel import Pregel

from llm_kit import conf
from llm_kit.checkpoint import DjangoSaver
from llm_kit.core.errors import UnknownGraph
from llm_kit.core.graphs import GraphRegistry

__all__ = ["UnknownGraph", "clear_cache", "default_graph_name", "get_checkpointer", "get_graph", "graph_names"]

_saver: DjangoSaver | None = None


def get_checkpointer() -> DjangoSaver:
    global _saver
    if _saver is None:
        _saver = DjangoSaver()
    return _saver


registry = GraphRegistry(
    targets=lambda: conf.get("GRAPHS"),
    checkpointer=get_checkpointer,
    default=lambda: conf.get("DEFAULT_GRAPH"),
)


def graph_names() -> list[str]:
    return registry.names()


def default_graph_name() -> str | None:
    return registry.default_name()


def get_graph(name: str | None = None) -> Pregel:
    """Return the compiled graph registered as ``name`` (cached per process)."""
    return registry.get(name)


def clear_cache() -> None:
    """Drop compiled graphs, e.g. after changing settings in tests."""
    registry.clear()
