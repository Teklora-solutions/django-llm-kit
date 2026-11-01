"""Django adapter for the graph registry (``LLM_KIT["GRAPHS"]`` and declared agents), with ``LLM_KIT["CHECKPOINTER"]``.

See ``llm_kit.core.graphs`` for what a registered target may be.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.pregel import Pregel

from llm_kit import conf
from llm_kit.core.errors import UnknownGraph
from llm_kit.core.graphs import GraphRegistry
from llm_kit.persistence.saver import DjangoSaver

__all__ = ["UnknownGraph", "clear_cache", "default_graph_name", "get_checkpointer", "get_graph", "graph_names"]

_saver: BaseCheckpointSaver | None = None
_saver_setting: str | None = None  # the CHECKPOINTER value _saver was built from


def build_checkpointer(value) -> BaseCheckpointSaver:
    """``LLM_KIT["CHECKPOINTER"]``: None/"django" (``DjangoSaver``), "postgres", or anything
    ``llm_kit.core.checkpointers.build_checkpointer`` accepts."""
    if value in (None, "django"):
        return DjangoSaver()
    from llm_kit.core.checkpointers import build_checkpointer as build

    if value == "postgres":
        from llm_kit.persistence.postgres import SAVERS

        value = SAVERS
    return build(value)


def get_checkpointer() -> BaseCheckpointSaver:
    """The checkpointer every graph and agent is compiled with (one per process)."""
    global _saver, _saver_setting
    value = conf.get("CHECKPOINTER")
    if _saver is None or repr(value) != _saver_setting:  # rebuilt only when the setting changes
        _saver, _saver_setting = build_checkpointer(value), repr(value)
    return _saver


def _targets() -> dict:
    """Declared agents are graphs by name, alongside ``LLM_KIT["GRAPHS"]``."""
    from llm_kit.agents.registry import graph_targets

    return {**graph_targets(), **conf.get("GRAPHS")}


registry = GraphRegistry(
    targets=_targets,
    checkpointer=get_checkpointer,
    default=lambda: conf.get("DEFAULT_GRAPH"),
)


def graph_names() -> list[str]:
    return registry.names()


def default_graph_name() -> str | None:
    return registry.default_name()


def touchpoint_name(graph: str) -> str | None:
    return registry.touchpoint_name(graph)


def get_graph(name: str | None = None) -> Pregel:
    """Return the compiled graph registered as ``name`` (cached per process)."""
    return registry.get(name)


def clear_cache() -> None:
    """Drop compiled graphs, e.g. after changing settings in tests.

    The checkpointer is kept (it may hold connection pools); it is rebuilt when
    ``CHECKPOINTER`` changes."""
    registry.clear()
