"""django-llm-kit: LangGraph & LangChain for Django.

Most-used entry points::

    from llm_kit import touchpoint, ClientChoice, ServerChoice, get_graph, DjangoSaver, agent
    from llm_kit import catalogue
"""

__version__ = "0.2.0"

__all__ = [
    "AllModelsFailed",
    "ClientChoice",
    "DjangoSaver",
    "Param",
    "RetryPolicy",
    "ServerChoice",
    "Settable",
    "agent",
    "get_graph",
    "get_touchpoint",
    "touchpoint",
]

_CORE = {"touchpoint", "get_touchpoint", "ClientChoice", "ServerChoice", "RetryPolicy"}


def __getattr__(name):
    # Lazy imports so `import llm_kit` works before Django apps are loaded.
    if name in _CORE:
        from llm_kit.core import touchpoints

        return getattr(touchpoints, name)
    if name in ("Settable", "Param"):
        from llm_kit.core import scoped

        return getattr(scoped, name)
    if name == "AllModelsFailed":
        from llm_kit.core.errors import AllModelsFailed

        return AllModelsFailed
    if name == "agent":
        from llm_kit.core.agents import agent

        return agent
    if name == "DjangoSaver":
        from llm_kit.persistence.saver import DjangoSaver

        return DjangoSaver
    if name == "get_graph":
        from llm_kit.runtime.graphs import get_graph

        return get_graph
    raise AttributeError(f"module 'llm_kit' has no attribute {name!r}")
