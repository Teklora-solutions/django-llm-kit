"""django-llm-kit: LangGraph & LangChain for Django.

Most-used entry points::

    from llm_kit import get_graph, get_chat_model, DjangoSaver
"""

__version__ = "0.1.0"

__all__ = ["DjangoSaver", "get_chat_model", "get_graph", "model_from_config"]


def __getattr__(name):
    # Lazy imports so `import llm_kit` works before Django apps are loaded.
    if name == "DjangoSaver":
        from llm_kit.persistence.saver import DjangoSaver

        return DjangoSaver
    if name in ("get_chat_model", "model_from_config"):
        from llm_kit.runtime import legacy_catalogue

        return getattr(legacy_catalogue, name)
    if name == "get_graph":
        from llm_kit.runtime.graphs import get_graph

        return get_graph
    raise AttributeError(f"module 'llm_kit' has no attribute {name!r}")
