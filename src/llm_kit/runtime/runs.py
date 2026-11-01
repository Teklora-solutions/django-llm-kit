"""Shared run logic for the SSE views, DRF API and Channels consumer.

Request payload (JSON), all keys optional:

    {"message": "hi"}                 # shorthand for {"input": {"messages": [user msg]}}
    {"input": {...}}                  # raw graph input
    {"resume": <value>}               # resume an interrupt: Command(resume=value)
    {"model": "fast"}                 # catalogue name, exposed as configurable.model
    {"config": {"recursion_limit": 50, "configurable": {...}}}

Every run yields events ``{"event": name, "data": json-able}``:

    token      streamed AI message chunk   {"node", "message": {id, type, content, ...}}
    message    complete message from a node
    update     node state update           {"<node>": {...}}
    interrupt  graph paused for input      {"interrupts": [{"id", "value"}]}
    end        run finished                {"next": [...], "interrupts": [...]}
    error      run failed                  {"message": "..."}
"""

from __future__ import annotations

from typing import Any

from llm_kit import conf  # noqa: F401  (wires settings.LLM_KIT into the core config)
from llm_kit.core.errors import BadRequest, LLMKitError
from llm_kit.core.runs import astream_run, build_input, error_message, invoke_run
from llm_kit.core.runs import build_config as _build_config
from llm_kit.core.serialize import message_text, message_to_json, state_to_json, to_json

__all__ = [
    "BadRequest",
    "astream_run",
    "build_config",
    "build_input",
    "error_message",
    "invoke_run",
    "message_text",
    "message_to_json",
    "state_to_json",
    "to_json",
    "validate_choice",
]


def build_config(thread, payload: dict[str, Any], user=None) -> dict[str, Any]:
    """RunnableConfig for a run on ``thread`` (a ``llm_kit.models.Thread``)."""
    from llm_kit.tools.caller import CONFIG_KEY, caller_for

    caller = caller_for(user)
    metadata = {"llm_kit_graph": thread.graph}
    if caller.is_authenticated:
        metadata["user_id"] = str(caller.user_id)
    if caller.tenant_id is not None:
        metadata["tenant"] = str(caller.tenant_id)
    config = _build_config(thread.thread_id, payload, metadata)
    # Who model tools act for: set by the server, never by the client (stripped above),
    # and never written to checkpoints (LangGraph skips "__" keys).
    config["configurable"][CONFIG_KEY] = caller
    return config


def validate_choice(graph: str, requested: str | None) -> None:
    """Check a client's ``"model"`` against the graph's touchpoint before the run starts.

    Raises BadRequest. Sync (catalogue lookups); call via ``sync_to_async`` in async code.
    """
    if not requested:
        return
    from llm_kit.core.touchpoints import get_touchpoint
    from llm_kit.runtime.graphs import touchpoint_name

    name = touchpoint_name(graph)
    if name is None:
        raise BadRequest(f"Graph {graph!r} does not accept a model choice (no touchpoint declared).")
    try:
        get_touchpoint(name).resolve(requested)
    except BadRequest:
        raise
    except LLMKitError as e:
        raise BadRequest(str(e)) from e
