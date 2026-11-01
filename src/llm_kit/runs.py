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
from llm_kit.catalogue import resolve
from llm_kit.core.errors import BadRequest, UnknownModel
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
    "validate_model",
]


def build_config(thread, payload: dict[str, Any], user=None) -> dict[str, Any]:
    """RunnableConfig for a run on ``thread`` (a ``llm_kit.models.Thread``)."""
    metadata = {}
    if user is not None and getattr(user, "is_authenticated", False):
        metadata["user_id"] = str(user.pk)
    return _build_config(thread.thread_id, payload, metadata)


def validate_model(name: str | None) -> None:
    """Raise BadRequest if a requested model is not resolvable (sync; call via sync_to_async)."""
    if name:
        try:
            resolve(name)
        except UnknownModel as e:
            raise BadRequest(str(e)) from e
