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

import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessageChunk
from langgraph.types import Command

from llm_kit.core import config as settings
from llm_kit.core.errors import BadRequest
from llm_kit.core.serialize import message_to_json, state_to_json, to_json

__all__ = ["BadRequest", "astream_run", "build_config", "build_input", "error_message", "invoke_run"]

logger = logging.getLogger("llm_kit")


def build_input(payload: dict[str, Any]) -> Any:
    if "resume" in payload:
        return Command(resume=payload["resume"])
    if "message" in payload:
        content = payload["message"]
        if not isinstance(content, (str, list)):
            raise BadRequest("'message' must be a string or a list of content blocks.")
        return {"messages": [{"role": "user", "content": content}]}
    return payload.get("input")


def build_config(thread_id: str, payload: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a RunnableConfig from a client payload.

    ``thread_id`` always comes from the server: a client can never point a run
    at another thread (a bug shipped in two of the audited projects).
    """
    extra = payload.get("config") or {}
    if not isinstance(extra, dict):
        raise BadRequest("'config' must be an object.")
    # "__" keys are LangGraph internals and llm-kit's server-set caller: never from a client.
    configurable = {k: v for k, v in dict(extra.get("configurable") or {}).items() if not str(k).startswith("__")}
    model = payload.get("model")
    if model is not None:
        if not isinstance(model, str):
            raise BadRequest("'model' must be a string.")
        configurable["model"] = model
    configurable["thread_id"] = str(thread_id)
    config: dict[str, Any] = {"configurable": configurable}
    if "recursion_limit" in extra:
        config["recursion_limit"] = int(extra["recursion_limit"])
    config["metadata"] = {"llm_kit_thread": str(thread_id), **(metadata or {})}
    from llm_kit.core.observability.handler import get_handler

    if (handler := get_handler()) is not None:
        config["callbacks"] = [handler]
    return config


def error_message(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}" if settings.get("EXPOSE_ERRORS") else "The run failed."


# ------------------------------------------------------------------- running


def _interrupts(chunk: dict) -> list[dict]:
    return [to_json(i) for i in chunk.get("__interrupt__", ())]


async def astream_run(graph, input: Any, config: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """Run ``graph`` and yield llm_kit events. Errors become an ``error`` event."""
    modes = list(settings.get("STREAM_MODES"))
    try:
        async for mode, chunk in graph.astream(input, config, stream_mode=modes):
            if mode == "messages":
                msg, meta = chunk
                yield {
                    "event": "token" if isinstance(msg, AIMessageChunk) else "message",
                    "data": {"node": meta.get("langgraph_node"), "message": message_to_json(msg, compact=True)},
                }
            elif mode == "updates":
                if "__interrupt__" in chunk:
                    yield {"event": "interrupt", "data": {"interrupts": _interrupts(chunk)}}
                else:
                    yield {"event": "update", "data": to_json(chunk)}
            else:
                yield {"event": mode, "data": to_json(chunk)}
        state = await graph.aget_state(config)
        yield {
            "event": "end",
            "data": {"next": list(state.next), "interrupts": [to_json(i) for i in state.interrupts]},
        }
    except Exception as exc:  # surfaced to the client as an event
        logger.exception("llm_kit run failed for thread %s", config["configurable"].get("thread_id"))
        yield {"event": "error", "data": {"message": error_message(exc)}}


def invoke_run(graph, input: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Run ``graph`` to completion (or the next interrupt) and return its state."""
    graph.invoke(input, config)
    return state_to_json(graph.get_state(config))
