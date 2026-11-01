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

import dataclasses
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from django.conf import settings
from langchain_core.messages import AIMessageChunk, BaseMessage
from langgraph.types import Command, Interrupt, StateSnapshot
from pydantic import BaseModel

from llm_kit import conf
from llm_kit.catalogue import UnknownModel, resolve

logger = logging.getLogger("llm_kit")


class BadRequest(ValueError):
    pass


# ------------------------------------------------------------- serialization


def message_to_json(msg: BaseMessage, compact: bool = False) -> dict[str, Any]:
    if compact:
        out: dict[str, Any] = {"id": msg.id, "type": msg.type, "content": msg.content}
        if getattr(msg, "tool_call_chunks", None):
            out["tool_call_chunks"] = to_json(msg.tool_call_chunks)
        if getattr(msg, "tool_calls", None):
            out["tool_calls"] = to_json(msg.tool_calls)
        return out
    return to_json(msg.model_dump(exclude_none=True))


def to_json(obj: Any) -> Any:
    """Convert LangChain / LangGraph objects into JSON-serializable data."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, BaseMessage):
        return message_to_json(obj)
    if isinstance(obj, Interrupt):
        return {"id": obj.id, "value": to_json(obj.value)}
    if isinstance(obj, BaseModel):
        return to_json(obj.model_dump())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_json(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_json(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def state_to_json(state: StateSnapshot) -> dict[str, Any]:
    cfg = (state.config or {}).get("configurable", {})
    parent = (state.parent_config or {}).get("configurable", {})
    return {
        "checkpoint_id": cfg.get("checkpoint_id"),
        "parent_checkpoint_id": parent.get("checkpoint_id"),
        "values": to_json(state.values),
        "next": list(state.next),
        "metadata": to_json(state.metadata),
        "created_at": state.created_at,
        "interrupts": [to_json(i) for i in state.interrupts],
        "tasks": [{"id": t.id, "name": t.name, "error": str(t.error) if t.error else None} for t in state.tasks],
    }


# ------------------------------------------------------------------ payloads


def build_input(payload: dict[str, Any]) -> Any:
    if "resume" in payload:
        return Command(resume=payload["resume"])
    if "message" in payload:
        content = payload["message"]
        if not isinstance(content, (str, list)):
            raise BadRequest("'message' must be a string or a list of content blocks.")
        return {"messages": [{"role": "user", "content": content}]}
    return payload.get("input")


def build_config(thread, payload: dict[str, Any], user=None) -> dict[str, Any]:
    extra = payload.get("config") or {}
    if not isinstance(extra, dict):
        raise BadRequest("'config' must be an object.")
    configurable = dict(extra.get("configurable") or {})
    model = payload.get("model")
    if model is not None:
        if not isinstance(model, str):
            raise BadRequest("'model' must be a string.")
        configurable["model"] = model
    configurable["thread_id"] = thread.thread_id  # never let the client pick another thread
    config: dict[str, Any] = {"configurable": configurable}
    if "recursion_limit" in extra:
        config["recursion_limit"] = int(extra["recursion_limit"])
    metadata = {"llm_kit_thread": thread.thread_id}
    if user is not None and getattr(user, "is_authenticated", False):
        metadata["user_id"] = str(user.pk)
    config["metadata"] = metadata
    return config


def validate_model(name: str | None) -> None:
    """Raise BadRequest if a requested model is not resolvable (sync; call via sync_to_async)."""
    if name:
        try:
            resolve(name)
        except UnknownModel as e:
            raise BadRequest(str(e)) from e


def error_message(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}" if settings.DEBUG else "The run failed."


# ------------------------------------------------------------------- running


def _interrupts(chunk: dict) -> list[dict]:
    return [to_json(i) for i in chunk.get("__interrupt__", ())]


async def astream_run(graph, input: Any, config: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """Run ``graph`` and yield llm_kit events. Errors become an ``error`` event."""
    modes = list(conf.get("STREAM_MODES"))
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
