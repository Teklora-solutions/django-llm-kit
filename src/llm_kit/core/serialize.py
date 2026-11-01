"""Convert LangChain / LangGraph objects into JSON-serializable data."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.types import Interrupt, StateSnapshot
from pydantic import BaseModel


def message_text(msg: BaseMessage | str | list | None) -> str:
    """Plain text of a message, whatever shape its ``content`` has.

    Gemini (and others) return ``content`` as a list of blocks; both
    Projects using LangGraph keep writing this helper.
    """
    content = getattr(msg, "content", msg)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type", "text") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def message_to_json(msg: BaseMessage, compact: bool = False) -> dict[str, Any]:
    if compact:
        out: dict[str, Any] = {"id": msg.id, "type": msg.type, "content": msg.content}
        if getattr(msg, "tool_call_chunks", None):
            out["tool_call_chunks"] = to_json(msg.tool_call_chunks)
        if getattr(msg, "tool_calls", None):
            out["tool_calls"] = to_json(msg.tool_calls)
        if msg.type == "tool":  # which tool replied, and whether it failed
            out["name"] = getattr(msg, "name", None)
            out["tool_call_id"] = getattr(msg, "tool_call_id", None)
            if getattr(msg, "status", "success") != "success":
                out["status"] = msg.status
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
