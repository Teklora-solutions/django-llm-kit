"""LangChain callback handler that turns model, tool and run events into records.

It only does in-memory bookkeeping (``run_inline=True``) and hands finished
records to the recorder; it never blocks on I/O.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from llm_kit.core import config
from llm_kit.core.catalogue.fields import normalise_provider
from llm_kit.core.observability.classify import classify_error
from llm_kit.core.observability.context import current
from llm_kit.core.observability.cost import estimate_tokens
from llm_kit.core.observability.recorder import recorder
from llm_kit.core.observability.records import CallRecord, RunRecord, ToolRecord

_INTERRUPTS = ("GraphInterrupt", "GraphBubbleUp", "ParentCommand")
_MAX_OPEN = 50_000  # safety valve for runs whose end event never arrives


def _now() -> datetime:
    return datetime.now(UTC)


def _hash(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _messages_payload(messages: list[list[Any]]) -> list[dict[str, Any]]:
    flat = messages[0] if messages else []
    return [{"type": getattr(m, "type", ""), "content": getattr(m, "content", m)} for m in flat]


def _text_of(payload: list[dict[str, Any]]) -> str:
    out = []
    for m in payload:
        c = m["content"]
        out.append(c if isinstance(c, str) else json.dumps(c, default=str))
    return "\n".join(out)


def _status_for(error: BaseException) -> tuple[str, str]:
    if type(error).__name__ in _INTERRUPTS:
        return "interrupted", ""
    cls = classify_error(error)
    return ("cancelled" if cls == "cancelled" else "error"), cls


class ObservabilityHandler(BaseCallbackHandler):
    run_inline = True
    raise_error = False

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._root: dict[UUID, UUID] = {}  # any run -> its root run
        self._runs: dict[UUID, tuple[RunRecord, float]] = {}
        self._calls: dict[UUID, dict[str, Any]] = {}
        self._tools: dict[UUID, tuple[ToolRecord, float]] = {}

    # ------------------------------------------------------------- helpers

    def _root_of(self, run_id: UUID, parent_run_id: UUID | None) -> UUID:
        root = self._root.get(parent_run_id, parent_run_id) if parent_run_id else run_id
        self._root[run_id] = root
        return root

    def _gc(self) -> None:
        if len(self._root) > _MAX_OPEN:
            self._root.clear()
            self._calls.clear()
            self._tools.clear()
            self._runs.clear()

    # --------------------------------------------------------------- runs

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        with self._lock:
            self._gc()
            root = self._root_of(run_id, parent_run_id)
            if root != run_id:
                return
            meta, ctx = metadata or {}, current()
            name = kwargs.get("name") or (serialized or {}).get("name") or "run"
            rec = RunRecord(
                run_id=str(run_id),
                name=str(name),
                started_at=_now(),
                graph=str(meta.get("llm_kit_graph", "")),
                thread_id=str(meta.get("llm_kit_thread") or meta.get("thread_id") or ""),
                user=str(meta.get("user_id") or ctx.user or ""),
                tenant=str(meta.get("tenant") or ctx.tenant or ""),
                tags=dict(ctx.tags),
            )
            self._runs[run_id] = (rec, time.perf_counter())

    def _end_run(self, run_id: UUID, error: BaseException | None) -> None:
        with self._lock:
            self._root.pop(run_id, None)
            entry = self._runs.pop(run_id, None)
        if entry is None:
            return
        rec, t0 = entry
        rec.ended_at = _now()
        rec.duration_ms = int((time.perf_counter() - t0) * 1000)
        if error is not None:
            rec.status, rec.error_class = _status_for(error)
        recorder.emit(rec)

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        self._end_run(run_id, None)

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._end_run(run_id, error)

    # ------------------------------------------------------------- models

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs
    ):
        with self._lock:
            if run_id in self._calls:  # handler attached twice (model + config)
                return
            root = self._root_of(run_id, parent_run_id)
        meta, ctx = metadata or {}, current()
        payload = _messages_payload(messages)
        invocation = kwargs.get("invocation_params") or {}
        model = meta.get("llm_kit_model") or (
            f"{meta['ls_provider']}/{meta['ls_model_name']}"
            if meta.get("ls_provider") and meta.get("ls_model_name")
            else ""
        )
        capture = meta.get("llm_kit_capture")
        if capture is None:
            capture = bool(config.get("CAPTURE_CONTENT"))
        rec = CallRecord(
            call_id=str(run_id),
            run_id=str(root) if root != run_id else None,
            started_at=_now(),
            touchpoint=str(meta.get("llm_kit_touchpoint", "")),
            graph=str(meta.get("llm_kit_graph", "")),
            node=str(meta.get("langgraph_node", "")),
            thread_id=str(meta.get("llm_kit_thread") or meta.get("thread_id") or ""),
            user=str(meta.get("user_id") or ctx.user or ""),
            tenant=str(meta.get("tenant") or ctx.tenant or ""),
            tags=dict(ctx.tags),
            requested_model=str(meta.get("llm_kit_requested") or ""),
            resolved_model=str(meta.get("llm_kit_resolved") or ""),
            model=str(model),
            provider=normalise_provider(str(model).split("/", 1)[0]) if "/" in str(model) else "",
            chosen_by=str(meta.get("llm_kit_chosen_by") or ""),
            fallback_index=int(meta.get("llm_kit_candidate") or 0),
            prompt_hash=_hash(payload),
            tools_hash=_hash(invocation.get("tools")) if invocation.get("tools") else "",
            prompt=payload if capture else None,
        )
        with self._lock:
            self._calls[run_id] = {
                "rec": rec,
                "t0": time.perf_counter(),
                "first": None,
                "prompt_text": _text_of(payload),
            }

    def on_llm_new_token(self, token, *, chunk=None, run_id, parent_run_id=None, **kwargs):
        state = self._calls.get(run_id)
        if state is not None and state["first"] is None:
            state["first"] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        with self._lock:
            state = self._calls.pop(run_id, None)
            self._root.pop(run_id, None)
        if state is None:
            return
        rec: CallRecord = state["rec"]
        end = time.perf_counter()
        rec.latency_ms = int((end - state["t0"]) * 1000)
        if state["first"] is not None:
            rec.streamed = True
            rec.time_to_first_token_ms = int((state["first"] - state["t0"]) * 1000)

        gen = response.generations[0][0] if response.generations and response.generations[0] else None
        msg = getattr(gen, "message", None)
        usage = getattr(msg, "usage_metadata", None) or {}
        meta = getattr(msg, "response_metadata", None) or {}
        if usage:
            rec.input_tokens = int(usage.get("input_tokens") or 0)
            rec.output_tokens = int(usage.get("output_tokens") or 0)
            details_in = usage.get("input_token_details") or {}
            details_out = usage.get("output_token_details") or {}
            rec.cache_read_tokens = int(details_in.get("cache_read") or 0)
            rec.cache_write_tokens = int(details_in.get("cache_creation") or 0)
            rec.reasoning_tokens = int(details_out.get("reasoning") or 0)
        else:
            legacy = (response.llm_output or {}).get("token_usage") or (response.llm_output or {}).get("usage") or {}
            if legacy:
                rec.input_tokens = int(legacy.get("prompt_tokens") or legacy.get("input_tokens") or 0)
                rec.output_tokens = int(legacy.get("completion_tokens") or legacy.get("output_tokens") or 0)
            else:  # provider sent nothing: estimate rather than record zero (the call was still billed)
                rec.input_tokens = estimate_tokens(state["prompt_text"])
                rec.output_tokens = estimate_tokens(getattr(gen, "text", "") or "")
                rec.usage_estimated = True
        rec.served_model = str(meta.get("model_name") or meta.get("model") or "")
        rec.provider_request_id = str(meta.get("id") or meta.get("request_id") or "")
        if rec.prompt is not None and gen is not None:
            rec.completion = {
                "content": getattr(msg, "content", getattr(gen, "text", "")),
                "tool_calls": getattr(msg, "tool_calls", None),
            }
        self._count(rec)
        recorder.emit(rec)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        with self._lock:
            state = self._calls.pop(run_id, None)
            self._root.pop(run_id, None)
        if state is None:
            return
        rec: CallRecord = state["rec"]
        rec.latency_ms = int((time.perf_counter() - state["t0"]) * 1000)
        rec.status, rec.error_class = _status_for(error)
        rec.error_message = f"{type(error).__name__}: {error}"[:500]
        self._count(rec)
        recorder.emit(rec)

    def _count(self, rec: CallRecord) -> None:
        if rec.run_id is None:
            return
        with self._lock:
            entry = self._runs.get(UUID(rec.run_id))
            if entry is not None:
                run = entry[0]
                run.calls += 1
                run.input_tokens += rec.input_tokens
                run.output_tokens += rec.output_tokens

    # -------------------------------------------------------------- tools

    def on_tool_start(
        self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, inputs=None, **kwargs
    ):
        with self._lock:
            root = self._root_of(run_id, parent_run_id)
        rec = ToolRecord(
            tool_call_id=str(run_id),
            run_id=str(root) if root != run_id else None,
            name=str(kwargs.get("name") or (serialized or {}).get("name") or "tool"),
            started_at=_now(),
            input_hash=_hash(inputs if inputs is not None else input_str),
        )
        with self._lock:
            self._tools[run_id] = (rec, time.perf_counter())

    def _end_tool(self, run_id: UUID, output: Any, error: BaseException | None) -> None:
        with self._lock:
            entry = self._tools.pop(run_id, None)
            self._root.pop(run_id, None)
        if entry is None:
            return
        rec, t0 = entry
        rec.duration_ms = int((time.perf_counter() - t0) * 1000)
        if error is not None:
            rec.status, rec.error_class = _status_for(error)
        else:
            rec.output_hash = _hash(getattr(output, "content", output))
        if rec.run_id is not None:
            with self._lock:
                run = self._runs.get(UUID(rec.run_id))
                if run is not None:
                    run[0].tool_calls += 1
        recorder.emit(rec)

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        self._end_tool(run_id, output, None)

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._end_tool(run_id, None, error)


handler = ObservabilityHandler()


def get_handler() -> ObservabilityHandler | None:
    """The shared handler, or None when recording is switched off (``OBSERVABILITY: False``)."""
    return handler if config.get("OBSERVABILITY") else None
