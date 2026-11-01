"""What gets recorded. Sinks receive these; the Django sink stores them as rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any


@dataclass
class CallRecord:
    """One model call (one attempt: a failed primary and its fallback are two records)."""

    call_id: str
    run_id: str | None  # root run this call belongs to, if any
    started_at: datetime
    touchpoint: str = ""
    graph: str = ""
    agent: str = ""  # the agent whose step this was
    node: str = ""
    thread_id: str = ""
    user: str = ""
    tenant: str = ""
    tags: dict[str, Any] = field(default_factory=dict)
    requested_model: str = ""
    resolved_model: str = ""  # what resolution picked for the touchpoint
    model: str = ""  # the candidate this attempt used
    provider: str = ""  # provider part of ``model`` (LiteLLM name)
    served_model: str = ""  # what the provider reports
    chosen_by: str = ""
    fallback_index: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    usage_estimated: bool = False
    cost: Decimal | None = None
    unpriced: bool = False
    latency_ms: int = 0
    time_to_first_token_ms: int | None = None
    streamed: bool = False
    status: str = "ok"  # ok | error | cancelled
    error_class: str = ""
    error_message: str = ""
    prompt_hash: str = ""
    tools_hash: str = ""
    provider_request_id: str = ""
    prompt: Any = None  # only when content capture is on
    completion: Any = None
    agent_runs: list[str] = field(default_factory=list)  # subagent runs it also counts towards (not stored)


@dataclass
class RunRecord:
    """One top-level invocation (a graph run or a ``complete()`` call), or a subagent's run inside one.

    A request's run counts every call made under it, subagents included; a subagent's run
    counts its own calls and those of the subagents it started."""

    run_id: str
    name: str
    started_at: datetime
    ended_at: datetime | None = None
    graph: str = ""
    agent: str = ""
    parent_run_id: str | None = None  # a subagent's run: the run that handed it the task
    depth: int = 0  # 0 for the run a request started, 1 for its subagents, ...
    thread_id: str = ""
    user: str = ""
    tenant: str = ""
    tags: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # ok | error | interrupted | cancelled
    error_class: str = ""
    calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: Decimal = Decimal(0)
    unpriced_calls: int = 0
    duration_ms: int = 0


@dataclass
class ToolRecord:
    tool_call_id: str
    run_id: str | None
    name: str
    started_at: datetime
    duration_ms: int = 0
    status: str = "ok"
    error_class: str = ""
    input_hash: str = ""
    output_hash: str = ""
