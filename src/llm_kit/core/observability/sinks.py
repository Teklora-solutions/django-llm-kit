"""Built-in sinks: structured logging and OpenTelemetry. The Django app adds a database sink."""

from __future__ import annotations

import dataclasses
import json
import logging

from llm_kit.core.observability.records import CallRecord, RunRecord, ToolRecord

call_logger = logging.getLogger("llm_kit.calls")


class LogSink:
    """One JSON line per call/run/tool on the ``llm_kit.calls`` logger (content never included)."""

    def write(self, calls: list[CallRecord], runs: list[RunRecord], tools: list[ToolRecord]) -> None:
        if not call_logger.isEnabledFor(logging.INFO):
            return
        for kind, records in (("call", calls), ("run", runs), ("tool", tools)):
            for r in records:
                data = dataclasses.asdict(r)
                data.pop("prompt", None)
                data.pop("completion", None)
                call_logger.info(json.dumps({"event": f"llm_kit.{kind}", **data}, default=str))


class OTelSink:
    """OpenTelemetry spans using the GenAI semantic conventions (``pip install opentelemetry-api``).

    Spans are created after the fact with the recorded start and end times.
    """

    def __init__(self) -> None:
        from opentelemetry import trace

        self._trace = trace
        self._tracer = trace.get_tracer("llm_kit")

    def write(self, calls: list[CallRecord], runs: list[RunRecord], tools: list[ToolRecord]) -> None:
        for c in calls:
            start = int(c.started_at.timestamp() * 1e9)
            provider, _, model = (c.model or c.served_model).partition("/")
            span = self._tracer.start_span(
                f"chat {model or c.model}", start_time=start, kind=self._trace.SpanKind.CLIENT
            )
            attrs = {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": provider,
                "gen_ai.request.model": model,
                "gen_ai.response.model": c.served_model,
                "gen_ai.usage.input_tokens": c.input_tokens,
                "gen_ai.usage.output_tokens": c.output_tokens,
                "llm_kit.touchpoint": c.touchpoint,
                "llm_kit.chosen_by": c.chosen_by,
                "llm_kit.fallback_index": c.fallback_index,
                "llm_kit.thread_id": c.thread_id,
                "llm_kit.run_id": c.run_id or "",
            }
            if c.cost is not None:
                attrs["llm_kit.cost_usd"] = float(c.cost)
            if c.error_class:
                attrs["error.type"] = c.error_class
            span.set_attributes({k: v for k, v in attrs.items() if v not in (None, "")})
            if c.status == "error":
                span.set_status(self._trace.Status(self._trace.StatusCode.ERROR, c.error_message))
            span.end(end_time=start + c.latency_ms * 1_000_000)
