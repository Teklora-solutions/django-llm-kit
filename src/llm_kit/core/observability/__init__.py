"""Recording every model call, tool call and run: tokens, cost, latency, fallbacks, errors.

from llm_kit.core.observability import context, classify_error

with context(user=user.pk, tenant="acme"):
    ...
"""

from llm_kit.core.observability.classify import RETRYABLE, classify_error, is_retryable
from llm_kit.core.observability.context import context, current
from llm_kit.core.observability.handler import ObservabilityHandler, get_handler, handler
from llm_kit.core.observability.recorder import Recorder, Sink, recorder
from llm_kit.core.observability.records import CallRecord, RunRecord, ToolRecord
from llm_kit.core.observability.sinks import LogSink, OTelSink


def flush(timeout: float = 10.0) -> bool:
    """Wait for pending records to be written (tests, scripts, before exit)."""
    return recorder.flush(timeout)


__all__ = [
    "RETRYABLE",
    "CallRecord",
    "LogSink",
    "OTelSink",
    "ObservabilityHandler",
    "Recorder",
    "RunRecord",
    "Sink",
    "ToolRecord",
    "classify_error",
    "context",
    "current",
    "flush",
    "get_handler",
    "handler",
    "is_retryable",
    "recorder",
]
