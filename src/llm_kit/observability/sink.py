"""Writes call, run and tool records to the llm_kit tables (runs on the recorder thread)."""

from __future__ import annotations

import dataclasses
import logging

from django.db import DEFAULT_DB_ALIAS, close_old_connections

from llm_kit.core.observability.records import CallRecord, RunRecord, ToolRecord
from llm_kit.models import LLMCall, LLMRun, ToolCall

logger = logging.getLogger("llm_kit.observability")


def _fields(model) -> set[str]:
    return {f.name for f in model._meta.concrete_fields} - {"id"}


class DatabaseSink:
    def __init__(self, using: str = DEFAULT_DB_ALIAS) -> None:
        self.using = using
        self._call_fields = _fields(LLMCall)
        self._run_fields = _fields(LLMRun)
        self._tool_fields = _fields(ToolCall)

    def _row(self, model, record, allowed: set[str]):
        data = {k: v for k, v in dataclasses.asdict(record).items() if k in allowed}
        for key in ("call_id", "run_id", "tool_call_id"):
            if key in data and data[key] == "":
                data[key] = None
        return model(**data)

    def write(self, calls: list[CallRecord], runs: list[RunRecord], tools: list[ToolRecord]) -> None:
        # This thread is long-lived: drop expired/broken connections like a request would.
        close_old_connections()
        for model, records, fields in (
            (LLMCall, calls, self._call_fields),
            (LLMRun, runs, self._run_fields),
            (ToolCall, tools, self._tool_fields),
        ):
            if records:
                model.objects.using(self.using).bulk_create(
                    [self._row(model, r, fields) for r in records], ignore_conflicts=True
                )
