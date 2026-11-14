"""Configure the recorder's sinks from ``LLM_KIT["RECORD_SINKS"]`` (called from ``AppConfig.ready``)."""

from __future__ import annotations

import logging

from llm_kit import conf
from llm_kit.core.errors import ConfigError
from llm_kit.core.graphs import import_target
from llm_kit.core.observability import LogSink, OTelSink, recorder

logger = logging.getLogger("llm_kit.observability")


def build_sinks(names) -> list:
    sinks = []
    for name in names or []:
        if name == "database":
            from llm_kit.observability.sink import DatabaseSink

            sinks.append(DatabaseSink())
        elif name == "log":
            sinks.append(LogSink())
        elif name == "otel":
            try:
                sinks.append(OTelSink())
            except ImportError as e:
                raise ConfigError("RECORD_SINKS 'otel' needs: pip install opentelemetry-api") from e
        else:
            sinks.append(import_target(name)())
    return sinks


def configure() -> None:
    from llm_kit.core.observability.spend import spend
    from llm_kit.monitoring.breaker import observe
    from llm_kit.observability.budgets import period_start, seed
    from llm_kit.observability.prices import lookup

    enabled = conf.get("OBSERVABILITY")
    sinks = build_sinks(conf.get("RECORD_SINKS")) if enabled else []
    recorder.configure(sinks, lookup, observers=[observe, spend.observe] if enabled else [])
    # Budgets: each period's recorded spend, plus this process's calls since.
    spend.configure(seed, period_start)
