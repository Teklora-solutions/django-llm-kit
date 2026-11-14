"""System checks for background runs."""

from __future__ import annotations

from django.core import checks


@checks.register("llm_kit")
def worker_events_reach_watchers(app_configs=None, **kwargs):
    """A run in a worker process can only reach watchers in the web processes through a
    channel layer; with the local bus they would see the snapshot and nothing after it."""
    from llm_kit import conf
    from llm_kit.background.bus import LocalBus, get_bus
    from llm_kit.core.errors import ConfigError

    if conf.get("RUN_EXECUTOR") == "thread":
        return []
    try:
        bus = get_bus()
    except ConfigError as e:
        return [checks.Error(str(e), id="llm_kit.E001")]
    if isinstance(bus, LocalBus):
        return [
            checks.Warning(
                f"RUN_EXECUTOR={conf.get('RUN_EXECUTOR')!r} runs background runs in other processes, but their "
                "events only reach watchers in the same process.",
                hint='Set CHANNEL_LAYERS to a shared layer (e.g. channels_redis) and RUN_EVENTS to "auto" or '
                '"channels".',
                id="llm_kit.W002",
            )
        ]
    return []
