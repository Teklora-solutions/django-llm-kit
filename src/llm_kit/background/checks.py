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


@checks.register("llm_kit")
def worker_events_are_kept_where_watchers_read(app_configs=None, **kwargs):
    """A log in process memory can't be read by watchers in other processes."""
    from llm_kit import conf
    from llm_kit.background.log import MemoryStore, get_log
    from llm_kit.core.errors import ConfigError

    try:
        log = get_log()
    except ConfigError as e:
        return [checks.Error(str(e), id="llm_kit.E002")]
    if conf.get("RUN_EXECUTOR") == "thread" or log is None or not isinstance(log.store, MemoryStore):
        return []
    return [
        checks.Warning(
            f"RUN_EXECUTOR={conf.get('RUN_EXECUTOR')!r} runs background runs in other processes, but their "
            "event log is kept in process memory: watchers elsewhere can't replay what they missed.",
            hint='Use a shared cache (Redis, Memcached) for RUN_EVENT_LOG_CACHE, or set RUN_EVENT_LOG = "off".',
            id="llm_kit.W003",
        )
    ]


@checks.register("llm_kit")
def media_webhooks_can_arrive(app_configs=None, **kwargs):
    """Webhooks for media jobs need a secret (or your own check) and, where the job tells the
    provider where to call, the URL to give it."""
    from llm_kit import conf
    from llm_kit.media.webhooks import config_for

    messages = []
    for provider in conf.get("MEDIA_WEBHOOKS") or {}:
        config = config_for(provider) or {}
        how = config.get("verify") or "standard"
        if how in ("standard", "hmac") and not config.get("secret"):
            messages.append(
                checks.Error(
                    f"MEDIA_WEBHOOKS[{provider!r}] checks signatures with a secret but has none.",
                    hint="Add the signing secret the provider gave you, or set verify to your own function.",
                    id="llm_kit.E003",
                )
            )
        if config.get("param") and not conf.get("MEDIA_WEBHOOK_URL"):
            messages.append(
                checks.Warning(
                    f"MEDIA_WEBHOOKS[{provider!r}] gives jobs a webhook URL, but MEDIA_WEBHOOK_URL is not set: "
                    "its jobs fall back to checking every MEDIA_WEBHOOK_POLL_EVERY seconds.",
                    hint='Set MEDIA_WEBHOOK_URL = "https://<your site>/llm/media/webhooks/{provider}/".',
                    id="llm_kit.W004",
                )
            )
    return messages
