from django.apps import AppConfig


class LLMKitConfig(AppConfig):
    name = "llm_kit"
    label = "llm_kit"
    verbose_name = "LLM Kit"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        # conf wires settings.LLM_KIT into llm_kit.core; the signal modules connect receivers on import.
        # Before anything imports litellm: use its bundled price map, not a network fetch at import.
        from llm_kit.core.touchpoints.backends import _quiet_litellm_import

        _quiet_litellm_import()

        # Touchpoints resolve against the catalogue, overrides and stored keys.
        from django.utils.module_loading import autodiscover_modules

        from llm_kit import conf  # noqa: F401
        from llm_kit.catalogue import signals as catalogue_signals  # noqa: F401
        from llm_kit.core.touchpoints.env import set_default_env
        from llm_kit.persistence import signals as persistence_signals  # noqa: F401
        from llm_kit.touchpoints import signals as touchpoint_signals  # noqa: F401
        from llm_kit.touchpoints.env import DjangoEnv

        set_default_env(DjangoEnv())
        autodiscover_modules("touchpoints")
        # Recording: sinks from LLM_KIT["RECORD_SINKS"], prices from the catalogue.
        from llm_kit.observability.setup import configure as configure_observability

        configure_observability()

        # Model tools: AgentAccess declarations, plus expose()/register() calls in apps' agent_tools.py.
        # A bad declaration stops start-up here rather than failing at the first tool call.
        from llm_kit.tools import registry as tool_registry

        autodiscover_modules("agent_tools")
        tool_registry.load()

        # Touchpoints declared next to graphs register when their module is imported.
        from llm_kit.runtime.graphs import registry

        registry.import_modules()
