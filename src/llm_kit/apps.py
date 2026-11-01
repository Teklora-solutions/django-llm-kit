from django.apps import AppConfig


class LLMKitConfig(AppConfig):
    name = "llm_kit"
    label = "llm_kit"
    verbose_name = "LLM Kit"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        # conf wires settings.LLM_KIT into llm_kit.core; the signal modules connect receivers on import.
        from llm_kit import conf  # noqa: F401
        from llm_kit.catalogue import signals as catalogue_signals  # noqa: F401
        from llm_kit.persistence import signals as persistence_signals  # noqa: F401
