from django.apps import AppConfig


class LLMKitConfig(AppConfig):
    name = "llm_kit"
    label = "llm_kit"
    verbose_name = "LLM Kit"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        from llm_kit import conf, signals  # noqa: F401  (conf wires settings into llm_kit.core)
