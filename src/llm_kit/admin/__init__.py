"""Admin registrations, one module per feature (imported by Django's admin autodiscover)."""

from llm_kit.admin import (  # noqa: F401
    catalogue,
    checkpoints,
    credentials,
    monitoring,
    observability,
    threads,
    tools,
    touchpoints,
)
