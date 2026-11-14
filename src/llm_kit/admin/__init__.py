"""Admin registrations, one module per feature (imported by Django's admin autodiscover)."""

from llm_kit.admin import (  # noqa: F401
    agents,
    catalogue,
    checkpoints,
    credentials,
    monitoring,
    observability,
    scoped,
    threads,
    tools,
    touchpoints,
)
