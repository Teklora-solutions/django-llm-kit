"""Admin registrations, one module per feature (imported by Django's admin autodiscover)."""

from llm_kit.admin import catalogue, checkpoints, credentials, observability, threads, touchpoints  # noqa: F401
