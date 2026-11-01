"""Admin registrations, one module per feature (imported by Django's admin autodiscover)."""

from llm_kit.admin import catalogue, checkpoints, legacy, threads  # noqa: F401
