"""Celery autodiscovery looks for ``<app>.tasks``; the tasks live with their features."""

from llm_kit.catalogue.tasks import *  # noqa: F401,F403
from llm_kit.monitoring.tasks import *  # noqa: F401,F403
