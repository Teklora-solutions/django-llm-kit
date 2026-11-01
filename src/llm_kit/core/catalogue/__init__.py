"""Model catalogue built from online sources (models.dev, OpenRouter, LiteLLM).

Framework-independent: storage is supplied through ``CatalogueStore``.
"""

from llm_kit.core.catalogue.fields import FIELDS, PRICE_FIELDS, Key, model_ref, normalise_provider
from llm_kit.core.catalogue.merge import Merged, merge
from llm_kit.core.catalogue.refresh import Response, SyncReport, reapply_overrides, sync, urllib_fetcher
from llm_kit.core.catalogue.sources import DEFAULT_SOURCES, LITELLM, MODELS_DEV, OPENROUTER, Source
from llm_kit.core.catalogue.store import CatalogueState, CatalogueStore, Change, Entry, MemoryStore, SourceState

__all__ = [
    "DEFAULT_SOURCES",
    "FIELDS",
    "LITELLM",
    "MODELS_DEV",
    "OPENROUTER",
    "PRICE_FIELDS",
    "CatalogueState",
    "CatalogueStore",
    "Change",
    "Entry",
    "Key",
    "MemoryStore",
    "Merged",
    "Response",
    "Source",
    "SourceState",
    "SyncReport",
    "merge",
    "model_ref",
    "normalise_provider",
    "reapply_overrides",
    "sync",
    "urllib_fetcher",
]
