"""Model catalogue for Django: stored in the DB, synced from models.dev, OpenRouter and LiteLLM.

    from llm_kit import catalogue

    catalogue.sync()                                  # or: manage.py llm_kit_sync_catalogue
    m = catalogue.get("anthropic/claude-sonnet-5-5")  # CatalogueModel or None
    catalogue.search(requires={"tool_call": True, "input": ["image"]}, max_price_input=3)
    catalogue.has_credentials("gemini")

Never syncs during a request: run the command or the scheduled task
(``llm_kit.tasks.run_catalogue_sync``).
"""

from llm_kit.catalogue.queries import active, get, has_credentials, prices, recent_changes, search
from llm_kit.catalogue.refresh import apply_override, configured_sources, is_stale, last_synced, sync
from llm_kit.catalogue.store import DjangoCatalogueStore

__all__ = [
    "DjangoCatalogueStore",
    "active",
    "apply_override",
    "configured_sources",
    "get",
    "has_credentials",
    "is_stale",
    "last_synced",
    "prices",
    "recent_changes",
    "search",
    "sync",
]
