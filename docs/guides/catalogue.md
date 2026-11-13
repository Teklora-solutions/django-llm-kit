---
description: The model catalogue syncs prices, limits and capabilities for about 12,600 models from models.dev, OpenRouter and LiteLLM into your database.
---

# Model catalogue

Metadata for about 12,600 models (prices, context limits, capabilities, modalities, release and deprecation dates), synced from **models.dev**, **OpenRouter** and **LiteLLM** into your database. Touchpoints are checked against it, and call costs are priced from it.

## Syncing

```
python manage.py llm_kit_sync_catalogue            # conditional requests; safe to run often
python manage.py llm_kit_sync_catalogue --offline  # bundled snapshot, no network
python manage.py llm_kit_sync_catalogue --force    # ignore ETags and re-parse every source
python manage.py llm_kit_sync_catalogue --source openrouter   # only some sources (repeatable)
```

Schedule it every few hours:

- Celery Beat: the task `llm_kit.sync_catalogue`.
- Django-Q2 or anything that calls a function: `llm_kit.tasks.run_catalogue_sync`.
- cron: the management command.

The catalogue is never refreshed during a request. When it is older than `CATALOGUE_MAX_AGE_HOURS` (24), `catalogue.is_stale()` is true and `llm_kit_status` reports it.

## Querying

```python
from llm_kit import catalogue

m = catalogue.get("anthropic/claude-sonnet-5-5")   # also "anthropic:…" or a bare id
m.price_input, m.context_window, m.tool_call        # USD per 1M tokens, ints, bools
catalogue.search(requires={"tool_call": True, "input": ["image"]}, max_price_input=3, usable_only=True)
catalogue.has_credentials("gemini")                 # key env var names come from models.dev
```

Models are keyed by LiteLLM provider names, so `m.ref` is a valid LiteLLM model string.

## How sources are merged

- Each field comes from the best source for it. Prices prefer LiteLLM; capabilities prefer models.dev; a provider's own API (OpenRouter) wins for its own models.
- The admin shows which source supplied each field, and where sources disagree.
- A source that fails keeps its previous data; nothing is marked removed because of a network error.
- Only changed rows are written.

## Overrides and changes

- **Overrides** (*Catalogue overrides* in the admin): correct any field, hide a model, or add a private or self-hosted one. They apply immediately and survive syncs.
- **Change log** (*Catalogue changes* in the admin): added, removed, restored, price changes, capability changes, deprecations. The first sync is a silent baseline.
- A `catalogue_change` [alert rule](monitoring.md#alerts) tells you when a model one of your touchpoints uses is removed, deprecated or changes price.

## Sources

`CATALOGUE_SOURCES` limits the sources, by name or dotted path to your own source class. The default is all built-in sources.
