---
description: The llm_kit management commands, their options and exit codes.
---

# Management commands

| Command | What it does |
|---|---|
| `llm_kit_sync_catalogue [--offline] [--force] [--source NAME]…` | Refresh the [model catalogue](../guides/catalogue.md). `--offline` loads the bundled snapshot; `--force` ignores ETags; `--source` limits the sources (repeatable). |
| `llm_kit_touchpoints [--check]` | List [touchpoints](../guides/touchpoints.md), what each resolves to, and problems (missing from the catalogue, deprecated, unpriced, no key). `--check` exits with an error if there are any. |
| `llm_kit_agents [--user NAME]` | List [agents](../guides/agents.md): model, tools, subagents, limits and admin overrides. With `--user`, the tools that user gets. |
| `llm_kit_tools [--user NAME]` | List [model tools](../guides/tools.md) and hand-written tools, their scope and permissions. With `--user`, only that user's tools. |
| `llm_kit_status [--no-fail]` | Health summary: catalogue age, open circuits, touchpoint problems, last 24 h. Exits 1 when unhealthy unless `--no-fail`. |
| `llm_kit_check_alerts` | Evaluate enabled [alert rules](../guides/monitoring.md#alerts) and deliver the ones that fire. |
| `llm_kit_prune [--dry-run] [--calls DAYS] [--content DAYS] [--changes DAYS] [--threads DAYS]` | Delete old calls, runs and tool calls, clear captured content, delete old catalogue changes and alerts (180 days). Threads are deleted only with `--threads`. |

## Celery tasks

If Celery is installed, these tasks are registered:

| Task | Same as |
|---|---|
| `llm_kit.sync_catalogue` | `llm_kit_sync_catalogue` |
| `llm_kit.check_alerts` | `llm_kit_check_alerts` |

Without Celery, call `llm_kit.tasks.run_catalogue_sync()` or `llm_kit.monitoring.tasks.run_alert_checks()` from your own scheduler.
