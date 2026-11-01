"""System checks for storing graph state in the Django database."""

from __future__ import annotations

from django.core import checks
from django.db import connections


@checks.register("llm_kit")
def sqlite_test_database_is_shared(app_configs=None, **kwargs):
    """LangGraph saves checkpoints from a background thread. Django's default SQLite test
    database is in memory with a shared cache, where another thread's write locks whole
    tables ("database table is locked"), so tests that run graphs fail.
    """
    from llm_kit import conf

    if conf.get("CHECKPOINTER") not in (None, "django"):
        return []
    found = []
    for alias in connections:
        s = connections.settings[alias]
        if s.get("ENGINE", "").rsplit(".", 1)[-1] != "sqlite3":
            continue
        test_name = (s.get("TEST") or {}).get("NAME")
        if not test_name or str(test_name) == ":memory:" or "mode=memory" in str(test_name):
            found.append(
                checks.Warning(
                    f"The SQLite test database for {alias!r} is in memory; tests that run graphs or agents "
                    "will fail with 'database table is locked'.",
                    hint=f'Give it a file: DATABASES["{alias}"]["TEST"] = {{"NAME": BASE_DIR / "test.sqlite3"}}, '
                    'and add "OPTIONS": {"timeout": 20}.',
                    id="llm_kit.W001",
                )
            )
    return found
