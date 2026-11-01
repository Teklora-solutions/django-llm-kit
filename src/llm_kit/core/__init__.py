"""Framework-independent core of llm-kit.

Nothing under ``llm_kit.core`` may import Django (enforced by
``tests/test_core_boundary.py``). Framework adapters, such as the Django app in
``llm_kit``, supply configuration through ``llm_kit.core.config.configure`` and
their own storage.
"""
