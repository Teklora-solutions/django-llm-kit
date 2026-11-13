# Contributing

Thanks for helping. Bug reports, docs fixes and pull requests are all welcome.

## Set up

```
git clone https://github.com/Teklora-solutions/django-llm-kit
cd django-llm-kit
uv venv && uv pip install -e ".[dev]"
```

## Run the checks

```
pytest                                    # SQLite
ruff check . && ruff format --check .     # lint and formatting
```

To run the suite on Postgres too:

```
docker run -d --name llmkit-pg -e POSTGRES_PASSWORD=pw -p 5432:5432 postgres:16
LLM_KIT_TEST_PG=postgres://postgres:pw@localhost:5432/postgres pytest
LLM_KIT_TEST_PG=… LLM_KIT_TEST_CHECKPOINTER=postgres pytest   # with LangGraph's PostgresSaver
```

The tests never call a real model provider. They use fake chat models (`tests/graphs.py`) and, for the catalogue, recorded payloads (`tests/catalogue_payloads.py`). Please keep it that way.

## Where code goes

- `src/llm_kit/core/` must not import Django. `tests/test_core_boundary.py` enforces this.
- Everything else is a Django adapter, grouped by feature (`touchpoints/`, `agents/`, `tools/`, `observability/`, `monitoring/`, `transport/`, …). Add new code to the feature package it belongs to, not as a new module at the package root.
- Model changes need a migration: `python -m django makemigrations llm_kit --settings tests.settings` (run from the repo root with `PYTHONPATH=src:.`).

## Docs

The documentation lives in `docs/` and is built with MkDocs Material:

```
uv pip install -e ".[docs]"
mkdocs serve
```

Every page is also published as plain Markdown, and `llms.txt` / `llms-full.txt` are generated at build time, so keep pages readable as Markdown on their own: no HTML-only content.

## Pull requests

- One change per pull request, with a test that fails without it.
- Update the docs page and `CHANGELOG.md` (under *Unreleased*) for anything a user would notice.
- CI runs lint, the test matrix (Python 3.11–3.13, SQLite and Postgres) and a strict docs build.
