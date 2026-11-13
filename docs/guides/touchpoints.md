---
description: Touchpoints declare, in code, which model each LLM call uses, its fallbacks, required capabilities, and whether clients may choose the model.
---

# Touchpoints

A **touchpoint** is one place your code calls a model. You declare it once; which model runs there, what it falls back to, and who may change it all follow from the declaration.

```python
from llm_kit import touchpoint, ClientChoice, ServerChoice

summarise = touchpoint(
    "crm.summarise_lead",
    default="anthropic/claude-haiku-5-5",     # LiteLLM-style provider/model_id
    fallbacks=["gemini/gemini-3.8-flash"],    # tried in order if the call fails
    params={"temperature": 0},
    requires={"structured_output": True},     # checked against the catalogue
    choice=ServerChoice(),                    # the default: clients can't pick
)
```

Declare touchpoints in `<app>/touchpoints.py` (imported at startup), or next to the graph or agent that uses them.

## Using a touchpoint

```python
lead = summarise.complete(text, schema=Lead)  # one-shot call, structured if you pass a schema; no graph needed
llm = summarise.model()                       # a LangChain chat model; bind_tools etc. keep the fallbacks
llm = summarise.model(config=config)          # inside a graph node: honours the client's pick
summarise.resolve()                           # which model(s) and why, without calling anything
```

`model()` returns a LangChain chat model that wraps the default and its fallbacks. `bind_tools` and `with_structured_output` are applied to every candidate, so fallbacks keep working after you bind tools. `complete()` also parses the raw reply when a model answers in text instead of structured output.

## Who chooses the model

| Policy | When a client sends `"model"` | Options |
|---|---|---|
| `ServerChoice()` (default) | ignored, with the reason recorded | `on_requested="reject"` returns 400 instead |
| `ClientChoice(allow=…, requires=…, max_price_input=…, max_price_output=…)` | accepted if it matches the `allow` patterns, has the required capabilities and is under the price caps (USD per million tokens, from the catalogue) | `on_disallowed="ignore"` uses the default instead of returning 400; `use_fallbacks=False` drops fallbacks behind a client's pick |

A client's pick also needs a usable API key; a model without one is refused.

## Resolution order

1. A client pick that the policy allows.
2. An **admin override** (*Touchpoint overrides* in the admin): default, fallbacks, params, a kill switch, and an `allow` list that can **narrow but never widen** what the code allows. Overrides apply without a deploy.
3. The default from code.

Then:

- Fallbacks without an API key, or failing `requires`, are skipped.
- If the default has no key, the first usable fallback is promoted.
- Providers whose [circuit](monitoring.md#circuit-breaker) is open move behind healthy fallbacks.
- Every result records who chose the model and why. You'll see it on each recorded call.

Run `python manage.py llm_kit_touchpoints [--check]` to list every touchpoint, what it resolves to, and problems: models missing from the catalogue, deprecated, unpriced, or without a key. `--check` exits non-zero when there are problems, for CI.

## Backends

Set `LLM_KIT["MODEL_BACKEND"]`, or `touchpoint(..., backend=…)` for one touchpoint:

| Backend | How models are built |
|---|---|
| `"auto"` (default) | `litellm` when installed, otherwise `langchain` |
| `"litellm"` | `ChatLiteLLM` (the `[litellm]` extra). One dependency covers every provider. With several keys for a provider, calls are spread across them through a LiteLLM Router (`LITELLM_ROUTER` settings) |
| `"langchain"` | `init_chat_model` with the provider's `langchain-*` package |
| `"myapp.ai:build"` | your own `fn(ref, params, api_keys) -> BaseChatModel` |

## API keys

Keys are looked up in this order:

1. **`CREDENTIALS`**: a dotted path to `fn(provider)` that returns a secret, a list of secrets, or `None`. Use it for a settings object, a vault, a service-account path, or OAuth tokens.
2. **Encrypted keys in the database** (*Provider keys* in the admin). Needs the `[crypto]` extra and `LLM_KIT["ENCRYPTION_KEYS"]`, a list of Fernet keys or passphrases, newest first, so you can rotate. Keys are never shown again after saving.
3. **Environment variables**: `ANTHROPIC_API_KEY`, or any name models.dev lists for the provider, or a comma-separated pool in `ANTHROPIC_API_KEYS`.

## Recording content

Prompts and responses are not stored by default, only hashes. `touchpoint(..., capture=True)` stores them for that touchpoint. See [Observability](observability.md).
