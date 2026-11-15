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

Called inside a running agent (from a tool or a code agent) without a `config`, `complete()` joins that agent's run: see [running agents](agents.md#running-agents).

`model()` returns a LangChain chat model that wraps the default and its fallbacks. `bind_tools` and `with_structured_output` are applied to every candidate, so fallbacks keep working after you bind tools. `complete()` also parses the raw reply when a model answers in text instead of structured output.

## Rate limits and fallbacks

Each candidate is tried in turn. A refusal that clears on its own (a rate limit, or a provider that's overloaded) is **waited out on the same model**, if the wait is short enough; anything else moves to the next fallback at once.

- The wait is the one the provider asked for: a `Retry-After` header, a `retry_after` attribute, or the error text (`"retryDelay": "4s"`, `try again in 250ms`). A rate limit that doesn't say waits 20 s; an overloaded provider, 5 s.
- A **daily quota** is never waited out. Gemini reports per-minute and per-day limits with the same 429; llm-kit reads the quota id to tell them apart.
- How long to wait depends on who is waiting, so it's a `RetryPolicy(attempts, max_wait)`: up to `attempts` waits per model, none longer than `max_wait` seconds.

```python
from llm_kit import RetryPolicy, touchpoint

digest = touchpoint("crm.digest", default=..., retry=RetryPolicy.BATCH)  # 4 waits of up to 65 s: a worker can wait
chat.model(config=config, retry="never")                                 # this call never waits
agents.run("crm.digest_writer", task, retry="batch")                     # every model call in this run
```

| Policy | Waits | Use for |
|---|---|---|
| `RetryPolicy.INTERACTIVE` (the default, `LLM_KIT["RETRY"]`) | 1 wait of up to 6 s | someone is watching the reply |
| `RetryPolicy.BATCH` | 4 waits of up to 65 s | background jobs |
| `RetryPolicy.NEVER` | none | fall back at once |

The policy for a call is the first of: `model(retry=)` / `complete(retry=)`, `agents.run(retry=)`, `touchpoint(retry=)`, `LLM_KIT["RETRY"]`. A stream waits and falls back only before its first chunk; after that the caller already has part of the answer.

`model(fallbacks=False)` (or `complete(fallbacks=False)`) uses only the resolved model for that call, for results you compare across runs, such as an evaluation's judge. It still waits per its policy.

When **every** model fails, the call raises `llm_kit.AllModelsFailed`. Its message lists each model's error, `errors` holds `(model, exception)` pairs in the order tried, and `__cause__` is the last one. A touchpoint with a single model raises that model's own error. Each attempt is a [recorded call](observability.md) with its error class.

Provider SDKs retry on their own too (Anthropic's and OpenAI's clients retry twice by default). llm-kit's wait comes after theirs, so lower the SDK's `max_retries` in `params` if you want llm-kit to decide.

### Your account's rate limits

Waiting out a 429 only helps after the provider refused you. When you know your account's limits, llm-kit can keep within them. It counts requests and tokens per minute and requests per day for each model, shared by every process, and doesn't send a call that would go over.

```python
LLM_KIT = {
    "RATE_LIMITS": {
        "gemini/gemini-3.8-flash": {"rpm": 10, "tpm": 250_000, "rpd": 500},
        "anthropic/*": {"rpm": 50, "tpm": 40_000},     # each matching model gets these
    },
}
```

Or set them in the admin (*LLM Kit → Rate limits*). A row wins over the setting for the same pattern, and a model's own limits win over a pattern's. These describe your account's tier with each provider, so they are set by people and never synced from the catalogue. `0` means no limit of that kind.

- **Before each call** a slot is taken. When the model is at a limit, the call waits for a slot up to its retry policy's `max_wait` (6 s by default; never with `RetryPolicy.NEVER`). It doesn't wait if no slot frees up in time. Then it moves to the next fallback.
- **When no model has a slot**, the call fails with `llm_kit.core.touchpoints.ratelimit.RateLimited` (in `AllModelsFailed` when there were fallbacks). It is classified `rate_limit`, or `quota_exhausted` for the daily limit, and gives the `reason` (`rpm`, `tpm`, `rpd`) and `retry_in`.
- **Tokens** are counted after each call, from the usage the provider reports, so a token limit applies from the next call. One call can go over it.
- A refused request takes no slot. Once any limit is set, all three counts are kept for every model, so `GET llm/api/monitoring/rate-limits/` (or `ratelimits.usage()`) can show use against limits.
- Calls made with keys a tenant or user [brought](settings.md#bring-your-own-key) aren't counted: those are their account's limits, not yours.
- Embedding calls take slots too.

The counts live in Django's cache (`RATE_LIMIT_BACKEND`, default `"auto"`):
- **Redis as the default cache** (Django's `RedisCache` or django-redis): one atomic script with a sliding one-minute window, so it never goes over.
- **Memcached or another shared cache:** fixed clock-minute windows. Up to twice the per-minute limit can go out across a minute boundary.
- **The local-memory cache:** each process counts on its own.

Set `"redis"` to require Redis, or `"off"` to stop counting.

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

Your factory may build a model once and return the same object every time, for example to keep its HTTP client. llm-kit never changes the model it gets. The touchpoint's metadata, tags and recording handler are added as call config: `model()` returns the factory's model wrapped in a `TouchpointModel` (a `RunnableBinding`; the model is `.bound`, attributes are forwarded). `bind_tools` and `with_structured_output` keep that config, and a call is recorded once even if you also pass llm-kit's handler yourself.

## Input for each model

Fallbacks don't all read the same input. One model hears audio, another needs a transcript. A file uploaded to one provider can only be opened by that provider. `adapt_input` gives each model input it can read:

```python
def media_for(input, ref, info):
    """Called for each model just before it is tried, fallbacks included."""
    hears_audio = bool(info and "audio" in (info.values.get("input_modalities") or []))
    return with_media(input, audio=hears_audio, files_api=ref.startswith("gemini/"))

assistant = touchpoint("support.assistant", default=..., fallbacks=[...], adapt_input="myapp.media:media_for")
assistant.model(config=config, adapt_input=other_fn)   # for one call; adapt_input=False: none
```

- It gets the input as passed to the model (a string, messages, or a prompt value), the model's `ref`, and its catalogue entry (`ModelInfo`, or `None` when the catalogue doesn't know it). It returns that model's input.
- It runs for `invoke`, `ainvoke`, `stream` and `astream`, after `bind_tools` and `with_structured_output` too, once per model (not per retry wait), and before the model's [rate limit](#your-accounts-rate-limits) slot is taken.
- It may be an `async def`; then it is awaited in async calls (and run on an event loop in sync ones).
- If it raises, that model is skipped like one that failed, and the next fallback is tried. The error is kept in `AllModelsFailed`, or raised as it is when there was no other model.
- Recording sees what each model was actually sent. The prompt is stored only when [content capture](#recording-content) is on.

## Embeddings

A touchpoint can embed as well as chat. Embedding calls go through the same resolution, admin overrides, API keys, fallbacks, retry waits and circuit breaker, and are recorded as calls with `kind="embedding"`:

```python
search = touchpoint(
    "search.embed",
    default="gemini/gemini-embedding-2",
    fallbacks=["openai/text-embedding-3-small"],
    params={"dimensions": 768},
    requires={"kind": "embedding"},
)

vectors = search.embed(["first passage", "second passage"])    # list[list[float]], one per input
[query] = await search.aembed("what is overdue?", task_type="RETRIEVAL_QUERY")
```

- `embed()` takes one input or a list. Keyword arguments, such as `dimensions` or a provider's task type, go to the backend for that call.
- `requires={"kind": "embedding", "input": ["image"]}` keeps fallbacks to embedding models that accept images, according to the catalogue.
- Fallbacks only make sense between models with the same output size: vectors from different models can't be compared. Pin `dimensions`, or use `fallbacks=False` for a call whose vectors go into an existing index.
- Inside a running agent, the call is recorded in its run and counts towards its [budget](agents.md#budgets).

Embeddings have their own backend setting, `LLM_KIT["EMBEDDING_BACKEND"]`. One touchpoint can use another backend: `touchpoint(..., backend=...)` applies to `embed()` too, so a touchpoint used only for embeddings can name its embedder there. Use `embedding_backend=` for a touchpoint that both chats and embeds with different backends. Each of these can be a name, a dotted path or the factory itself:

| Backend | How embeddings are made |
|---|---|
| `"auto"` (default) | `litellm` when installed, otherwise `langchain` |
| `"litellm"` | `litellm.embedding` / `aembedding`. Inputs are passed as given, so media parts reach providers that embed images, audio or PDFs, in the form LiteLLM accepts for that provider |
| `"langchain"` | `init_embeddings` with the provider's `langchain-*` package; text only |
| `"myapp.ai:embedder"` | your own `fn(ref, params, api_keys)` returning a LangChain `Embeddings`, or an object with `embed(inputs, **params)` and `aembed(...)` that return `Embedded(vectors, input_tokens)` (from `llm_kit.core.touchpoints.embeddings`). Use this for a provider SDK's multimodal inputs |

### One vector per input

Some models take only one input per call. On Vertex AI, Gemini embedding models other than `gemini-embedding-001` use the `embedContent` endpoint, which embeds a list as one input and returns a single vector for all of it. llm-kit always returns one vector per input:

- **Splitting:** a list is split into calls of at most the model's batch size, and the vectors come back in order. The batch size is the first of:
  1. `LLM_KIT["EMBEDDING_BATCH_SIZE"]`, for example `{"vertex_ai/gemini-embedding-2": 1, "openai/*": 2048}`. Patterns are allowed and the most specific wins; `0` means no limit.
  2. Your backend's `max_batch` attribute.
  3. What LiteLLM knows about the model (with the `litellm` backend).
  4. llm-kit's own list: one input for `vertex_ai/gemini-embedding-*` except `-001`.
- **Each call counts:** it takes its own [rate-limit](#your-accounts-rate-limits) slot and is recorded. `aembed()` runs up to `EMBEDDING_CONCURRENCY` (4) calls at once.
- **A wrong count is an error.** A reply with a different number of vectors than inputs raises `EmbeddingCountMismatch`, naming the model. It is never a short list that would pair vectors with the wrong passages.
- **Fallbacks:** when a model fails part-way through a list, the next fallback embeds the whole list again, because vectors from different models can't be mixed.

For a large one-off job where latency doesn't matter, such as re-embedding a whole corpus, consider your provider's batch API (Vertex AI batch prediction, for one). It runs outside llm-kit, often at a lower price; check your provider's pricing.

## API keys

Keys are looked up in this order:

1. **`CREDENTIALS`**: a dotted path to `fn(provider)` that returns a secret, a list of secrets, or `None`. Use it for a settings object, a vault, a service-account path, or OAuth tokens.
2. **Encrypted keys in the database** (*Provider keys* in the admin). Needs the `[crypto]` extra and `LLM_KIT["ENCRYPTION_KEYS"]`, a list of Fernet keys or passphrases, newest first, so you can rotate. Keys are never shown again after saving.
3. **Environment variables**: `ANTHROPIC_API_KEY`, or any name models.dev lists for the provider, or a comma-separated pool in `ANTHROPIC_API_KEYS`.

## Recording content

Prompts and responses are not stored by default, only hashes. `touchpoint(..., capture=True)` stores them for that touchpoint. See [Observability](observability.md).
