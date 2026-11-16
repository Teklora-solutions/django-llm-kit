---
description: Generate images, video and speech, and transcribe or read documents, through touchpoints with the same keys, fallbacks, records and budgets as chat.
---

# Generated media

A touchpoint can make media as well as text: images, video, speech, transcripts, the text of documents, and any kind you add. A media call gets what a chat call gets:

- **Keys:** your platform keys, a tenant's or a user's keys, and keyless providers.
- **Resilience:** rate limits, retry waits and fallbacks to other models and providers.
- **Records and budgets:** an `LLMCall` row for each attempt, with what it made and what it cost. That cost counts towards the touchpoint's, tenant's, user's and request's budgets.

LiteLLM makes it for every provider it supports (OpenAI, Google, fal, Replicate, Runway, xAI, ElevenLabs, Bedrock and many more). Any other provider plugs in through [your own backend](#your-own-backend).

```python
from llm_kit import touchpoint

covers = touchpoint(
    "media.covers",
    default="openai/gpt-image-1",
    fallbacks=["gemini/imagen-4.0-generate-001"],
    requires={"kind": "image"},
)

result = covers.generate("image", prompt="A lighthouse at dawn, watercolour", n=2)
names = result.save()                 # into Django storage; returns the saved names
```

## Calls

`tp.generate(kind, **request)` and `await tp.agenerate(...)` make media of any [kind](#kinds). The request's fields depend on the kind (`prompt=`, `text=`, `audio=`, ...). Each kind also has its own method with named arguments.

`generate()` takes the same options as `complete()` and `embed()`:

- `requested=`: a model the caller asked for, checked against the touchpoint's [choice policy](touchpoints.md#who-picks-the-model).
- `config=`, `caller=`: who the call is for (their keys, saved settings and budgets).
- `retry=`, `fallbacks=False`: how long a short rate limit is waited out, and whether fallbacks are used.
- `params={...}`: given to the provider as they are.
- `wait=`: for a [long job](#long-jobs), how many seconds to wait for it.

It returns a `MediaResult`:

| Field | What it holds |
|---|---|
| `media` | what was made, as `Media` (`result.first` for the first) |
| `text` | what was read: a transcript, a document's text |
| `data` | the rest the provider said: `revised_prompt`, `segments`, `language`, `duration`... (also as attributes: `result.language`) |
| `units` | what the call was billed by: `images`, `size`, `output_seconds`, `characters`... |
| `cost` | USD, when known |
| `model` | the model that answered |
| `job_id` | the provider's job, for a long kind |

## Kinds

| Kind | What it does | Catalogue `kind` |
|---|---|---|
| `image` | images from a prompt | `image` |
| `image_edit` | changes to images it is given | `image_edit` |
| `video` | video from a prompt, optionally an image ([a long job](#long-jobs)) | `video` |
| `speech` | text read aloud | `speech` |
| `transcription` | speech written down | `transcription` |
| `ocr` | the text of a document or image | `ocr` |

The catalogue labels each media model with what it does, from LiteLLM's model data, so `requires={"kind": "transcription"}` keeps a speech-to-text touchpoint's fallbacks to speech-to-text models. The catalogue also has `realtime`, `music`, `rerank` and `moderation`.

### Kinds of your own

Register a kind for anything a provider sells, and give the touchpoint a [backend](#your-own-backend) that makes it:

```python
from llm_kit.media import MediaKind, register_kind

def by_length(request, result):
    return {"output_seconds": request.get("seconds")}

register_kind(MediaKind("music", long=True, measure=by_length))

jingles = touchpoint("media.jingles", default="acme/music-2", media_backend="myapp.media:AcmeMusic")
song = jingles.generate("music", prompt="upbeat, 15 seconds", seconds=15)
```

`long=True` makes it a [long job](#long-jobs). `measure(request, result)` returns what the call is billed by (see [Prices](#records-and-prices)). Units the backend sets on the result win.

## Images

```python
covers = touchpoint(
    "media.covers",
    default="openai/gpt-image-1",
    fallbacks=["gemini/imagen-4.0-generate-001", "fal_ai/fal-ai/flux-pro/v1.1"],
    requires={"kind": "image"},
    model_params={"gemini/*": {"aspect_ratio": "16:9"}, "openai/*": {"size": "1536x1024"}},
)

result = covers.image("A lighthouse at dawn, watercolour", n=2, quality="high")
result.media                     # two Media, already downloaded
result.revised_prompt            # when the model rewrote the prompt
result.save(name="covers/lighthouse")

edited = editor.edit_image("Make it night", result.first, mask=mask_png)
```

- **`image(prompt, *, n=None, size=None, quality=None, style=None, fetch=True, **params)`** and `aimage(...)`. `size`, `quality` and `style` go to models that take them. Other `params` (`aspect_ratio`, `output_format`, `background`, `seed`...) go to the provider as they are.
- **`edit_image(prompt, images, *, mask=None, n=None, size=None, quality=None, fetch=True, **params)`** and `aedit_image(...)`. `images` is one image or a list (bytes, URLs, files, `Media`). Where `mask` is transparent, the image may change.
- **Downloads:** images the provider returns as URLs are downloaded before the call returns, because those URLs often expire within the hour. `fetch=False` keeps the URLs instead.
- **Prices:** `units` has `images`, `size` and `quality`, `input_images` for edits, and image tokens for models billed by tokens. LiteLLM's reported cost is used when it gives one.

## Settings for each model

Providers don't agree on settings. OpenAI's `alloy` voice doesn't exist at ElevenLabs, and one image API takes `size="1024x1024"` where another takes `aspect_ratio="1:1"`. `model_params` gives each model its own, over the touchpoint's `params`:

```python
voices = touchpoint(
    "media.voiceover",
    default="openai/gpt-4o-mini-tts",
    fallbacks=["elevenlabs/eleven_multilingual_v2"],
    params={"format": "mp3"},
    model_params={
        "openai/*": {"voice": "alloy"},
        "elevenlabs/*": {"voice": "21m00Tcm4TlvDq8ikWAM"},
    },
)
```

Keys are model refs or patterns; when several match, the most specific is applied last. Parameters passed to a call go to every model, so leave settings that differ by provider to `model_params`. `model_params` works for chat and embedding calls too.

## Refusals

When a provider refuses a request under its content policy, the call raises `ContentRefused` and the next model is **not** tried. Quietly sending a refused prompt to a more permissive provider is a decision for your app, not the kit. Allow it per touchpoint:

```python
touchpoint("media.drafts", default=..., fallbacks=[...], fallback_on_refusal=True)
```

Refusals are recorded with `error_class="refused"`. Outages, rate limits and other failures fall back as usual. Raise `ContentRefused` from your own backend to say the same.

## Long jobs

Some media takes minutes: you submit a job, check on it until it's done, then fetch the result. A kind with `long=True` (`video`, and your own) works like that when its backend can submit jobs:

- `generate()` submits the job, then checks it every `MEDIA_POLL_EVERY` seconds (5) until it is done, for at most `wait=` seconds (`MEDIA_JOB_WAIT`, 900).
- A job that fails at the provider falls back like any failure. A refused one raises `ContentRefused`.
- Past the wait, `MediaJobTimeout` carries the provider's `job_id`, and the next model is **not** tried: the job may still finish, and be billed.

## Media in and out

`Media.of(value)` takes what you have: bytes, an http(s) URL, a `data:` URI, an open file, a Django `File`, or `{"url": ...}` / `{"data": <base64>, "mime_type": ...}`. Results hold `Media` too. A string is never read as a file path, because it may have come from a user. For a file of your own, pass a `Path` or use `Media.from_path(path)`.

| | |
|---|---|
| `read()`, `await aread()` | all of it, in memory |
| `chunks()` | it in pieces, without loading it all |
| `save(storage=None, name=None)`, `await asave(...)` | streams it into Django storage. `storage` defaults to `STORAGES[MEDIA_STORAGE]`, and `name` to a fresh name under `MEDIA_PATH` (`llm_kit/media/`). Returns the saved name |
| `to_json()` | its type, name, URL and details, never its bytes |

`MediaResult.save(storage, name)` saves each piece (`name-1.png`, `name-2.png`, ...).

- A download is spooled to a temporary file past 8 MB, so a large video isn't held in memory.
- Provider URLs often expire after an hour, so save what you want to keep.
- Media larger than `MEDIA_MAX_BYTES` (1 GiB) raises `MediaTooLarge`.
- The kit fetches public http(s) URLs only. A URL on a private address (your own network, the cloud's metadata service) raises `UnsafeURL`, unless `MEDIA_FETCH_PRIVATE` is on. The address is checked when each connection opens, redirects included, so a name that changes what it resolves to can't get past the check. URLs your users give are checked the same way.

## Records and prices

Each attempt is an `LLMCall` row:

- `kind` is the media kind.
- `units` says what it was billed by: `images`, `size`, `input_images`, `input_seconds` (audio heard), `output_seconds` (audio or video made), `resolution`, `characters`, `videos`, `pages`, and image or audio tokens.
- `provider_request_id` is the job id of a long job.
- With content capture on, `prompt` holds the request and `completion` the text and a description of the media. Neither ever holds the bytes.

Its cost comes from, in order:

1. what the backend reported (LiteLLM works out a cost for most of its media calls);
2. the units times the catalogue's prices. The catalogue keeps LiteLLM's per-unit prices (`output_cost_per_image`, `output_cost_per_image_2K`, `output_cost_per_second_720p`, `input_cost_per_second`, `input_cost_per_character`, `output_cost_per_pixel`, ...) under `price_units`, and the per-token prices for models billed by tokens;
3. otherwise the call is unpriced.

The cost counts towards every budget a chat call's would: the touchpoint's `daily_budget`, the tenant's and user's budgets, and the request's `AGENT_BUDGET`.

## Your own backend

`touchpoint(..., media_backend=)` or `LLM_KIT["MEDIA_BACKEND"]` names the backend: `"litellm"` (the default; `"auto"` is the same), a dotted path, or the factory itself. A factory is `fn(ref, params, api_keys)` returning an object with:

```python
from llm_kit.media import JobState, Media, MediaResult

class AcmeMusic:
    def __init__(self, ref, params, api_keys): ...

    def generate(self, kind, request, **params) -> MediaResult: ...
    async def agenerate(self, kind, request, **params) -> MediaResult: ...   # optional

    # for a long kind
    def submit(self, kind, request, **params) -> str: ...                   # the provider's job id
    def poll(self, kind, job_id) -> JobState: ...                           # "pending", "running", "done" (with result) or "failed"
    def cancel(self, kind, job_id) -> None: ...                             # optional
```

Methods without an async version run on a thread in async calls. Return `MediaResult(kind, media=[Media.of(data)], cost=...)`, setting `cost` or `units` when the provider says what it billed.
