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

Only `video` is a [long job](#long-jobs) among the built-in kinds. Any kind can still run [in the background](#background-jobs).

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

## Speech

```python
voiceover = touchpoint(
    "media.voiceover",
    default="openai/gpt-4o-mini-tts",
    fallbacks=["elevenlabs/eleven_multilingual_v2"],
    requires={"kind": "speech"},
    model_params={"openai/*": {"voice": "alloy"}, "elevenlabs/*": {"voice": "21m00Tcm4TlvDq8ikWAM"}},
)

audio = voiceover.speak("Welcome back to your workspace.", format="mp3")
audio.first.save(name="greetings/welcome.mp3")
```

- **`speak(text, *, voice=None, format=None, speed=None, instructions=None, **params)`** and `aspeak(...)`.
- **`voice`** is the provider's voice name or id. Set it per model with [`model_params`](#settings-for-each-model): no two providers share voices.
- **`format`** is `mp3` (most providers' default), `wav`, `opus`, `aac`, `flac` or `pcm`.
- **`instructions`** steer the delivery (tone, pace) on models that take them.
- **Prices:** `units` has `characters`, and `output_seconds` when the length is known (a WAV clip, or the provider says).

## Transcription

```python
notes = touchpoint(
    "media.notes",
    default="openai/gpt-4o-transcribe",
    fallbacks=["groq/whisper-large-v3", "deepgram/nova-3"],
    requires={"kind": "transcription"},
)

result = notes.transcribe(request.FILES["audio"], language="en")
result.text

timed = notes.transcribe(clip, timestamps=["word", "segment"], requested="openai/whisper-1")
timed.segments, timed.words, timed.duration, timed.language
```

- **`transcribe(audio, *, language=None, prompt=None, format=None, timestamps=None, temperature=None, **params)`** and `atranscribe(...)`. `audio` is bytes, a URL, an uploaded file or `Media`.
- **Hints:** `language` (`"en"`) and `prompt` (names and spellings to expect) help the model.
- **`format`** is `json`, `text`, `srt`, `vtt` or `verbose_json`. With `srt` or `vtt` the subtitles are the result's `text`.
- **`timestamps=True`** (segments) or `["word", "segment"]` asks for timings, and the format becomes `verbose_json` unless you set one. Not every model gives timings.
- **Prices:** `units` has `input_seconds`, from the provider, or from the clip when it is WAV. Models billed by tokens record `input_audio_tokens`.
- **Size:** providers limit the size of a clip (OpenAI's is 25 MB). Split longer recordings before sending them.

## Video

```python
clips = touchpoint(
    "media.clips",
    default="openai/sora-2",
    fallbacks=["gemini/veo-3.0-generate-preview", "runwayml/gen4_turbo"],
    requires={"kind": "video"},
)

run = clips.start_video("Waves on a rocky shore at dusk", seconds=8, size="1280x720", user=request.user)
# returns at once: a BackgroundRun (kind="media") to watch, cancel or wait for

result = clips.video("A paper boat on a stream", image=first_frame)   # waits (MEDIA_JOB_WAIT)
```

- **`video(prompt, *, image=None, seconds=None, size=None, resolution=None, wait=None, **params)`** and `avideo(...)` wait for the video. `image` is the first frame or a reference. `wait` defaults to `MEDIA_JOB_WAIT` (900 seconds).
- **`start_video(...)`** takes the same arguments plus `owner`/`user`/`tenant` and runs the job [in the background](#background-jobs). Use it in requests: a video takes minutes.
- The video is downloaded when the job finishes. Providers keep it only for a while.
- **Prices:** `units` has `videos`, `output_seconds`, `size` and `resolution` (`720p` from `1280x720`, or as you give it). These match the catalogue's per-second prices by resolution.

## Background jobs

`tp.start_video(...)`, or `tp.start(kind, *, owner=None, user=None, tenant=None, requested=None, wait=None, params=None, **request)` for any kind, saves a `BackgroundRun` with `kind="media"` and returns it at once. It runs where [background runs](streaming.md#background-runs) run (`RUN_EXECUTOR`: a thread, Celery, Django-Q2 or your own) and has what they have:

- **Before it starts:** a caller over budget, a model they may not use or a missing key is refused when you call `start`, before anything is saved. It acts for its owner: their keys, saved settings and budgets.
- **Watching:** `background.watch(run)`, `GET llm/runs/<id>/watch/`, and the WebSocket `watch` message. A watcher gets these events:
    - `progress`: first `{"status": "submitted", "job": ...}`, then the provider's `status` and `progress`;
    - `media`: the result;
    - `status`: the run's final status.
- **Cancelling:** `run.cancel()` from any process. The provider's job is cancelled too when the backend can do that.
- **Listing:** `GET llm/api/runs/?kind=media`.
- **Lost runs:** detected as for other background runs.
- **When it ends:** the `run_finished` signal fires, and so does `llm_kit.media.signals.media_job_finished` (`run`, `status`, `result`, `error`). Use them to tell your users (a push, an email) or call your own webhooks.

When the job is done, its media are saved to storage (`MEDIA_STORAGE`, as `MEDIA_PATH/jobs/<run id>/1.mp4`). `run.result` has their `name` and `url`, plus the `text`, `units`, `cost`, `model` and the provider's `job` id. While the job runs, `run.result["job"]` already has that id.

A job waits up to `MEDIA_BACKGROUND_WAIT` seconds (3,600; per job `wait=`). Past that it ends as `error` and keeps the job id: the provider may still finish it.

A media job has no thread. The rules for who may watch, list or cancel it are the thread rules, applied to whoever started it: its user, its visitor, or staff.

Media given to a job (an image to animate) is kept with the run until it runs, up to 25 MB. Pass a URL for anything larger.

## Webhooks

A job checks on its provider every `MEDIA_POLL_EVERY` seconds. That works with every provider and stays on. A provider that sends webhooks can say when a job changed instead:

```python
LLM_KIT = {
    "MEDIA_WEBHOOKS": {
        "replicate": {"secret": "whsec_...", "param": "webhook"},
        "openai": {"secret": "whsec_..."},          # the URL is set in OpenAI's dashboard
        "fal_ai": {"param": "webhook_url"},
        "acme": {"verify": "hmac", "header": "X-Acme-Signature", "secret": "...", "id": "job.id"},
        "other": {"verify": "myapp.hooks:verify", "id": "data.job_id"},
    },
    "MEDIA_WEBHOOK_URL": "https://app.example.com/llm/media/webhooks/{provider}/",
}
```

The route is `POST llm/media/webhooks/<provider>/`, included with `llm_kit.transport.sse.urls`. Each provider's entry has these keys:

| Key | What it is |
|---|---|
| `verify` | how a webhook is checked: `"standard"` ([Standard Webhooks](https://www.standardwebhooks.com/), used by Replicate and OpenAI), `"hmac"` (an HMAC-SHA256 of the body in `header`), `"fal"` (fal's ED25519 signatures, checked with fal's published keys; needs `cryptography`), or a dotted path to your own `fn(request, config) -> bool`. Replicate, OpenAI and fal have theirs by default |
| `secret` | the signing secret, for `standard` and `hmac` |
| `id` | where the job id is in the JSON body (dotted). There are defaults for Replicate, OpenAI and fal |
| `param` | the provider parameter that takes the webhook URL when a job is submitted, sent with `MEDIA_WEBHOOK_URL`. Leave it out when the URL is set in the provider's dashboard, and check what your provider's API calls it |

What happens to a webhook:

- **A checked webhook** for a running job wakes that job, which asks the provider at once.
- **The body isn't trusted:** a webhook only says "look now". The result always comes from the provider.
- **While webhooks are on** for a provider, its jobs check only every `MEDIA_WEBHOOK_POLL_EVERY` seconds (60), in case a webhook never comes. Local development, where providers can't reach you, still finishes jobs this way.
- **Responses:**
    - a webhook that fails its check gets 401;
    - one for a provider without an entry gets 404;
    - one for no running job gets 202 and changes nothing.
- **System checks:**
    - `llm_kit.E003` reports an entry that needs a secret and has none;
    - `llm_kit.W004` reports a `param` without `MEDIA_WEBHOOK_URL`.

## Documents (OCR)

```python
reader = touchpoint("media.reader", default="mistral/mistral-ocr-latest", requires={"kind": "ocr"})

doc = reader.ocr(request.FILES["invoice"])
doc.text                      # the whole document, in Markdown
doc.pages                     # [{"index": 0, "markdown": "..."}, ...]
```

- **`ocr(document, *, pages=None, **params)`** and `aocr(...)`. `document` is a PDF or an image: bytes, a URL, a file or `Media`. `pages` picks pages (`[0, 2]`) on models that take it.
- **Annotations:** other `params` go to the provider as they are, for example Mistral's annotation formats. A document annotation comes back as `result.annotation`.
- **Prices:** `units` has `pages` (and `annotated_pages`), priced by the catalogue's `ocr_cost_per_page` and `annotation_cost_per_page`.

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

## Adding a kind of your own

Anything a provider sells can go through a touchpoint: music, 3D models, sound effects, upscaling. A kind of your own takes four steps.

**1. Register the kind** once at startup, for example in your app's `touchpoints.py`:

```python
from llm_kit.media import MediaKind, register_kind

def by_seconds(request, result):
    return {"output_seconds": result.data.get("seconds") or request.get("seconds")}

register_kind(MediaKind("music", long=True, measure=by_seconds, description="Music from a prompt."))
```

**2. Write a backend** for the provider. This one submits a job and checks on it, because songs take a while:

```python
import httpx
from llm_kit.media import DONE, FAILED, RUNNING, ContentRefused, JobState, Media, MediaResult

class AcmeMusic:
    def __init__(self, ref, params, api_keys):
        self.model = ref.split("/", 1)[1]
        self.params = params
        self.http = httpx.Client(base_url="https://api.acme.example", headers={"Authorization": f"Bearer {api_keys[0]}"})

    def submit(self, kind, request, **params):
        body = {"model": self.model, "prompt": request["prompt"], "duration": request.get("seconds"), **self.params, **params}
        response = self.http.post("/v1/songs", json=body)
        if response.status_code == 422 and "policy" in response.text:
            raise ContentRefused(response.json()["detail"])
        response.raise_for_status()
        return response.json()["id"]

    def poll(self, kind, job_id):
        song = self.http.get(f"/v1/songs/{job_id}").json()
        if song["status"] == "failed":
            return JobState(FAILED, error=song.get("error", ""))
        if song["status"] != "complete":
            return JobState(RUNNING, progress=song.get("progress"))
        audio = Media.of(song["audio_url"], "audio/mpeg")
        return JobState(DONE, result=MediaResult("music", media=[audio], data={"seconds": song["duration"]}, cost=song.get("cost")))

    def cancel(self, kind, job_id):
        self.http.delete(f"/v1/songs/{job_id}")
```

A kind that answers at once implements `generate(kind, request, **params)` instead of `submit`/`poll`. Add `agenerate`/`asubmit`/`apoll` for a native async client; without them, async calls run your methods on a thread.

**3. Tell the catalogue about the model**, if it isn't there already. Add a catalogue override in the admin (or `CatalogueOverride`) for `acme/song-2`, with `{"kind": "music", "price_units": {"output_cost_per_second": "0.002"}}`. Then `requires={"kind": "music"}` accepts it, and its calls are priced when the backend doesn't report a cost.

**4. Declare a touchpoint and use it:**

```python
songs = touchpoint(
    "media.songs",
    default="acme/song-2",
    fallbacks=["acme/song-1"],
    requires={"kind": "music"},
    media_backend="myapp.media:AcmeMusic",
)

result = songs.generate("music", prompt="Upbeat jingle for a bakery", seconds=15)   # waits
run = songs.start("music", prompt="Upbeat jingle for a bakery", seconds=15, user=request.user)  # background
```

It gets the same keys, rate limits, fallbacks, refusals, records, prices, budgets, background jobs and [webhooks](#webhooks) as the built-in kinds. A provider's keys come from the usual places (`CREDENTIALS`, stored keys, `ACME_API_KEY`), and a tenant's or user's own keys work too.

**Testing:** point `MEDIA_BACKEND` (or `media_backend=`) at a fake factory that returns `MediaResult`s, as llm-kit's own tests do, so tests never reach the provider.
