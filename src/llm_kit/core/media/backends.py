"""Who makes the media: a backend per model, built by a factory like the embedding one.

``touchpoint(..., media_backend=)``, else ``LLM_KIT["MEDIA_BACKEND"]``: a name below, a dotted
path, or the factory itself. A factory is ``fn(ref, params, api_keys)`` returning a backend::

    class MyMusic:
        def __init__(self, ref, params, api_keys): ...
        def generate(self, kind, request, **params) -> MediaResult: ...
        async def agenerate(self, kind, request, **params) -> MediaResult: ...
        # a long kind (video, music): submit, then check until done
        def submit(self, kind, request, **params) -> str: ...        # the provider's job id
        def poll(self, kind, job_id) -> JobState: ...
        async def asubmit(...) / apoll(...)                           # optional: else run on a thread
        def cancel(self, kind, job_id) -> None: ...                   # optional

* ``litellm`` (and ``auto``, the default): LiteLLM's ``image_generation``, ``image_edit``,
  ``speech``, ``transcription``, ``ocr`` and ``video_generation`` / ``video_status`` /
  ``video_content``, for every provider LiteLLM supports.
* a dotted path or a factory: any other provider, or a kind of your own.
"""

from __future__ import annotations

import asyncio
import base64
import random
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from llm_kit.core import config
from llm_kit.core.errors import ConfigError
from llm_kit.core.graphs import import_target
from llm_kit.core.media.types import MediaResult

MediaFactory = Callable[[str, dict[str, Any], list[str]], Any]

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"

# Parameters that say where and how to reach a provider, kept for status checks and downloads.
CONNECTION_PARAMS = (
    "api_base",
    "base_url",
    "api_version",
    "organization",
    "project",
    "custom_llm_provider",
    "extra_headers",
    "timeout",
    "vertex_project",
    "vertex_location",
    "vertex_credentials",
    "aws_region_name",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
)


@dataclass
class JobState:
    """Where a long job is at the provider."""

    status: str  # pending | running | done | failed
    progress: float | None = None  # 0-100 when the provider says
    error: str = ""
    result: MediaResult | None = None  # when done

    @property
    def finished(self) -> bool:
        return self.status in (DONE, FAILED)


@runtime_checkable
class MediaBackend(Protocol):
    def generate(self, kind: str, request: dict[str, Any], **params: Any) -> MediaResult: ...


def supports_jobs(backend: Any) -> bool:
    return callable(getattr(backend, "submit", None)) and callable(getattr(backend, "poll", None))


async def agenerate(backend: Any, kind: str, request: dict[str, Any], **params: Any) -> MediaResult:
    if callable(getattr(backend, "agenerate", None)):
        return await backend.agenerate(kind, request, **params)
    return await asyncio.to_thread(backend.generate, kind, request, **params)


async def asubmit(backend: Any, kind: str, request: dict[str, Any], **params: Any) -> str:
    if callable(getattr(backend, "asubmit", None)):
        return await backend.asubmit(kind, request, **params)
    return await asyncio.to_thread(backend.submit, kind, request, **params)


async def apoll(backend: Any, kind: str, job_id: str) -> JobState:
    if callable(getattr(backend, "apoll", None)):
        return await backend.apoll(kind, job_id)
    return await asyncio.to_thread(backend.poll, kind, job_id)


async def acancel(backend: Any, kind: str, job_id: str) -> None:
    if callable(getattr(backend, "acancel", None)):
        await backend.acancel(kind, job_id)
    elif callable(getattr(backend, "cancel", None)):
        await asyncio.to_thread(backend.cancel, kind, job_id)


# ------------------------------------------------------------------ LiteLLM


def _litellm():
    from llm_kit.core.touchpoints.backends import _quiet_litellm_import

    _quiet_litellm_import()
    try:
        import litellm
    except ImportError as e:
        raise ConfigError("The litellm media backend needs: pip install django-llm-kit[litellm]") from e
    return litellm


def get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def reported_cost(response: Any) -> Any:
    """The cost LiteLLM worked out for a response, when it did."""
    hidden = getattr(response, "_hidden_params", None) or {}
    cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
    return cost if cost not in (None, 0, 0.0) else None


class LiteLLMMedia:
    """Media through LiteLLM. Each kind is a pair of methods, ``_<kind>`` and ``_a<kind>``."""

    def __init__(self, ref: str, params: dict[str, Any], keys: list[str]) -> None:
        self.ref, self.params, self.keys = ref, dict(params), list(keys)

    def _kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        kwargs = {**self.params, **params}
        if self.keys:
            kwargs.setdefault("api_key", random.choice(self.keys))
        return kwargs

    def _op(self, kind: str, prefix: str = "_") -> Callable[..., Any]:
        op = getattr(self, f"{prefix}{kind}", None)
        if op is None:
            raise ConfigError(f"The litellm media backend doesn't make {kind!r}; give the touchpoint a media_backend.")
        return op

    def generate(self, kind: str, request: dict[str, Any], **params: Any) -> MediaResult:
        return self._op(kind)(request, self._kwargs(params))

    async def agenerate(self, kind: str, request: dict[str, Any], **params: Any) -> MediaResult:
        return await self._op(kind, "_a")(request, self._kwargs(params))

    # documents

    def _ocr(self, request, kwargs):
        return ocr_result(_litellm().ocr(**_ocr_args(self, request, kwargs)))

    async def _aocr(self, request, kwargs):
        args = await asyncio.to_thread(_ocr_args, self, request, kwargs)  # reading the document may fetch it
        return ocr_result(await _litellm().aocr(**args))

    # long jobs: _submit_<kind>, _poll_<kind> and their async forms

    def _key(self) -> dict[str, Any]:
        """What a status check needs: the key and where the provider is (not generation params)."""
        where = {k: v for k, v in self.params.items() if k in CONNECTION_PARAMS}
        return {**where, "api_key": random.choice(self.keys)} if self.keys else where

    def submit(self, kind: str, request: dict[str, Any], **params: Any) -> str:
        return self._op(kind, "_submit_")(request, self._kwargs(params))

    async def asubmit(self, kind: str, request: dict[str, Any], **params: Any) -> str:
        return await self._op(kind, "_asubmit_")(request, self._kwargs(params))

    def poll(self, kind: str, job_id: str) -> JobState:
        return self._op(kind, "_poll_")(job_id, self._key())

    async def apoll(self, kind: str, job_id: str) -> JobState:
        return await self._op(kind, "_apoll_")(job_id, self._key())

    @staticmethod
    def provider_job_id(job_id: str) -> str:
        """The provider's own id for a job (LiteLLM's video ids carry the provider and model)."""
        try:
            decoded = _litellm().decode_video_id_with_provider(job_id)
        except Exception:
            return job_id
        return str(get(decoded, "video_id") or job_id)

    # video

    def _submit_video(self, request, kwargs):
        return str(get(_litellm().video_generation(**_video_args(self, request, kwargs)), "id"))

    async def _asubmit_video(self, request, kwargs):
        args = await asyncio.to_thread(_video_args, self, request, kwargs)  # reading the image may fetch it
        return str(get(await _litellm().avideo_generation(**args), "id"))

    def _poll_video(self, job_id, kwargs):
        litellm = _litellm()
        status = litellm.video_status(video_id=job_id, **kwargs)
        state = video_state(status)
        if state.status == DONE:
            state.result = video_result(status, litellm.video_content(video_id=job_id, **kwargs))
        return state

    async def _apoll_video(self, job_id, kwargs):
        litellm = _litellm()
        status = await litellm.avideo_status(video_id=job_id, **kwargs)
        state = video_state(status)
        if state.status == DONE:
            state.result = video_result(status, await litellm.avideo_content(video_id=job_id, **kwargs))
        return state

    # images

    def _image(self, request, kwargs):
        return images_result("image", _litellm().image_generation(**_image_args(self, request, kwargs)))

    async def _aimage(self, request, kwargs):
        return images_result("image", await _litellm().aimage_generation(**_image_args(self, request, kwargs)))

    def _image_edit(self, request, kwargs):
        return images_result("image_edit", _litellm().image_edit(**_edit_args(self, request, kwargs)))

    async def _aimage_edit(self, request, kwargs):
        args = await asyncio.to_thread(_edit_args, self, request, kwargs)  # reading the images may fetch them
        return images_result("image_edit", await _litellm().aimage_edit(**args))

    # speech and transcription

    def _speech(self, request, kwargs):
        return speech_result(request, _litellm().speech(**_speech_args(self, request, kwargs)))

    async def _aspeech(self, request, kwargs):
        return speech_result(request, await _litellm().aspeech(**_speech_args(self, request, kwargs)))

    def _transcription(self, request, kwargs):
        return transcription_result(_litellm().transcription(**_transcription_args(self, request, kwargs)))

    async def _atranscription(self, request, kwargs):
        args = await asyncio.to_thread(_transcription_args, self, request, kwargs)  # reading the audio may fetch it
        return transcription_result(await _litellm().atranscription(**args))


_factories: dict[str, MediaFactory] = {}
_lock = threading.Lock()


def get_media_factory(name: str | MediaFactory | None) -> MediaFactory:
    """A built-in backend by name, your factory by dotted path, or your factory itself."""
    if callable(name):
        return name
    name = name or config.get("MEDIA_BACKEND") or "auto"
    if callable(name):
        return name
    if name in ("auto", "litellm"):
        return LiteLLMMedia
    with _lock:
        if name not in _factories:
            factory = import_target(name)
            if not callable(factory):
                raise ConfigError(f"Media backend {name!r} is not callable.")
            _factories[name] = factory
    return _factories[name]


# ------------------------------------------------------------------ LiteLLM: images


def _usage_numbers(usage: Any) -> tuple[int | None, int | None, dict[str, Any]]:
    """(text tokens in, tokens out, image-token units) from an image response's usage."""
    if usage is None:
        return None, None, {}
    tokens_in, tokens_out = get(usage, "input_tokens"), get(usage, "output_tokens")
    details = get(usage, "input_tokens_details") or {}
    image_in, text_in = get(details, "image_tokens"), get(details, "text_tokens")
    units = {}
    if image_in:
        units["input_image_tokens"] = image_in
    if tokens_out:
        units["output_image_tokens"] = tokens_out
    return (text_in if text_in is not None else tokens_in), None, units


def images_result(kind: str, response: Any) -> MediaResult:
    """A LiteLLM ``ImageResponse`` as a ``MediaResult``: base64 images decoded, URLs kept to fetch."""
    import base64

    from llm_kit.core.media.types import Media, sniff

    fmt = get(response, "output_format")
    mime = f"image/{'jpeg' if fmt == 'jpg' else fmt}" if fmt else ""
    media = []
    for item in get(response, "data") or []:
        meta = {}
        if revised := get(item, "revised_prompt"):
            meta["revised_prompt"] = revised
        if b64 := get(item, "b64_json"):
            data = base64.b64decode(b64)
            media.append(Media(data=data, mime_type=sniff(data) or mime, meta=meta))
        elif url := get(item, "url"):
            media.append(Media(url=url, mime_type=mime, meta=meta))
    text_in, _, units = _usage_numbers(get(response, "usage"))
    data = {k: v for k in ("size", "quality", "background") if (v := get(response, k))}
    revised = [m.meta["revised_prompt"] for m in media if "revised_prompt" in m.meta]
    if revised:
        data["revised_prompt"] = revised[0]
    return MediaResult(kind, media=media, data=data, units=units, cost=reported_cost(response), input_tokens=text_in)


def _clean(request: dict[str, Any], *names: str) -> dict[str, Any]:
    return {n: request[n] for n in names if request.get(n) is not None}


def _image_args(self: LiteLLMMedia, request: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": self.ref,
        "prompt": request["prompt"],
        **_clean(request, "n", "size", "quality", "style"),
        **kwargs,
    }


def _edit_args(self: LiteLLMMedia, request: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    images = [m.as_file() for m in request["images"]]
    args = {"model": self.ref, "prompt": request["prompt"], "image": images if len(images) > 1 else images[0]}
    if request.get("mask") is not None:
        args["mask"] = request["mask"].as_file()
    return {**args, **_clean(request, "n", "size", "quality"), **kwargs}


# ------------------------------------------------------------------ LiteLLM: speech and transcription

AUDIO_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/L16",
    "ogg": "audio/ogg",
}


def _speech_args(self: LiteLLMMedia, request: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    args = {"model": self.ref, "input": request["text"], **kwargs}
    if request.get("format"):
        args["response_format"] = request["format"]
    return {**args, **_clean(request, "voice", "speed", "instructions")}


def speech_result(request: dict[str, Any], response: Any) -> MediaResult:
    from llm_kit.core.media.types import Media, audio_seconds, sniff

    data = response if isinstance(response, (bytes, bytearray)) else getattr(response, "content", None)
    if data is None and callable(getattr(response, "read", None)):
        data = response.read()
    data = bytes(data or b"")
    mime = sniff(data) or AUDIO_TYPES.get(str(request.get("format") or "mp3"), "audio/mpeg")
    media = Media(data=data, mime_type=mime) if data else None
    seconds = audio_seconds(media) if media is not None else None
    return MediaResult(
        "speech",
        media=[media] if media is not None else [],
        data={"duration": seconds} if seconds else {},
        cost=reported_cost(response),
    )


def _transcription_args(self: LiteLLMMedia, request: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    args = {"model": self.ref, "file": request["audio"].upload(), **kwargs}
    fmt = request.get("format")
    timestamps = request.get("timestamps")
    if timestamps:
        args["timestamp_granularities"] = list(timestamps)
        fmt = fmt or "verbose_json"
    if fmt:
        args["response_format"] = fmt
    return {**args, **_clean(request, "language", "prompt", "temperature")}


def transcription_result(response: Any) -> MediaResult:
    data: dict[str, Any] = {}
    for name in ("language", "duration", "segments", "words"):
        value = get(response, name)
        if value is not None:
            data[name] = _plain(value)
    usage = get(response, "usage")
    units: dict[str, Any] = {}
    tokens_in = tokens_out = None
    if usage is not None:
        if get(usage, "type") == "duration" and get(usage, "seconds") is not None:
            data.setdefault("duration", get(usage, "seconds"))
        tokens_out = get(usage, "output_tokens")
        details = get(usage, "input_token_details") or get(usage, "input_tokens_details") or {}
        audio_tokens = get(details, "audio_tokens")
        if audio_tokens:
            units["input_audio_tokens"] = audio_tokens
            tokens_in = get(details, "text_tokens") or 0
        else:
            tokens_in = get(usage, "input_tokens")
    text = get(response, "text")
    if text is None and isinstance(response, str):
        text = response
    return MediaResult(
        "transcription",
        text=text or "",
        data=data,
        units=units,
        cost=reported_cost(response),
        input_tokens=tokens_in,
        output_tokens=tokens_out,
    )


def _plain(value: Any) -> Any:
    """Provider objects as plain lists and dicts."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    dump = getattr(value, "model_dump", None) or getattr(value, "dict", None)
    if callable(dump):
        return _plain(dump())
    return value


# ------------------------------------------------------------------ LiteLLM: video

_VIDEO_STATUS = {"queued": PENDING, "pending": PENDING, "in_progress": RUNNING, "processing": RUNNING}


def _video_args(self: LiteLLMMedia, request: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    args = {"model": self.ref, "prompt": request["prompt"], **kwargs}
    if request.get("image") is not None:
        args["input_reference"] = _reference(self.ref, request["image"])
    if request.get("seconds") is not None:
        args["seconds"] = str(request["seconds"])
    if request.get("size") is not None:
        args["size"] = request["size"]
    return args


# Providers whose video APIs take the reference image as a public URL, not an upload.
URL_REFERENCES = ("fal_ai",)


def _reference(ref: str, image: Any) -> Any:
    from llm_kit.core.errors import BadRequest
    from llm_kit.core.touchpoints.env import provider_of

    if provider_of(ref) in URL_REFERENCES:
        if image.url is None:
            raise BadRequest(f"{ref} takes the image as a public URL: pass image='https://...'.")
        return image.url
    return image.as_file()


def video_state(status: Any) -> JobState:
    """A LiteLLM ``VideoObject`` as a ``JobState`` (without the result)."""
    raw = str(get(status, "status") or "").lower()
    progress = get(status, "progress")
    if raw in ("completed", "succeeded", "success", "done"):
        return JobState(DONE, progress=100)
    if raw in ("failed", "error", "cancelled", "canceled", "expired"):
        error = get(status, "error")
        message = get(error, "message") if error is not None and not isinstance(error, str) else error
        return JobState(FAILED, progress=progress, error=str(message or raw))
    return JobState(_VIDEO_STATUS.get(raw, RUNNING), progress=progress)


def video_result(status: Any, content: Any) -> MediaResult:
    from llm_kit.core.media.types import Media, sniff

    data = bytes(content or b"") if not hasattr(content, "content") else bytes(content.content)
    info = {k: v for k in ("seconds", "size") if (v := get(status, k)) is not None}
    if "seconds" in info:
        try:
            info["seconds"] = float(info["seconds"])
        except (TypeError, ValueError):
            info.pop("seconds")
    media = Media(data=data, mime_type=sniff(data) or "video/mp4", meta=dict(info))
    return MediaResult(
        "video", media=[media], data=info, cost=reported_cost(status), served_model=str(get(status, "model") or "")
    )


# ------------------------------------------------------------------ LiteLLM: OCR


def _ocr_args(self: LiteLLMMedia, request: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    from llm_kit.core.media.types import sniff

    document = request["document"]
    if document.url is not None and document.data is None:
        location = document.url
        is_image = (document.mime_type or "").startswith("image/")
    else:
        data = document.read()
        mime = document.mime_type or sniff(data) or "application/pdf"
        location = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        is_image = mime.startswith("image/")
    kind = "image_url" if is_image else "document_url"
    args = {"model": self.ref, "document": {"type": kind, kind: location}, **kwargs}
    if request.get("pages") is not None:
        args["pages"] = list(request["pages"])
    return args


def ocr_result(response: Any) -> MediaResult:
    pages = []
    for page in get(response, "pages") or []:
        pages.append({"index": get(page, "index"), "markdown": get(page, "markdown") or ""})
    usage = get(response, "usage_info") or {}
    units = {}
    if (processed := get(usage, "pages_processed")) is not None:
        units["pages"] = processed
    if annotated := get(usage, "pages_processed_annotation"):
        units["annotated_pages"] = annotated
    data: dict[str, Any] = {"pages": pages}
    if (annotation := get(response, "document_annotation")) is not None:
        data["annotation"] = annotation
    text = "\n\n".join(p["markdown"] for p in pages)
    return MediaResult(
        "ocr",
        text=text,
        data=data,
        units=units,
        cost=reported_cost(response),
        served_model=str(get(response, "model") or ""),
    )
