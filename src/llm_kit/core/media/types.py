"""Media going into and coming out of a model: an image, a clip of audio or video, a document.

``Media.of(value)`` takes what you have (bytes, a path, a URL, a ``data:`` URI, an open file, a
Django ``File``); results hold ``Media`` too. Nothing is loaded until it is needed, and
``save()`` streams into Django storage, so a large video is never held in memory by the kit.
"""

from __future__ import annotations

import base64
import contextlib
import ipaddress
import mimetypes
import os
import socket
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from llm_kit.core import config
from llm_kit.core.errors import BadRequest, LLMKitError

CHUNK = 1 << 20  # bytes per piece when streaming
SPOOL = 8 << 20  # a download larger than this goes to a temporary file, not memory


class MediaTooLarge(LLMKitError):
    """Media over ``MEDIA_MAX_BYTES``."""


class UnsafeURL(BadRequest):
    """A URL the kit won't fetch: not http(s), or it points at a private address."""


def _max_bytes() -> int:
    return int(config.get("MEDIA_MAX_BYTES") or 0)


def _check_url(url: str) -> None:
    """Fetch only http(s) URLs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise UnsafeURL(f"Only http(s) URLs can be fetched, not {url[:80]!r}.")


def public_address(host: str, port: int) -> str:
    """An address of ``host`` to connect to, refusing a host with any private address
    (``MEDIA_FETCH_PRIVATE`` allows them): a URL a user gave must not reach the server's own
    network. Called when connecting, so a name that resolves differently later can't slip by."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as e:
        raise UnsafeURL(f"Cannot resolve {host}: {e}") from e
    addresses = [info[4][0] for info in infos]
    if not config.get("MEDIA_FETCH_PRIVATE"):
        for address in addresses:
            if not ipaddress.ip_address(address.split("%")[0]).is_global:
                raise UnsafeURL(f"{host} is a private address; the kit fetches public URLs only.")
    return addresses[0]


def _data_uri(value: str) -> tuple[bytes, str]:
    head, _, payload = value.partition(",")
    mime = head[5:].split(";")[0] or "application/octet-stream"
    if ";base64" in head:
        return base64.b64decode(payload), mime
    from urllib.parse import unquote_to_bytes

    return unquote_to_bytes(payload), mime


_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/webm": ".webm",
    "audio/L16": ".pcm",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "application/pdf": ".pdf",
}


def extension_for(mime_type: str) -> str:
    return _EXTENSIONS.get(mime_type) or mimetypes.guess_extension(mime_type or "") or ".bin"


def sniff(data: bytes) -> str:
    """The type of ``data`` from its first bytes ("" when unknown)."""
    head = data[:16]
    if head.startswith(b"\x89PNG"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio/wav"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if head.startswith(b"OggS"):
        return "audio/ogg"
    if head.startswith(b"fLaC"):
        return "audio/flac"
    if head[4:8] == b"ftyp":
        return "video/quicktime" if head[8:10] == b"qt" else "video/mp4"
    if head.startswith(b"\x1aE\xdf\xa3"):
        return "video/webm"
    if head.startswith(b"%PDF"):
        return "application/pdf"
    return ""


@dataclass
class Media:
    """One piece of media. Exactly one of ``data``, ``url``, ``path`` or ``file`` holds it."""

    data: bytes | None = None
    url: str | None = None
    path: str | None = None
    file: Any = None  # an open binary file or a Django File
    mime_type: str = ""
    name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)  # what the provider said: revised_prompt, seconds...

    # ------------------------------------------------------------ making one

    @classmethod
    def of(cls, value: Any, mime_type: str = "", name: str = "") -> Media:
        """``value`` as ``Media``: bytes, an http(s) URL, a ``data:`` URI, an open binary file, a
        Django ``File``, ``{"url": ...}`` / ``{"data": <base64>, "mime_type": ...}``, ``Media``,
        or a file on disk as a ``Path``. A string is never read as a path (it may have come from a
        user): use ``Path(...)`` or ``Media.from_path()`` for files of your own."""
        if isinstance(value, Media):
            return value
        if isinstance(value, (bytes, bytearray, memoryview)):
            data = bytes(value)
            return cls(data=data, mime_type=mime_type or sniff(data), name=name)
        if isinstance(value, Path):
            return cls(path=str(value), mime_type=mime_type or _guess(str(value)), name=name or value.name)
        if isinstance(value, str):
            if value.startswith("data:"):
                data, mime = _data_uri(value)
                return cls(data=data, mime_type=mime_type or mime, name=name)
            if value.startswith(("http://", "https://")):
                return cls(url=value, mime_type=mime_type or _guess(urlparse(value).path), name=name)
            raise UnsafeURL(
                f"{value[:80]!r} is not an http(s) URL or a data: URI. For a file of your own, pass a Path."
            )
        if isinstance(value, dict):
            if value.get("url"):
                if not isinstance(value["url"], str):
                    raise BadRequest("A media dict's 'url' must be a string.")
                return cls.of(value["url"], value.get("mime_type", mime_type), value.get("name", name))
            if value.get("data") is not None:
                data = value["data"]
                data = base64.b64decode(data) if isinstance(data, str) else bytes(data)
                return cls(data=data, mime_type=value.get("mime_type") or mime_type or sniff(data), name=name)
            raise BadRequest("A media dict needs 'url' or 'data'.")
        if hasattr(value, "read"):
            file_name = os.path.basename(str(getattr(value, "name", "") or ""))
            return cls(file=value, mime_type=mime_type or _guess(file_name), name=name or file_name)
        raise BadRequest(f"Can't use {type(value).__name__} as media.")

    @classmethod
    def from_path(cls, path: str | Path, mime_type: str = "", name: str = "") -> Media:
        """A file on disk. Only for paths your code chose, never one a user gave."""
        return cls.of(Path(path), mime_type, name)

    # ------------------------------------------------------------ reading

    def read(self) -> bytes:
        """All of it, in memory (fetched once when it is a URL). Prefer ``save()`` for video."""
        if self.data is None:
            limit = _max_bytes()
            out = bytearray()
            for piece in self.chunks():
                out += piece
                if limit and len(out) > limit:
                    raise MediaTooLarge(f"Media is larger than MEDIA_MAX_BYTES ({limit} bytes).")
            self.data = bytes(out)
            if not self.mime_type:
                self.mime_type = sniff(self.data)
        return self.data

    async def aread(self) -> bytes:
        if self.data is not None:
            return self.data
        import asyncio

        return await asyncio.to_thread(self.read)

    def chunks(self, size: int = CHUNK) -> Iterator[bytes]:
        """It in pieces, without loading it all."""
        if self.data is not None:
            for i in range(0, len(self.data), size):
                yield self.data[i : i + size]
        elif self.path is not None:
            with open(self.path, "rb") as f:
                while piece := f.read(size):
                    yield piece
        elif self.file is not None:
            with contextlib.suppress(Exception):
                self.file.seek(0)
            while piece := self.file.read(size):
                yield piece
        elif self.url is not None:
            yield from _download(self.url, size)
        else:
            raise BadRequest("This media holds nothing.")

    @property
    def extension(self) -> str:
        if self.name and "." in self.name:
            return "." + self.name.rsplit(".", 1)[1]
        return extension_for(self.mime_type)

    def data_uri(self) -> str:
        return f"data:{self.mime_type or 'application/octet-stream'};base64,{base64.b64encode(self.read()).decode()}"

    def b64(self) -> str:
        return base64.b64encode(self.read()).decode()

    def as_file(self) -> Any:
        """It as an in-memory binary file with a ``name`` (what some HTTP clients need)."""
        import io

        name, data, _ = self.upload()
        file = io.BytesIO(data)
        file.name = name
        return file

    def upload(self) -> tuple[str, bytes, str]:
        """``(name, bytes, mime type)``, the way HTTP clients take a file."""
        data = self.read()
        return (self.name or f"file{self.extension}", data, self.mime_type or sniff(data) or "application/octet-stream")

    # ------------------------------------------------------------ keeping

    def save(self, storage: Any = None, name: str | None = None) -> str:
        """Stream it into ``storage`` (a Django storage; default ``MEDIA_STORAGE``) as ``name``
        (default a fresh name under ``llm_kit/media/``). Returns the name the storage gave it."""
        from django.core.files import File

        storage = storage if storage is not None else default_storage()
        name = name or _fresh_name(self.extension)
        if self.file is not None and self.data is None:
            with contextlib.suppress(Exception):
                self.file.seek(0)
            saved = storage.save(name, self.file if hasattr(self.file, "chunks") else File(self.file))
        elif self.data is not None:
            from django.core.files.base import ContentFile

            saved = storage.save(name, ContentFile(self.data))
        else:
            with tempfile.SpooledTemporaryFile(max_size=SPOOL) as spool:
                limit, total = _max_bytes(), 0
                for piece in self.chunks():
                    total += len(piece)
                    if limit and total > limit:
                        raise MediaTooLarge(f"Media is larger than MEDIA_MAX_BYTES ({limit} bytes).")
                    spool.write(piece)
                spool.seek(0)
                saved = storage.save(name, File(spool, name=name))
        self.meta["saved_as"] = saved
        return saved

    async def asave(self, storage: Any = None, name: str | None = None) -> str:
        from asgiref.sync import sync_to_async

        return await sync_to_async(self.save, thread_sensitive=False)(storage, name)

    def to_json(self) -> dict[str, Any]:
        """What can be stored or sent: never the bytes."""
        out: dict[str, Any] = {"mime_type": self.mime_type}
        if self.name:
            out["name"] = self.name
        if self.url:
            out["url"] = self.url
        if self.data is not None:
            out["bytes"] = len(self.data)
        if self.meta:
            out["meta"] = {k: v for k, v in self.meta.items() if _jsonable(v)}
        return out


def audio_seconds(media: Media) -> float | None:
    """How long a WAV clip is (None for other formats: the provider says, or it is unknown)."""
    if media.data is None or (media.mime_type or sniff(media.data)) not in ("audio/wav", "audio/x-wav"):
        return None
    import io
    import wave

    try:
        with wave.open(io.BytesIO(media.data)) as clip:
            rate = clip.getframerate()
            return round(clip.getnframes() / rate, 3) if rate else None
    except (wave.Error, EOFError):
        return None


def _jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict))


def _guess(name: str) -> str:
    return mimetypes.guess_type(name)[0] or ""


def _fresh_name(extension: str) -> str:
    import uuid

    prefix = str(config.get("MEDIA_PATH") or "llm_kit/media/").rstrip("/")
    return f"{prefix}/{uuid.uuid4().hex}{extension}"


def default_storage() -> Any:
    """The storage ``MEDIA_STORAGE`` names (a ``STORAGES`` alias; default ``"default"``)."""
    from django.core.files.storage import storages

    return storages[config.get("MEDIA_STORAGE") or "default"]


def _client():
    """An HTTP client that connects only to the addresses ``public_address`` allows, checked at
    connect time for every request and redirect (no DNS rebinding between check and use)."""
    import httpcore
    import httpx

    class PublicOnly(httpcore.NetworkBackend):
        def __init__(self, inner):
            self.inner = inner

        def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
            address = public_address(host, port)  # the TLS name and Host header stay the host's
            return self.inner.connect_tcp(
                address, port, timeout=timeout, local_address=local_address, socket_options=socket_options
            )

        def connect_unix_socket(self, path, timeout=None, socket_options=None):
            raise UnsafeURL("Unix sockets can't be fetched.")

        def sleep(self, seconds):
            self.inner.sleep(seconds)

    transport = httpx.HTTPTransport()
    transport._pool._network_backend = PublicOnly(transport._pool._network_backend)
    timeout = float(config.get("MEDIA_FETCH_TIMEOUT") or 120)
    return httpx.Client(transport=transport, timeout=timeout, trust_env=False)


def _download(url: str, size: int) -> Iterator[bytes]:
    with _client() as client:
        for _ in range(6):  # each redirect is checked like the first URL
            _check_url(url)
            with client.stream("GET", url) as r:
                if r.is_redirect and r.headers.get("location"):
                    url = str(r.url.join(r.headers["location"]))
                    continue
                r.raise_for_status()
                yield from r.iter_bytes(size)
                return
    raise UnsafeURL("Too many redirects.")


@dataclass
class MediaResult:
    """What a media call returns.

    ``media``: what it made (images, a clip, audio), in order. ``text``: what it read
    (a transcript, a document's text). ``data``: the rest the provider said (``segments``,
    ``words``, ``language``, ``duration``, ``revised_prompt``, ``pages``...). ``units`` is what
    it was billed by, ``cost`` its price in USD when known, ``model`` the model that answered."""

    kind: str
    media: list[Media] = field(default_factory=list)
    text: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    units: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    served_model: str = ""
    cost: Decimal | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    job_id: str = ""  # the provider's job, for a long kind

    @property
    def first(self) -> Media | None:
        return self.media[0] if self.media else None

    def __getattr__(self, name: str) -> Any:  # result.segments, result.language, ...
        data = self.__dict__.get("data") or {}
        if name in data:
            return data[name]
        raise AttributeError(name)

    def save(self, storage: Any = None, name: str | None = None) -> list[str]:
        """Save each piece of media (``name`` is a prefix for several). Returns the saved names."""
        if len(self.media) == 1:
            return [self.media[0].save(storage, name)]
        return [
            m.save(storage, f"{name}-{i}{m.extension}" if name else None) for i, m in enumerate(self.media, start=1)
        ]

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "media": [m.to_json() for m in self.media],
            "text": self.text,
            "data": {k: v for k, v in self.data.items() if _jsonable(v)},
            "units": self.units,
            "model": self.model,
            "cost": str(self.cost) if self.cost is not None else None,
        }
