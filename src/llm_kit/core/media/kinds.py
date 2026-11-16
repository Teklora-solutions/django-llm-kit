"""The kinds of media a touchpoint can make or read, and how each is measured.

Built in: ``image``, ``image_edit``, ``video``, ``speech``, ``transcription``, ``ocr``. Add your
own (music, 3D, anything a provider sells) with ``register_kind``; call it with
``tp.generate("music", prompt=...)`` and give the touchpoint a backend that knows it::

    from llm_kit.media import MediaKind, register_kind

    def seconds(request, result):
        return {"output_seconds": request.get("seconds")}

    register_kind(MediaKind("music", long=True, measure=seconds))
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llm_kit.core.errors import ConfigError

Measure = Callable[[dict[str, Any], Any], dict[str, Any]]


@dataclass(frozen=True)
class MediaKind:
    """``name``: what callers ask for. ``catalogue_kind``: the catalogue's ``kind`` for models that
    do it (default ``name``). ``long``: it runs as a job at the provider (submit, then check until
    done): ``tp.start(...)`` runs it in the background. ``measure(request, result)``: the units it
    is billed by (``llm_kit.core.observability.cost.UNITS``); what the backend reports wins."""

    name: str
    catalogue_kind: str = ""
    long: bool = False
    measure: Measure | None = None
    description: str = ""

    @property
    def catalogue(self) -> str:
        return self.catalogue_kind or self.name


_kinds: dict[str, MediaKind] = {}
_lock = threading.Lock()


def register_kind(kind: MediaKind) -> MediaKind:
    """Add (or replace) a kind of media."""
    with _lock:
        _kinds[kind.name] = kind
    return kind


def get_kind(name: str | MediaKind) -> MediaKind:
    if isinstance(name, MediaKind):
        return name
    try:
        return _kinds[name]
    except KeyError:
        raise ConfigError(f"Unknown media kind {name!r}. Known: {', '.join(sorted(_kinds))}.") from None


def all_kinds() -> list[MediaKind]:
    return sorted(_kinds.values(), key=lambda k: k.name)


# ------------------------------------------------------------------ built in


def _size_of(request: dict, result: Any) -> str | None:
    return (result.data.get("size") if result is not None else None) or request.get("size")


def _images(request: dict, result: Any) -> dict:
    return {"images": len(result.media), "size": _size_of(request, result), "quality": request.get("quality")}


def _image_edit(request: dict, result: Any) -> dict:
    return {**_images(request, result), "input_images": len(request.get("images") or ())}


def _video(request: dict, result: Any) -> dict:
    seconds = result.data.get("seconds") or request.get("seconds")
    try:
        seconds = float(seconds) if seconds is not None else None
    except (TypeError, ValueError):
        seconds = None
    return {
        "videos": len(result.media) or 1,
        "output_seconds": seconds,
        "resolution": result.data.get("resolution")
        or request.get("resolution")
        or _resolution(_size_of(request, result)),
        "size": _size_of(request, result),
    }


def _resolution(size: Any) -> str | None:
    """``"1280x720"`` -> ``"720p"``."""
    try:
        w, h = (int(x) for x in str(size).lower().split("x"))
    except (TypeError, ValueError):
        return None
    short = min(w, h)
    return "4k" if short >= 2160 else "2k" if short >= 1440 else f"{short}p"


def _speech(request: dict, result: Any) -> dict:
    return {"characters": len(request.get("text") or ""), "output_seconds": result.data.get("duration")}


def _transcription(request: dict, result: Any) -> dict:
    return {"input_seconds": result.data.get("duration") or request.get("duration")}


def _ocr(request: dict, result: Any) -> dict:
    pages = result.data.get("pages")
    return {"pages": len(pages) if isinstance(pages, list) else pages}


for _kind in (
    MediaKind("image", measure=_images, description="Images from a prompt."),
    MediaKind("image_edit", measure=_image_edit, description="Changes to images it is given."),
    MediaKind("video", long=True, measure=_video, description="Video from a prompt (and an image)."),
    MediaKind("speech", measure=_speech, description="Text read aloud."),
    MediaKind("transcription", measure=_transcription, description="Speech written down."),
    MediaKind("ocr", measure=_ocr, description="The text of a document or image."),
):
    register_kind(_kind)


def units_for(kind: MediaKind, request: dict, result: Any) -> dict[str, Any]:
    """The units ``result`` is billed by: the kind's measure, then what the backend reported."""
    units: dict[str, Any] = {}
    if kind.measure is not None:
        try:
            units.update(kind.measure(request, result) or {})
        except Exception:  # measuring must never fail a call that worked
            import logging

            logging.getLogger("llm_kit.touchpoints").warning("Could not measure a %s call", kind.name, exc_info=True)
    units.update(result.units or {})
    return {k: v for k, v in units.items() if v not in (None, "", 0)}
