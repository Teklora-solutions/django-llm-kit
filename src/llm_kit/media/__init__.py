"""Generated media: images, video, speech, transcription, OCR, and kinds you add.

Make media through a touchpoint (``tp.image()``, ``tp.video()``, ``tp.speak()``,
``tp.transcribe()``, ``tp.ocr()``, ``tp.generate(kind, ...)``); this module has the types,
the kind registry and the errors. See the media guide.
"""

from llm_kit.core.media import (
    DONE,
    FAILED,
    PENDING,
    RUNNING,
    ContentRefused,
    JobState,
    LiteLLMMedia,
    Media,
    MediaBackend,
    MediaJobFailed,
    MediaJobTimeout,
    MediaKind,
    MediaResult,
    MediaTooLarge,
    NoMediaReturned,
    UnsafeURL,
    all_kinds,
    get_kind,
    is_refusal,
    register_kind,
)

__all__ = [
    "DONE",
    "FAILED",
    "PENDING",
    "RUNNING",
    "ContentRefused",
    "JobState",
    "LiteLLMMedia",
    "Media",
    "MediaBackend",
    "MediaJobFailed",
    "MediaJobTimeout",
    "MediaKind",
    "MediaResult",
    "MediaTooLarge",
    "NoMediaReturned",
    "UnsafeURL",
    "all_kinds",
    "get_kind",
    "is_refusal",
    "register_kind",
]
