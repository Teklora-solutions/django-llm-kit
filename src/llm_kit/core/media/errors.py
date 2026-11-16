"""Errors of media calls, and how a provider's refusal is recognised."""

from __future__ import annotations

from llm_kit.core.errors import LLMKitError


class ContentRefused(LLMKitError):
    """The provider refused the request under its content policy.

    By default the touchpoint doesn't try its next model: quietly sending a refused prompt to a
    more permissive provider is a decision for the app, not the kit (``fallback_on_refusal=True``
    on the touchpoint allows it). Raise it from your own backend to say the same."""


class NoMediaReturned(LLMKitError):
    """The provider answered with nothing usable (no media, no text)."""


class MediaJobFailed(LLMKitError):
    """A long job (a video, say) ended without a result at the provider."""

    def __init__(self, message: str, job_id: str = "") -> None:
        self.job_id = job_id
        super().__init__(message)


class MediaJobTimeout(LLMKitError):
    """A long job didn't finish within the time it was given. It may still finish, and be billed,
    at the provider: ``job_id`` is its id there. It is never sent to the next model."""

    def __init__(self, message: str, job_id: str = "", model: str = "") -> None:
        self.job_id, self.model = job_id, model
        super().__init__(message)


_REFUSAL_CLASSES = ("ContentPolicyViolationError", "ContentRefused", "ContentFilterFinishReasonError")
_REFUSAL_WORDS = (
    "content_policy_violation",
    "content policy",
    "content_policy",
    "moderation_blocked",
    "safety system",
    "safety_violation",
    "responsible ai",
    "responsibleai",
    "blocked by safety",
    "prompt was blocked",
    "nsfw",
)


def is_refusal(error: BaseException) -> bool:
    """Whether ``error`` is a provider refusing the content (not an outage or a bad request)."""
    for cls in type(error).__mro__:
        if cls.__name__ in _REFUSAL_CLASSES:
            return True
    text = str(error).lower()
    return any(word in text for word in _REFUSAL_WORDS)
