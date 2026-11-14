"""One error classification for every provider and client library.

Several audited projects got this wrong in the same way: treating an exhausted
quota as a temporary rate limit, or parsing 429 text by hand.
"""

from __future__ import annotations

ERROR_CLASSES = (
    "rate_limit",  # temporary; retry later
    "quota_exhausted",  # won't clear by retrying soon (billing / daily quota)
    "timeout",
    "auth",
    "overloaded",  # provider-side 5xx / 529
    "content_filter",
    "context_length",
    "bad_request",
    "connection",
    "cancelled",
    "other",
)

RETRYABLE = frozenset({"rate_limit", "timeout", "overloaded", "connection"})

# Resets with the calendar: waiting won't help today. Checked first, because Gemini reports
# daily and per-minute limits with the same 429 and the same "exceeded your current quota".
_DAILY_HINTS = ("per day", "perday", "daily limit", "requests per day")
# A window that closes on the clock (Gemini's quotaId, or a stated retry delay).
_WINDOW_HINTS = ("per minute", "perminute", "per second", "persecond", "retrydelay", "retry_delay", "retry in ")
_QUOTA_HINTS = (
    "insufficient_quota",
    "quota exceeded",
    "exceeded your current quota",
    "resource has been exhausted",
    "billing",
    "credit balance",
    "out of credits",
)
_CONTEXT_HINTS = ("context length", "context_length", "maximum context", "too many tokens", "prompt is too long")
_FILTER_HINTS = ("content filter", "content_filter", "safety", "blocked", "responsible ai")


def _status(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def classify_error(exc: BaseException) -> str:
    import asyncio

    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
        return "cancelled"
    errors = getattr(exc, "errors", None)
    if type(exc).__name__ == "AllModelsFailed" and errors:
        return classify_error(errors[0][1])
    name = type(exc).__name__.lower()
    text = f"{name} {exc}".lower()
    status = _status(exc)

    if any(h in text for h in _DAILY_HINTS):
        return "quota_exhausted"
    if any(h in text for h in _WINDOW_HINTS) and (status == 429 or "429" in text or "resource_exhausted" in text):
        return "rate_limit"
    if any(h in text for h in _QUOTA_HINTS):
        return "quota_exhausted"
    if (
        status == 429
        or "ratelimit" in name
        or "rate limit" in text
        or "rate_limit" in text
        or "too many requests" in text
    ):
        return "rate_limit"
    if isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in text or status in (408, 504):
        return "timeout"
    if status in (401, 403) or "authentication" in name or "permissiondenied" in name or "api key" in text:
        return "auth"
    if any(h in text for h in _CONTEXT_HINTS) or "contextwindow" in name:
        return "context_length"
    if "contentpolicy" in name or any(h in text for h in _FILTER_HINTS):
        return "content_filter"
    if (
        status in (500, 502, 503, 529)
        or "overloaded" in text
        or "serviceunavailable" in name
        or "internalserver" in name
    ):
        return "overloaded"
    if isinstance(exc, ConnectionError) or "connection" in name or "apiconnection" in name:
        return "connection"
    if status in (400, 404, 422) or "badrequest" in name or "invalidrequest" in name or "notfound" in name:
        return "bad_request"
    return "other"


def is_retryable(exc: BaseException) -> bool:
    return classify_error(exc) in RETRYABLE
