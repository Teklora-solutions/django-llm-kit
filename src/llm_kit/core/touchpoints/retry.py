"""Waiting out a refusal that clears on its own, before falling back to the next model.

A provider that answers "retry in 4s" will answer; moving straight to a fallback (or
failing, when there is none) throws that away. A daily quota or a bad key won't clear
by waiting, so those go to the fallback at once.

How long to wait depends on who is waiting, so it is a policy::

    touchpoint("crm.digest", default=..., retry=RetryPolicy.BATCH)   # a worker can wait a minute
    chat.model(config=config, retry=RetryPolicy.NEVER)              # this call never waits

Order of precedence: ``model(retry=)`` / ``complete(retry=)``, then the run's
``agents.run(..., retry=)``, then ``touchpoint(retry=)``, then ``LLM_KIT["RETRY"]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar

from llm_kit.core.observability.classify import classify_error

RETRY_KEY = "__llm_kit_retry"  # configurable key set by agents.run(retry=); clients can't send "__" keys

# Used when a provider rate-limits without saying for how long: long enough to clear a
# per-minute window. Overloaded (503/529) errors usually clear faster.
DEFAULT_PAUSE = 20.0
OVERLOADED_PAUSE = 5.0

_DELAY_PATTERNS = (
    re.compile(r"retry[_-]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", re.I),  # Gemini: "retryDelay": "52s"
    re.compile(r"(?:retry|try again) (?:in|after)\s+(\d+(?:\.\d+)?)\s*(ms|s|sec|seconds?)?\b", re.I),
)


@dataclass(frozen=True)
class RetryPolicy:
    """Up to ``attempts`` waits per model, each at most ``max_wait`` seconds."""

    attempts: int = 0
    max_wait: float = 0.0

    BATCH: ClassVar[RetryPolicy]
    INTERACTIVE: ClassVar[RetryPolicy]
    NEVER: ClassVar[RetryPolicy]

    @property
    def waits(self) -> bool:
        return self.attempts > 0 and self.max_wait > 0

    def delay(self, exc: BaseException, attempt: int) -> float | None:
        """Seconds to wait before retrying after ``exc`` (the ``attempt``-th failure, from 0), or None."""
        if not self.waits or attempt >= self.attempts:
            return None
        seconds = retry_after(exc)
        if seconds is None or seconds > self.max_wait:
            return None
        return seconds

    @classmethod
    def coerce(cls, value: Any) -> RetryPolicy | None:
        if value is None or isinstance(value, RetryPolicy):
            return value
        if isinstance(value, dict):
            return cls(attempts=int(value.get("attempts", 0)), max_wait=float(value.get("max_wait", 0)))
        if isinstance(value, str) and value.upper() in ("BATCH", "INTERACTIVE", "NEVER"):
            return getattr(cls, value.upper())
        raise TypeError(f"retry must be a RetryPolicy, a dict or 'batch'/'interactive'/'never', not {value!r}")


# A background job: a minute of waiting is nothing next to the job itself.
RetryPolicy.BATCH = RetryPolicy(attempts=4, max_wait=65.0)
# Someone is watching: one short wait covers a spike; longer and the page looks broken.
RetryPolicy.INTERACTIVE = RetryPolicy(attempts=1, max_wait=6.0)
RetryPolicy.NEVER = RetryPolicy()


def _header_delay(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after-ms")
        if value is not None:
            return float(value) / 1000
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):  # an HTTP date, or an odd headers object
        return None


def stated_delay(exc: BaseException) -> float | None:
    """The delay the provider asked for, if it said: a ``retry_after`` attribute, the
    ``Retry-After`` header, or the error text ("retryDelay": "52s", "try again in 3s")."""
    value = getattr(exc, "retry_after", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    if (value := _header_delay(exc)) is not None:
        return value
    text = str(exc)
    for pattern in _DELAY_PATTERNS:
        if found := pattern.search(text):
            seconds = float(found.group(1))
            unit = found.group(2) if found.lastindex and found.lastindex >= 2 else None
            return seconds / 1000 if unit == "ms" else seconds
    return None


def retry_after(exc: BaseException) -> float | None:
    """Seconds after which ``exc`` is likely to clear, or None when waiting won't help
    (a daily quota, a bad key, a bad request, ...)."""
    kind = classify_error(exc)
    if kind == "rate_limit":
        stated = stated_delay(exc)
        return stated if stated is not None else DEFAULT_PAUSE
    if kind == "overloaded":
        stated = stated_delay(exc)
        return stated if stated is not None else OVERLOADED_PAUSE
    return None


def policy_for(tp, config: Any = None, override: Any = None) -> RetryPolicy:
    """The policy a call uses (see the module docstring for the order)."""
    from llm_kit.core import config as settings

    for value in (override, ((config or {}).get("configurable") or {}).get(RETRY_KEY), tp.retry):
        if value is not None:
            return RetryPolicy.coerce(value)
    return RetryPolicy.coerce(settings.get("RETRY")) or RetryPolicy.NEVER


__all__ = ["RetryPolicy", "retry_after", "stated_delay", "policy_for", "RETRY_KEY"]
