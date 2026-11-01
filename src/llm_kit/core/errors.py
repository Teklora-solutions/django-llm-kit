class LLMKitError(Exception):
    """Base class for llm-kit errors."""


class ConfigError(LLMKitError):
    """llm-kit is misconfigured (bad settings, missing credentials, bad registry entry)."""


class BadRequest(LLMKitError, ValueError):
    """A run payload from a client is invalid."""


class UnknownGraph(LLMKitError, LookupError):
    pass


class UnknownModel(LLMKitError, LookupError):
    pass


class UnknownTouchpoint(LLMKitError, LookupError):
    pass


class TouchpointDisabled(LLMKitError):
    """An admin switched this touchpoint off."""


class ModelNotAllowed(BadRequest):
    """A client asked for a model the touchpoint's policy does not allow."""

    def __init__(self, message: str, reasons: list[str] | None = None) -> None:
        super().__init__(message)
        self.reasons = reasons or []
