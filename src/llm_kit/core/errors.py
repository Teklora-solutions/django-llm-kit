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
