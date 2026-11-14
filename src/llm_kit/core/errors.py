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


class AllModelsFailed(LLMKitError):
    """Every model a touchpoint tried failed. ``errors`` holds ``(model, exception)`` for each,
    in the order tried; ``__cause__`` is the last one.

    Its message lists every failure, and :func:`~llm_kit.core.observability.classify.classify_error`
    classifies it by the primary model's error."""

    def __init__(self, touchpoint: str, errors: list[tuple[str, BaseException]]) -> None:
        self.touchpoint = touchpoint
        self.errors = list(errors)
        lines = "; ".join(f"{model}: {type(e).__name__}: {e}" for model, e in self.errors)
        super().__init__(f"All {len(self.errors)} models for {touchpoint!r} failed. {lines}")

    @property
    def last(self) -> BaseException:
        return self.errors[-1][1]


class BudgetExceeded(LLMKitError):
    """A user's or tenant's spend for today is over ``LLM_KIT["DAILY_BUDGETS"]``: the request
    is refused before it starts. (A request over its own ``AGENT_BUDGET`` isn't an error: its
    agents stop with ``stopped="budget"``.)"""

    def __init__(self, scope: str, spent, budget) -> None:
        self.scope, self.spent, self.budget = scope, spent, budget
        super().__init__(f"Today's {scope} budget of ${budget} is used up (${spent} spent).")


class NoKey(ConfigError):
    """No provider key a call may use: under a bring-your-own-key policy, the caller's tenant
    or user has none for the provider (your keys are not used for them)."""
