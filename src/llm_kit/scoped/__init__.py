"""Settings your tenants and users save: a model choice, parameters, your own settings.

You declare what may be saved and by whom; llm-kit stores it, applies it when a model is
resolved, and describes it to your UI (``GET llm/api/settings/``, ``{% llm_kit_settings %}``)::

    from llm_kit import touchpoint, ClientChoice
    from llm_kit.scoped import Param, Settable, Setting, register

    touchpoint(
        "support.chat",
        default="anthropic/claude-sonnet-5-5",
        choice=ClientChoice(allow=["anthropic/*", "openai/*"]),   # bounds saved choices
        settable=Settable(model=("tenant", "user"), params={"temperature": Param("number", 0, 1)}),
    )
    register(Setting("myapp.tone", type="choice", choices=["friendly", "formal"]))

    tone = llm_kit.scoped.value("myapp.tone", caller)   # user's, else tenant's, else platform's

Who may change a scope's values: ``LLM_KIT["SCOPE_PERMISSIONS"]`` (see ``service.can_edit``).
"""

from llm_kit.core.scoped import SCOPES, Param, Settable, Setting
from llm_kit.scoped.registry import definition, definitions, register, unregister
from llm_kit.scoped.service import NotPermitted, UnknownSetting, can_edit, clear, save, scope_ids, value

__all__ = [
    "SCOPES",
    "NotPermitted",
    "Param",
    "Settable",
    "Setting",
    "UnknownSetting",
    "can_edit",
    "clear",
    "definition",
    "definitions",
    "register",
    "save",
    "scope_ids",
    "unregister",
    "value",
]
