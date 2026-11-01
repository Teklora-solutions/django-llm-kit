"""Agent tools from Django models, plus hand-written tools, in one catalogue filtered per caller.

Declare on a model (or with ``expose(Model, ...)`` for models you can't edit)::

    class Invoice(models.Model):
        class AgentAccess:
            description = "invoices we sent"
            fields = ["number", "customer", "total", "status"]
            search_fields = ["number", "customer__name"]
            filter_fields = ["status", "issued_on"]
            tenant_field = "organisation"           # or owner_field / agent_queryset / unscoped = "why"
            writes = "direct"                       # "none" | "direct" | "approval"
            writable_fields = ["total", "customer"]
            approval_fields = ["status"]

and use them::

    llm.bind_tools(llm_kit.tools.for_user(request.user))    # what this user may use, bound to them
    ToolNode(llm_kit.tools.all_tools())                     # graph built once; caller read per call
    llm.bind_tools(llm_kit.tools.for_config(config))        # inside a node, for the run's caller

The caller (user + tenant) is always set by the server, never by the model.
"""

from llm_kit.tools.caller import AUTHENTICATED, PUBLIC, Caller, caller_for, caller_from
from llm_kit.tools.registry import (
    ToolEntry,
    ToolSelection,
    all_tools,
    catalogue,
    entries_for,
    expose,
    for_config,
    for_user,
    guidance,
    register,
    select,
)

__all__ = [
    "AUTHENTICATED",
    "PUBLIC",
    "Caller",
    "ToolEntry",
    "ToolSelection",
    "all_tools",
    "approve",
    "caller_for",
    "caller_from",
    "catalogue",
    "entries_for",
    "expose",
    "for_config",
    "for_user",
    "guidance",
    "pending_for",
    "proposal_created",
    "proposal_decided",
    "register",
    "reject",
    "select",
]

_PROPOSALS = {"approve", "reject", "pending_for", "proposal_created", "proposal_decided", "ProposalError"}


def __getattr__(name):
    # Proposals import llm_kit.models; keep `import llm_kit.tools` usable before apps are ready.
    if name in _PROPOSALS:
        from llm_kit.tools import proposals

        return getattr(proposals, name)
    raise AttributeError(name)
