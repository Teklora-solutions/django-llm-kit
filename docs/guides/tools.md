---
description: Generate agent tools from Django models with AgentAccess; every query and write is limited to the caller's rows, with Django permissions and approval proposals.
---

# Agent tools from your models

Declare what agents may see on a model, and llm-kit generates the tools:

- `find_<plural>`: search, filter, page, or look up by id.
- `save_<name>`: create, or update by id. Only if the model allows writes.

There are no delete tools.

## Declaring access

```python
class Invoice(models.Model):
    ...
    class AgentAccess:
        description = "invoices we sent to customers"
        fields = ["number", "customer", "total", "status", "issued_on"]   # nothing else is shown
        search_fields = ["number", "customer__name"]
        filter_fields = ["status", "issued_on", "customer"]              # dates get _from / _to
        tenant_field = "organisation"     # whose rows: owner_field, tenant_field, agent_queryset, or unscoped = "why"
        writes = "direct"                 # "none" | "direct" | "approval"
        writable_fields = ["total", "customer", "issued_on"]
        approval_fields = ["status"]      # proposed by the agent, applied when a person approves
```

Every model must say whose rows a caller sees: `owner_field`, `tenant_field`, an `agent_queryset(queryset, caller)` function that returns a narrowed queryset, or `unscoped = "reason"` to make the choice explicit.

For models you can't edit (third-party apps, `User`), put the same options in `<app>/agent_tools.py`, imported at startup:

```python
from llm_kit import tools

tools.expose(User, fields=["username", "first_name"], owner_field="pk", name="account", plural="accounts")
tools.register(my_langchain_tool, permission="crm.view_lead", category="crm")   # hand-written tools join the catalogue
tools.register(site_search, permission=tools.PUBLIC, category="site")            # anyone: visitors, and runs with no user
```

## Using the tools

```python
from llm_kit import tools

llm.bind_tools(tools.for_user(request.user))       # what this user may use, bound to them
ToolNode(tools.all_tools())                        # in a graph built once: each call reads the run's caller
llm.bind_tools(tools.for_config(config))           # inside a node: the caller the server put in config
system_prompt += tools.guidance(bound_tools)       # one usage line per tool
```

[Agents](agents.md) take `tools=tools.select(...)` and do this for you.

## The caller is set by the server

Runs started through llm-kit's views carry `Caller(user, tenant)` in the run config. The tenant comes from `TENANT_RESOLVER(user)` (which may also take the [request](access.md#users-in-several-tenants)). Clients can't set it (every client-sent `configurable` key starting with `__` is dropped), it is never written to checkpoints, and a tool with no caller refuses to run.

## Scoping

- Every query and write is limited to the caller's rows.
- On create, the owner and tenant are filled in by the server.
- A writable link (foreign key) only accepts rows the caller can see.
- A write that would move a row out of the caller's scope is rolled back.

## Permissions

- Django's view/add/change permissions by default; set `permission` / `write_permission` to override.
- Checked when tools are listed or bound, and again on every call.
- Anonymous users get no tools, except tools registered with `permission=tools.PUBLIC`.

## Approvals

- `writes = "approval"`, or a change to any of `approval_fields`, creates a `ToolProposal` instead of writing.
- `tools.pending_for(user)`, `tools.approve(p, by=user)` and `tools.reject(p, by=user, reason=...)`, plus admin actions and the JSON API.
- Approving refuses if the record changed since it was proposed.
- Signals: `proposal_created`, `proposal_decided`.

## Refused at startup

- Fields that look like secrets (password, token, api_key…), binary or vector columns.
- Writable links to users or file fields.
- Fields like `status`/`is_active` as direct writes (use `approval_fields`).
- Creates that would miss a required field.
- A model with no declared scope.
- Duplicate tool names.

`paused = "why"` keeps a declaration checked but hidden.

## Inspecting tools

- `GET /llm/api/tools/` lists the current user's tools with their JSON schemas.
- `GET /llm/api/tools/proposals/`, and `POST .../<id>/approve/` or `.../<id>/reject/`.
- `python manage.py llm_kit_tools [--user name]`.
- Tool calls are recorded like everything else (*Tool calls* in the admin).
