---
description: Let tenants and users save their own model choice, parameters and settings, within limits you set in code, and show them in your UI through an API or Django templates.
---

# Settings your users manage

Some choices belong to your customers, not to your code. A workspace might pick the model its assistant uses, and a user might turn its temperature down. You decide which settings exist, who may change them, and what values are allowed. llm-kit stores the values, applies them on every call, and describes them to your UI, through a [JSON API](#the-settings-api) or [Django templates](#django-templates).

## Scopes

Each saved value belongs to a **scope**. From broadest to narrowest:

| Scope | Whose | Its id comes from |
|---|---|---|
| `platform` | yours: every tenant and user | |
| `tenant` | a workspace and its members | `TENANT_RESOLVER` ([access control](access.md#users-in-several-tenants)) |
| `user` | one signed-in user | the request's user |

A narrower scope's value wins: a user's choice beats their tenant's, which beats the platform's, which beats your code. A client's explicit choice for one request (`{"model": ...}`) still comes first.

## Model choice and parameters

Declare what may be saved for a touchpoint with `settable`:

```python
from llm_kit import ClientChoice, Param, Settable, touchpoint

support = touchpoint(
    "support.chat",
    default="anthropic/claude-sonnet-5-5",
    choice=ClientChoice(allow=["anthropic/*", "openai/*"], max_price_output=20),
    settable=Settable(
        model=("tenant", "user"),                        # who may save a model choice
        params={"temperature": Param("number", min=0, max=1, scopes=("user",))},
    ),
)
```

- A saved model is checked like a client's pick: against the touchpoint's `ClientChoice` (allow patterns, `requires`, price caps), an admin's `allow`, and whether there's a key for its provider. If a saved model isn't allowed any more (you narrowed the choice, or the model was hidden), it is ignored and the call uses the default; the resolution's `notes` say why. For a touchpoint that uses `ServerChoice`, `Settable(model_allow=[...])` bounds saved choices instead.
- Only the params you declare can be saved, each checked against its `Param` (`number`, `integer`, `boolean`, `choice`, `text`; `min`, `max`, `choices`).
- `Resolution.chosen_by` is `"tenant"` or `"user"` when a saved choice was used, and every call records it.

Saved values apply wherever the caller is known: llm-kit's views and WebSocket consumer, `agents.run(user=...)`, and touchpoint calls made inside a running agent. Code that calls a touchpoint with no caller gets your code's settings.

## Your own settings

Register settings that aren't tied to a touchpoint, and read them anywhere:

```python
from llm_kit import scoped
from llm_kit.scoped import Setting

scoped.register(Setting("support.tone", type="choice", choices=["friendly", "formal"], default="friendly",
                        label="Tone", help="How the assistant writes.", scopes=("tenant", "user")))

tone = scoped.value("support.tone", caller)   # the user's, else the tenant's, else the platform's, else the default
```

`choices` can be a function `fn(owner) -> [...]` when the options depend on who is asking. Register settings in your app's `ready()` or in the module that declares your touchpoints.

## Bring your own key

Tenants and users can store their own provider keys, so their calls bill their provider account rather than yours. Turn it on for the scopes you want:

```python
LLM_KIT = {
    "BYOK_SCOPES": ("tenant",),            # who may store keys ("tenant", "user")
    "KEY_POLICY": "own_then_platform",     # the default for every touchpoint
}
```

Each touchpoint uses one **key policy**: `Settable(keys=...)`, otherwise `KEY_POLICY`:

| `KeyPolicy` | Whose keys a call uses |
|---|---|
| `PLATFORM_ONLY` (default) | yours only |
| `OWN_ONLY` | only the keys the caller's user or tenant stored. Fallbacks to providers they have no key for are skipped. With no key at all the call raises `NoKey` before contacting anyone, and your keys are never used for them |
| `OWN_THEN_PLATFORM` | the user's keys, then the tenant's, then yours. Each one is a separate attempt, so yours are used when they have no key for that provider, or when theirs fails |
| `TENANT_CHOOSES` | the tenant picks `own_only` or `own_then_platform` with the `keys.policy` setting (default `own_then_platform`) |

```python
from llm_kit import KeyPolicy, Settable, touchpoint

touchpoint("support.chat", default="anthropic/claude-sonnet-5-5", settable=Settable(keys=KeyPolicy.OWN_ONLY))
```

Every call records whose key it used (`LLMCall.key_owner`: `platform`, `tenant` or `user`), for billing and for the monitoring filters. Calls without a caller, and callers whose tenant brought nothing, use your keys unless the policy is `OWN_ONLY`. Local models (Ollama and other keyless providers) need no key under any policy.

Stored keys are encrypted (`ENCRYPTION_KEYS`), like your own `ProviderKey` rows, and are never returned: the API shows a fingerprint (`…3456 1a2b3c4d`).

```
GET    llm/api/keys/?scope=tenant     {"scope", "editable", "providers": ["anthropic", "openai"], "keys": [{"id", "provider", "label", "fingerprint", "active", "created_at", "works", "checked_at", "check_error"}]}
POST   llm/api/keys/                  {"scope": "tenant", "provider": "anthropic", "key": "sk-...", "label": "production"}
DELETE llm/api/keys/<id>/?scope=tenant
```

Each key can be tested with one tiny call: a few tokens on the cheapest model your touchpoints use for that provider, sent with that key alone. The result is stored on the key (`works`, `checked_at`, `check_error`). A rate limit or quota error still counts as working, because the provider accepted the key.

```
POST   llm/api/keys/<id>/test/?scope=tenant
POST   llm/api/keys/   {..., "verify": true}     # test first; a refused key isn't saved (400)
```

The keys template has a **Test** button per key and a "Check it works first" box, on by default. Your admins can test any keys with the **Test the selected keys** action in the Django admin.

`providers` lists the providers your touchpoints call, to fill a dropdown. Anyone in the scope can list its keys. Adding and removing keys follows `SCOPE_PERMISSIONS`, as for settings. The model options in `llm/api/settings/` count the keys the asker brought.

## Budgets

Tenants and users can have their own spending limits, and you decide who sets them: your admins (a plan's limit), a tenant's managers (a cap for each member), or users themselves.

```python
LLM_KIT = {
    "BUDGET_PERIODS": ["day", "month"],          # or {"quarter": "myapp.billing:quarter_start"}
    "BUDGET_EDITORS": {                          # kind of budget -> scopes that may set it
        "tenant": ["platform"],                  # a workspace's total, set by you
        "member": ["platform", "tenant"],        # each member of a workspace, set by you or its managers
        "user": ["user"],                        # a user's own cap
    },
}
```

| Budget scope | Limits |
|---|---|
| `tenant` | everything a workspace's members spend together |
| `member` | each member of a workspace, separately |
| `user` | one user, across workspaces |

A budget has a period, an amount in USD, an optional touchpoint (blank means all of them), and `counts`. Set `counts` to `"platform_keys"` to leave out calls made with keys the tenant or user [brought](#bring-your-own-key), so a plan limit covers only what you pay for. Periods are `day`, `week` (from Monday) and `month` in your time zone, or your own as `{"name": "dotted.path:fn(now) -> start"}`. Only the periods listed in `BUDGET_PERIODS` can be used. No one can set budgets until `BUDGET_EDITORS` says who.

Before every call (`model()`, `complete()`, `embed()`), each budget that applies to the caller is checked against a running total, without a database query per call (see [budgets](agents.md#budgets) for how the total is kept). Once a budget is used up, the call raises `BudgetExceeded`, for example "The workspace budget of $50 is used up ($50.12 spent this month).", and an agent stops with `stopped="budget"`. Your admins can set any tenant's or user's budget under **Scoped budgets** in the Django admin.

```
GET    llm/api/budgets/?scope=tenant
PUT    llm/api/budgets/   {"scope": "member", "period": "day", "amount": "2", "touchpoint": "", "counts": "all"}
DELETE llm/api/budgets/   {"scope": "member", "period": "day"}
```

`GET` returns what a UI needs to show "$3.10 of $10 this month":

```json
{"scope": "tenant", "periods": ["day", "month"], "editable": false,
 "budgets": [{"scope": "tenant", "touchpoint": "", "period": "month", "amount": "10", "counts": "all",
              "spent": "3.1000", "remaining": "6.9000", "resets_at": "2026-12-01T00:00:00+03:00"}]}
```

The budget always belongs to the requester's own tenant or user. `editable` says whether they may change it.

## Who may change what

`LLM_KIT["SCOPE_PERMISSIONS"]` maps a scope to `fn(user, tenant) -> bool`:

```python
LLM_KIT = {"SCOPE_PERMISSIONS": {"tenant": "myapp.permissions:manages_workspace"}}
```

Without a function, the defaults are:

- `platform`: users with the `llm_kit.change_scopedsetting` permission (superusers have it).
- `tenant`: nobody. Say who manages a workspace before tenants can save anything.
- `user`: the user themselves.

A setting can also limit its own scopes (`scopes=("user",)`), so even a tenant manager can't save it for the whole tenant.

## The settings API

Include `llm_kit.transport.api.urls` (see [streaming and APIs](streaming.md)), then:

```
GET    llm/api/settings/?scope=user[&touchpoint=support.chat]
PATCH  llm/api/settings/   {"scope": "tenant", "touchpoint": "support.chat", "key": "model", "value": "openai/gpt-5"}
DELETE llm/api/settings/   {"scope": "tenant", "touchpoint": "support.chat", "key": "model"}
```

`GET` returns everything the scope may save, grouped into sections your UI can render as they come:

```json
{
  "scope": "user",
  "editable": true,
  "sections": [
    {"touchpoint": "", "title": "General", "settings": [
      {"key": "support.tone", "type": "choice", "label": "Tone", "value": null,
       "inherited": {"value": "formal", "scope": "tenant"}, "default": "friendly",
       "options": ["friendly", "formal"], "editable": true}
    ]},
    {"touchpoint": "support.chat", "title": "support.chat", "description": "...", "settings": [
      {"key": "model", "type": "model", "value": "openai/gpt-5", "inherited": null,
       "options": [{"value": "anthropic/claude-sonnet-5-5", "label": "Claude Sonnet 5.5",
                    "price_input": "3", "price_output": "15", "context_window": 1000000}]},
      {"key": "param.temperature", "type": "number", "min": 0, "max": 1, "value": 0.3}
    ]}
  ]
}
```

`value` is what this scope saved (null: nothing), and `inherited` is what it gets otherwise. `PATCH` and `DELETE` return the setting's new description. Errors: 400 for a value that isn't allowed, 403 when the requester may not change that scope, and 404 for an unknown setting or a scope the requester doesn't have (no tenant, say). The scope's id always comes from the request, never from the body.

Admins see and edit every saved value under **Scoped settings** in the Django admin.

## Django templates

The same sections as server-rendered forms, for pages you build with Django templates. They post to llm-kit's views (included with `llm_kit.transport.sse.urls`), which save through the same checks as the API and redirect back with a message:

```django
{% load llm_kit %}

<h2>Workspace AI settings</h2>
{% llm_kit_settings scope="tenant" %}
{% llm_kit_keys scope="tenant" %}
{% llm_kit_budgets scope="member" %}

<h2>Mine</h2>
{% llm_kit_settings scope="user" touchpoint="support.chat" %}
```

- The tags need `request` in the template context (`django.template.context_processors.request`) and use the CSRF token. Messages appear if `django.contrib.messages` is installed.
- Someone who may not change a scope sees its values with the inputs disabled. Someone who doesn't have the scope (no tenant, say) sees a short note.
- The HTML is plain and has no classes beyond `llm-kit-*` hooks, so your CSS styles it. To change the markup, override any template in your project: `llm_kit/settings/sections.html`, `section.html`, `setting.html`, `fields/<type>.html` (`text`, `number`, `integer`, `boolean`, `choice`, `model`), `keys.html` and `budgets.html`.

The example project's `/settings/` page (`example/chat/templates/chat/settings.html`) uses all three tags, with a stylesheet of about 30 lines.
