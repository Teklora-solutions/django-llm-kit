"""Are the providers your touchpoints depend on ready, before an outage sends traffic to them?

``readiness()`` looks at every model each touchpoint is configured with (its default and its
fallbacks, after admin overrides) whether or not it has had any calls, and says for each
provider whether credentials are there and where they come from. It never returns a secret:
only counts, sources and the result of the last key test (``POST llm/api/keys/<id>/test/``).

``resolve()`` quietly drops a fallback with no key, so a missing key only shows up when the
primary fails. This is the place to see it first.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("llm_kit")


def _provider(provider: str, env_names: list[str]) -> dict[str, Any]:
    from llm_kit.core.touchpoints.env import KEYLESS_PROVIDERS, env_api_keys
    from llm_kit.credentials.keys import _from_source
    from llm_kit.models import ProviderKey
    from llm_kit.monitoring.breaker import breaker

    sources: dict[str, int] = {}
    problems: list[str] = []
    try:
        if n := len(_from_source(provider)):
            sources["credentials"] = n  # LLM_KIT["CREDENTIALS"]: a vault, a service account, ...
    except Exception as e:
        problems.append(f"CREDENTIALS source failed: {type(e).__name__}")
    rows = list(
        ProviderKey.objects.filter(provider=provider, active=True, scope="").values_list(
            "last_check_ok", "last_check_error", "last_checked_at"
        )
    )
    if rows:
        sources["database"] = len(rows)
    if n := len(env_api_keys(provider, env_names)):
        sources["environment"] = n
    keyless = provider in KEYLESS_PROVIDERS
    if keyless:
        auth = "none"  # local servers, or the cloud SDK's own credentials (vertex_ai, bedrock)
    elif "credentials" in sources:
        auth = "credentials_source"
    elif sources:
        auth = "api_key"
    else:
        auth = "missing"
    checked = [r for r in rows if r[0] is not None]
    failed = [r for r in checked if r[0] is False]
    if rows and len(failed) == len(rows) and len(sources) == 1:
        problems.append("every stored key failed its last test")
    last = max(checked, key=lambda r: r[2]) if checked else None
    return {
        "provider": provider,
        "configured": keyless or bool(sources),
        "ready": (keyless or bool(sources)) and not problems,
        "auth_type": auth,
        "sources": sources,  # how many keys each source has; never the keys
        "key_checks": {
            "passed": len(checked) - len(failed),
            "failed": len(failed),
            "unchecked": len(rows) - len(checked),
            "last": None if last is None else {"ok": last[0], "error": last[1], "at": last[2]},
        },
        "circuit": breaker.states().get(provider, {"state": "closed"}),
        "problems": problems,
        "touchpoints": [],
    }


def readiness() -> dict[str, Any]:
    """Per touchpoint, each configured model and whether its provider is ready; per provider,
    its credential sources. ``ready`` is false when any touchpoint's default, or any of its
    fallbacks, would fail for want of credentials."""
    from llm_kit import conf
    from llm_kit.core.catalogue.fields import normalise_provider
    from llm_kit.core.touchpoints import all_touchpoints
    from llm_kit.core.touchpoints.env import provider_of
    from llm_kit.core.touchpoints.resolve import canonical
    from llm_kit.models import CatalogueProvider
    from llm_kit.touchpoints.env import DjangoEnv

    env = DjangoEnv()
    env_names = {p: list(e or []) for p, e in CatalogueProvider.objects.values_list("provider", "env")}
    providers: dict[str, dict[str, Any]] = {}
    touchpoints = []
    for tp in sorted(all_touchpoints(), key=lambda t: t.name):
        ov = env.override(tp.name)
        default = (ov.default if ov and ov.default else None) or tp.default
        fallbacks = list(ov.fallbacks) if ov and ov.fallbacks is not None else list(tp.fallbacks)
        settable = getattr(tp, "settable", None)
        policy = (settable.keys if settable is not None else None) or conf.get("KEY_POLICY")
        models, seen = [], set()
        for role, model in [("default", default), *(("fallback", f) for f in fallbacks)]:
            if not model:
                continue
            ref, info = canonical(model, env)
            if ref in seen:
                continue
            seen.add(ref)
            provider = normalise_provider(provider_of(ref)) if provider_of(ref) else ""
            if provider and provider not in providers:
                providers[provider] = _provider(provider, env_names.get(provider, []))
            p = providers.get(provider)
            if p is not None and tp.name not in p["touchpoints"]:
                p["touchpoints"].append(tp.name)
            models.append(
                {
                    "role": role,
                    "ref": ref,
                    "provider": provider,
                    "in_catalogue": info is not None,
                    "ready": bool(p and p["ready"]),
                }
            )
        fallback_rows = [m for m in models if m["role"] == "fallback"]
        touchpoints.append(
            {
                "name": tp.name,
                "disabled": bool(ov and ov.disabled),
                "key_policy": policy,
                "models": models,
                "ready": bool(models and models[0]["ready"]),
                # None: no fallbacks to fail over to
                "failover_ready": all(m["ready"] for m in fallback_rows) if fallback_rows else None,
                "usable": sum(1 for m in models if m["ready"]),
            }
        )
    # Touchpoints whose callers bring their own keys don't need yours to be ready.
    counted = [t for t in touchpoints if t["key_policy"] != "own_only" and not t["disabled"]]
    return {
        "ready": all(t["ready"] and t["failover_ready"] is not False for t in counted),
        "touchpoints": touchpoints,
        "providers": [providers[p] for p in sorted(providers)],
    }
