"""Online model metadata sources. Each turns its payload into normalised partial records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llm_kit.core.catalogue.fields import Key, clean, normalise_provider, per_mtok


@dataclass
class ParseResult:
    models: dict[Key, dict[str, Any]] = field(default_factory=dict)
    # provider -> {"name", "env": [...], "doc"}
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, key: Key, partial: dict[str, Any]) -> None:
        partial = clean(partial)
        existing = self.models.get(key)
        if existing is None:
            self.models[key] = partial
        else:  # same model listed twice by one source: keep the first non-empty value per field
            for k, v in partial.items():
                existing.setdefault(k, v)


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    parse: Callable[[Any], ParseResult]


# ----------------------------------------------------------------- models.dev


def _md_date(value: Any) -> str | None:
    return str(value) if value else None


def parse_models_dev(data: dict) -> ParseResult:
    out = ParseResult()
    for provider_id, prov in data.items():
        if not isinstance(prov, dict):
            continue
        provider = normalise_provider(prov.get("id") or provider_id)
        info = out.providers.setdefault(
            provider, {"name": prov.get("name") or provider, "env": [], "doc": prov.get("doc")}
        )
        for env in prov.get("env") or []:
            if env not in info["env"]:
                info["env"].append(env)
        for model_id, m in (prov.get("models") or {}).items():
            cost = m.get("cost") or {}
            limit = m.get("limit") or {}
            modalities = m.get("modalities") or {}
            tiers = []
            for t in cost.get("tiers") or []:
                size = (t.get("tier") or {}).get("size")
                if size:
                    tiers.append(
                        {
                            "above_tokens": int(size),
                            **{
                                f"price_{k}": per_mtok(t.get(k), per_token=False)
                                for k in ("input", "output", "cache_read", "cache_write")
                                if t.get(k) is not None
                            },
                        }
                    )
            output = modalities.get("output") or []
            kind = "embedding" if m.get("type") == "embedding" or "embedding" in output else "chat"
            out.add(
                (provider, model_id),
                {
                    "name": m.get("name"),
                    "family": m.get("family"),
                    "kind": kind,
                    "canonical_id": m.get("canonical_model_id"),
                    "input_modalities": modalities.get("input"),
                    "output_modalities": output,
                    "context_window": limit.get("context") or None,
                    "max_input": limit.get("input") or None,
                    "max_output": limit.get("output") or None,
                    "tool_call": m.get("tool_call"),
                    "structured_output": m.get("structured_output"),
                    "reasoning": m.get("reasoning"),
                    "attachments": m.get("attachment"),
                    "temperature": m.get("temperature"),
                    "prompt_caching": True if cost.get("cache_read") is not None else None,
                    "open_weights": m.get("open_weights"),
                    "price_input": per_mtok(cost.get("input"), per_token=False),
                    "price_output": per_mtok(cost.get("output"), per_token=False),
                    "price_cache_read": per_mtok(cost.get("cache_read"), per_token=False),
                    "price_cache_write": per_mtok(cost.get("cache_write"), per_token=False),
                    "price_reasoning": per_mtok(cost.get("reasoning"), per_token=False),
                    "price_tiers": tiers or None,
                    "release_date": _md_date(m.get("release_date")),
                    "knowledge_cutoff": _md_date(m.get("knowledge")),
                    "status": m.get("status"),
                },
            )
    return out


# ----------------------------------------------------------------- OpenRouter


def _openrouter(data: dict, kind: str) -> ParseResult:
    out = ParseResult()
    for m in data.get("data") or []:
        pricing = m.get("pricing") or {}
        arch = m.get("architecture") or {}
        params = set(m.get("supported_parameters") or [])
        top = m.get("top_provider") or {}
        created = m.get("created")
        release = None
        if isinstance(created, (int, float)) and created > 0:
            from datetime import UTC, datetime

            release = datetime.fromtimestamp(created, UTC).date().isoformat()
        out.add(
            ("openrouter", m["id"]),
            {
                "name": m.get("name"),
                "kind": kind,
                "canonical_id": m.get("canonical_slug"),
                "input_modalities": arch.get("input_modalities"),
                "output_modalities": arch.get("output_modalities"),
                "context_window": m.get("context_length") or top.get("context_length"),
                "max_output": top.get("max_completion_tokens"),
                "tool_call": ("tools" in params) if params else None,
                "structured_output": bool(params & {"structured_outputs", "response_format"}) if params else None,
                "reasoning": ("reasoning" in params) if params else None,
                "temperature": ("temperature" in params) if params else None,
                "prompt_caching": True if pricing.get("input_cache_read") else None,
                "price_input": per_mtok(pricing.get("prompt"), per_token=True),
                "price_output": per_mtok(pricing.get("completion"), per_token=True),
                "price_cache_read": per_mtok(pricing.get("input_cache_read"), per_token=True),
                "price_cache_write": per_mtok(pricing.get("input_cache_write"), per_token=True),
                "price_reasoning": per_mtok(pricing.get("internal_reasoning"), per_token=True),
                "release_date": release,
                "knowledge_cutoff": m.get("knowledge_cutoff"),
                "deprecation_date": m.get("expiration_date"),
            },
        )
    return out


def parse_openrouter(data: dict) -> ParseResult:
    return _openrouter(data, "chat")


def parse_openrouter_embeddings(data: dict) -> ParseResult:
    return _openrouter(data, "embedding")


# -------------------------------------------------------------------- LiteLLM

_LITELLM_KIND = {
    "chat": "chat",
    "responses": "chat",
    "completion": "chat",
    "embedding": "embedding",
    "image_generation": "image",
    "image_edit": "image",
    "audio_transcription": "audio",
    "audio_speech": "audio",
    "realtime": "audio",
    "rerank": "rerank",
}


def parse_litellm(data: dict) -> ParseResult:
    out = ParseResult()
    for raw_key, m in data.items():
        if raw_key == "sample_spec" or not isinstance(m, dict):
            continue
        raw_provider = m.get("litellm_provider") or ""
        provider = normalise_provider(raw_provider)
        model_id = raw_key
        for prefix in {raw_provider, provider, raw_provider.split("-")[0]}:
            if prefix and model_id.startswith(prefix + "/"):
                model_id = model_id[len(prefix) + 1 :]
                break
        if not provider:
            provider, _, model_id = raw_key.partition("/") if "/" in raw_key else ("unknown", "", raw_key)
        inputs = ["text"]
        if m.get("supports_vision"):
            inputs.append("image")
        if m.get("supports_pdf_input"):
            inputs.append("pdf")
        if m.get("supports_audio_input"):
            inputs.append("audio")
        mode = m.get("mode")
        kind = _LITELLM_KIND.get(mode, "other")
        out.add(
            (provider, model_id),
            {
                "kind": kind,
                "input_modalities": inputs if kind == "chat" else None,
                "context_window": m.get("max_input_tokens") or (m.get("max_tokens") if kind == "chat" else None),
                "max_input": m.get("max_input_tokens"),
                "max_output": m.get("max_output_tokens"),
                "tool_call": m.get("supports_function_calling"),
                "structured_output": m.get("supports_response_schema"),
                "reasoning": m.get("supports_reasoning"),
                "attachments": True if (m.get("supports_vision") or m.get("supports_pdf_input")) else None,
                "prompt_caching": m.get("supports_prompt_caching"),
                "price_input": per_mtok(m.get("input_cost_per_token"), per_token=True),
                "price_output": per_mtok(m.get("output_cost_per_token"), per_token=True),
                "price_cache_read": per_mtok(m.get("cache_read_input_token_cost"), per_token=True),
                "price_cache_write": per_mtok(m.get("cache_creation_input_token_cost"), per_token=True),
                "price_reasoning": per_mtok(m.get("output_cost_per_reasoning_token"), per_token=True),
                "deprecation_date": m.get("deprecation_date"),
            },
        )
    return out


MODELS_DEV = Source("models.dev", "https://models.dev/api.json", parse_models_dev)
OPENROUTER = Source("openrouter", "https://openrouter.ai/api/v1/models", parse_openrouter)
OPENROUTER_EMBEDDINGS = Source(
    "openrouter-embeddings", "https://openrouter.ai/api/v1/embeddings/models", parse_openrouter_embeddings
)
LITELLM = Source(
    "litellm",
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
    parse_litellm,
)

DEFAULT_SOURCES: tuple[Source, ...] = (MODELS_DEV, OPENROUTER, OPENROUTER_EMBEDDINGS, LITELLM)
