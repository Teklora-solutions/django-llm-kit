"""Small payloads shaped like the real catalogue sources (checked against live data 2026-11-01)."""

import copy
import json

from llm_kit.core.catalogue import Response
from llm_kit.core.catalogue.sources import LITELLM, MODELS_DEV, OPENROUTER, OPENROUTER_EMBEDDINGS

MODELS_DEV_DATA = {
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic",
        "env": ["ANTHROPIC_API_KEY"],
        "doc": "https://docs.anthropic.com",
        "models": {
            "claude-sonnet-5-5": {
                "id": "claude-sonnet-5-5",
                "name": "Claude Sonnet 5.5",
                "family": "claude-sonnet",
                "attachment": True,
                "reasoning": True,
                "tool_call": True,
                "structured_output": True,
                "temperature": True,
                "knowledge": "2026-03",
                "release_date": "2026-06-01",
                "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                "open_weights": False,
                "limit": {"context": 1000000, "output": 64000},
                "cost": {"input": 2.2, "output": 10, "cache_read": 0.1, "cache_write": 2.5},
                "canonical_model_id": "anthropic/claude-sonnet-5-5",
            },
            "claude-old-1": {
                "id": "claude-old-1",
                "name": "Claude Old",
                "tool_call": False,
                "attachment": False,
                "reasoning": False,
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 100000, "output": 4096},
                "cost": {"input": 8, "output": 24},
            },
        },
    },
    "google": {
        "id": "google",
        "name": "Google",
        "env": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "models": {
            "gemini-3.8-flash": {
                "id": "gemini-3.8-flash",
                "name": "Gemini 3.8 Flash",
                "attachment": True,
                "reasoning": True,
                "tool_call": True,
                "modalities": {"input": ["text", "image", "audio", "video", "pdf"], "output": ["text"]},
                "limit": {"context": 1048576, "output": 65536},
                "cost": {
                    "input": 0.75,
                    "output": 3.75,
                    "cache_read": 0.075,
                    "tiers": [{"input": 1.5, "output": 7.5, "tier": {"type": "context", "size": 200000}}],
                },
            }
        },
    },
    "openrouter": {
        "id": "openrouter",
        "name": "OpenRouter",
        "env": ["OPENROUTER_API_KEY"],
        "models": {
            "google/gemini-3.8-flash": {
                "id": "google/gemini-3.8-flash",
                "name": "Gemini 3.8 Flash",
                "tool_call": True,
                "attachment": True,
                "reasoning": True,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": 1048576, "output": 65536},
                "cost": {"input": 0.75, "output": 3.75},
            }
        },
    },
}

OPENROUTER_DATA = {
    "data": [
        {
            "id": "google/gemini-3.8-flash",
            "canonical_slug": "google/gemini-3.8-flash-20260901",
            "name": "Google: Gemini 3.8 Flash",
            "created": 1788220800,
            "context_length": 1048576,
            "architecture": {"input_modalities": ["text", "image", "file"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0.000000825", "completion": "0.000004125", "input_cache_read": "0.0000000825"},
            "supported_parameters": ["tools", "tool_choice", "response_format", "reasoning", "temperature"],
            "top_provider": {"context_length": 1048576, "max_completion_tokens": 65536},
        },
        {
            "id": "openrouter/auto",
            "name": "Auto Router",
            "context_length": 2000000,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "pricing": {"prompt": "-1", "completion": "-1"},
            "supported_parameters": [],
            "top_provider": {},
        },
    ]
}

OPENROUTER_EMB_DATA = {
    "data": [
        {
            "id": "openai/text-embedding-3-large",
            "name": "OpenAI: Text Embedding 3 Large",
            "context_length": 8192,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["embeddings"]},
            "pricing": {"prompt": "0.00000013", "completion": "0"},
        }
    ]
}

LITELLM_DATA = {
    "sample_spec": {"input_cost_per_token": 0},
    "claude-sonnet-5-5": {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "input_cost_per_token": 2e-06,
        "output_cost_per_token": 1e-05,
        "cache_read_input_token_cost": 1e-07,
        "max_input_tokens": 1000000,
        "max_output_tokens": 64000,
        "supports_function_calling": True,
        "supports_vision": True,
        "supports_prompt_caching": True,
    },
    "gemini/gemini-3.8-flash": {
        "litellm_provider": "gemini",
        "mode": "chat",
        "input_cost_per_token": 7.5e-07,
        "output_cost_per_token": 3.75e-06,
        "max_input_tokens": 1048576,
        "max_output_tokens": 65536,
        "supports_function_calling": True,
        "supports_response_schema": True,
    },
    "us.anthropic.claude-sonnet-5-5-v1:0": {
        "litellm_provider": "bedrock_converse",
        "mode": "chat",
        "input_cost_per_token": 2.2e-06,
        "output_cost_per_token": 1.1e-05,
        "supports_function_calling": True,
    },
    "vertex_ai/claude-sonnet-5-5": {
        "litellm_provider": "vertex_ai-anthropic_models",
        "mode": "chat",
        "input_cost_per_token": 2e-06,
        "output_cost_per_token": 1e-05,
    },
    "text-embedding-3-large": {
        "litellm_provider": "openai",
        "mode": "embedding",
        "input_cost_per_token": 1.3e-07,
        "output_cost_per_token": 0,
        "max_input_tokens": 8191,
    },
}


class FakeSources:
    """A fetcher serving the payloads above, with ETags, 304s and failures on demand."""

    def __init__(self):
        self.data = {
            MODELS_DEV.url: copy.deepcopy(MODELS_DEV_DATA),
            OPENROUTER.url: copy.deepcopy(OPENROUTER_DATA),
            OPENROUTER_EMBEDDINGS.url: copy.deepcopy(OPENROUTER_EMB_DATA),
            LITELLM.url: copy.deepcopy(LITELLM_DATA),
        }
        self.version = {url: 1 for url in self.data}
        self.fail: set[str] = set()
        self.requests: list[tuple[str, dict]] = []

    def bump(self, url):
        self.version[url] += 1

    def __call__(self, url, headers):
        self.requests.append((url, dict(headers)))
        if url in self.fail:
            raise OSError("connection refused")
        etag = f'"v{self.version[url]}"'
        if headers.get("If-None-Match") == etag:
            return Response(304)
        return Response(200, json.dumps(self.data[url]).encode(), {"etag": etag})
