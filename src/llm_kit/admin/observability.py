import json

from django.contrib import admin
from django.utils.html import format_html

from llm_kit.models import LLMCall, LLMRun, ToolCall


def _usd(value) -> str:
    return "—" if value is None else f"${value.normalize():f}"


class _ReadOnly(admin.ModelAdmin):
    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(LLMCall)
class LLMCallAdmin(_ReadOnly):
    list_display = ("started_at", "touchpoint", "model_used", "chosen_by", "tokens", "usd", "latency", "outcome")
    list_filter = ("status", "error_class", "chosen_by", "touchpoint", "unpriced", "usage_estimated", "streamed")
    search_fields = ("touchpoint", "model", "served_model", "thread_id", "user", "tenant", "run_id", "call_id")
    date_hierarchy = "started_at"
    list_per_page = 50
    exclude = ("prompt", "completion")
    readonly_fields = ("run_link", "content")

    @admin.display(description="Cost", ordering="cost")
    def usd(self, obj):
        return "unpriced" if obj.unpriced else _usd(obj.cost)

    @admin.display(description="Model", ordering="model")
    def model_used(self, obj):
        return f"{obj.model} (fallback {obj.fallback_index})" if obj.fallback_index else obj.model

    @admin.display(description="Tokens in / out")
    def tokens(self, obj):
        text = f"{obj.input_tokens} / {obj.output_tokens}"
        return f"~{text}" if obj.usage_estimated else text

    @admin.display(description="Latency", ordering="latency_ms")
    def latency(self, obj):
        first = f" (first token {obj.time_to_first_token_ms} ms)" if obj.time_to_first_token_ms is not None else ""
        return f"{obj.latency_ms} ms{first}"

    @admin.display(description="Status", ordering="status")
    def outcome(self, obj):
        return f"{obj.status}: {obj.error_class}" if obj.error_class else obj.status

    @admin.display(description="Run")
    def run_link(self, obj):
        if not obj.run_id:
            return "—"
        return format_html('<a href="../../../llmrun/?q={}">{}</a>', obj.run_id, obj.run_id)

    @admin.display(description="Captured content")
    def content(self, obj):
        if obj.prompt is None:
            return "Not captured (enable with touchpoint(..., capture=True) or CAPTURE_CONTENT)."
        data = {"prompt": obj.prompt, "completion": obj.completion}
        return format_html("<pre style='white-space:pre-wrap'>{}</pre>", json.dumps(data, indent=2, default=str))


@admin.register(LLMRun)
class LLMRunAdmin(_ReadOnly):
    list_display = ("started_at", "name", "graph", "status", "calls", "tool_calls", "tokens", "usd", "duration_ms")
    list_filter = ("status", "graph", "name")
    search_fields = ("run_id", "thread_id", "user", "tenant", "name")
    date_hierarchy = "started_at"
    readonly_fields = ("call_links",)

    @admin.display(description="Tokens in/out")
    def tokens(self, obj):
        return f"{obj.input_tokens} / {obj.output_tokens}"

    @admin.display(description="Cost", ordering="cost")
    def usd(self, obj):
        text = _usd(obj.cost)
        return f"{text} (+{obj.unpriced_calls} unpriced)" if obj.unpriced_calls else text

    @admin.display(description="Calls")
    def call_links(self, obj):
        return format_html('<a href="../../../llmcall/?q={}">Model calls in this run</a>', obj.run_id)


@admin.register(ToolCall)
class ToolCallAdmin(_ReadOnly):
    list_display = ("started_at", "name", "status", "error_class", "duration_ms", "run_id")
    list_filter = ("status", "name")
    search_fields = ("name", "run_id")
    date_hierarchy = "started_at"
