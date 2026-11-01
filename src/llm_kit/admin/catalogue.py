import json

from django.contrib import admin
from django.utils.html import format_html

from llm_kit.models import (
    CatalogueChange,
    CatalogueModel,
    CatalogueOverride,
    CatalogueProvider,
    CatalogueSource,
)


def _pretty(data) -> str:
    return format_html("<pre style='white-space:pre-wrap;margin:0'>{}</pre>", json.dumps(data, indent=2, default=str))


def _price(value) -> str:
    return "—" if value is None else f"${format(value.normalize(), 'f')}"


class _ReadOnly(admin.ModelAdmin):
    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(CatalogueModel)
class CatalogueModelAdmin(_ReadOnly):
    list_display = (
        "ref",
        "kind",
        "context_window",
        "input_price",
        "output_price",
        "tool_call",
        "reasoning",
        "status",
        "conflict_flag",
        "hidden",
        "removed",
    )
    list_filter = ("kind", "removed", "hidden", "tool_call", "structured_output", "reasoning", "attachments", "status")
    search_fields = ("provider", "model_id", "name", "family", "canonical_id")
    list_per_page = 50
    actions = ["hide", "unhide"]
    exclude = ("values", "source_values", "provenance", "conflicts")
    readonly_fields = ("provenance_view", "conflicts_view", "source_values_view")

    @admin.display(description="Input $/1M", ordering="price_input")
    def input_price(self, obj):
        return _price(obj.price_input)

    @admin.display(description="Output $/1M", ordering="price_output")
    def output_price(self, obj):
        return _price(obj.price_output)

    @admin.display(description="Conflicts", boolean=True)
    def conflict_flag(self, obj):
        return bool(obj.conflicts)

    @admin.display(description="Field sources")
    def provenance_view(self, obj):
        return _pretty(obj.provenance)

    @admin.display(description="Source disagreements")
    def conflicts_view(self, obj):
        return _pretty(obj.conflicts) if obj.conflicts else "—"

    @admin.display(description="What each source says")
    def source_values_view(self, obj):
        return _pretty(obj.source_values)

    def _set_hidden(self, queryset, hidden: bool):
        for m in queryset:
            o, _ = CatalogueOverride.objects.get_or_create(provider=m.provider, model_id=m.model_id)
            o.values = {**(o.values or {}), "hidden": hidden}
            o.save()  # signal re-applies the override

    @admin.action(description="Hide from selection")
    def hide(self, request, queryset):
        self._set_hidden(queryset, True)

    @admin.action(description="Unhide")
    def unhide(self, request, queryset):
        self._set_hidden(queryset, False)


@admin.register(CatalogueOverride)
class CatalogueOverrideAdmin(admin.ModelAdmin):
    list_display = ("__str__", "values", "note", "updated_at")
    search_fields = ("provider", "model_id", "note")


@admin.register(CatalogueChange)
class CatalogueChangeAdmin(_ReadOnly):
    list_display = ("synced_at", "kind", "provider", "model_id", "field", "old", "new")
    list_filter = ("kind", "synced_at")
    search_fields = ("provider", "model_id")
    date_hierarchy = "synced_at"


@admin.register(CatalogueSource)
class CatalogueSourceAdmin(_ReadOnly):
    list_display = ("name", "last_status", "record_count", "last_success_at", "last_fetched_at", "last_error")


@admin.register(CatalogueProvider)
class CatalogueProviderAdmin(_ReadOnly):
    list_display = ("provider", "name", "env", "credentials")
    search_fields = ("provider", "name")

    @admin.display(description="Key in env", boolean=True)
    def credentials(self, obj):
        return obj.has_credentials
