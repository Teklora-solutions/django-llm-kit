from django.contrib import admin

from llm_kit.models import ScopedBudget, ScopedSetting


@admin.register(ScopedSetting)
class ScopedSettingAdmin(admin.ModelAdmin):
    list_display = ("scope", "scope_id", "touchpoint", "key", "value", "updated_by", "updated_at")
    list_filter = ("scope", "touchpoint", "key")
    search_fields = ("scope_id", "touchpoint", "key")
    readonly_fields = ("updated_by", "updated_at")


@admin.register(ScopedBudget)
class ScopedBudgetAdmin(admin.ModelAdmin):
    list_display = ("scope", "scope_id", "touchpoint", "period", "amount", "counts", "updated_by", "updated_at")
    list_filter = ("scope", "period", "counts")
    search_fields = ("scope_id", "touchpoint")
    readonly_fields = ("updated_by", "updated_at")
