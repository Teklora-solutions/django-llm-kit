from django.contrib import admin

from llm_kit.models import LLMModel


@admin.register(LLMModel)
class LLMModelAdmin(admin.ModelAdmin):
    list_display = ("name", "model", "provider", "is_active", "is_default", "updated_at")
    list_filter = ("is_active", "is_default", "provider")
    list_editable = ("is_active",)
    search_fields = ("name", "model", "description")
