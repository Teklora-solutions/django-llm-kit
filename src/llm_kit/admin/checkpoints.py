from django.contrib import admin

from llm_kit.models import Checkpoint


@admin.register(Checkpoint)
class CheckpointAdmin(admin.ModelAdmin):
    list_display = ("thread_id", "checkpoint_ns", "checkpoint_id", "step", "source", "created_at")
    search_fields = ("thread_id", "checkpoint_id")
    readonly_fields = [f.name for f in Checkpoint._meta.fields if f.name not in ("checkpoint", "metadata_blob")]
    exclude = ("checkpoint", "metadata_blob")

    @admin.display(description="Step")
    def step(self, obj):
        return obj.metadata.get("step")

    @admin.display(description="Source")
    def source(self, obj):
        return obj.metadata.get("source")

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
