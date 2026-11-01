from django.contrib import admin

from llm_kit.models import Checkpoint, LLMModel, Thread


@admin.register(LLMModel)
class LLMModelAdmin(admin.ModelAdmin):
    list_display = ("name", "model", "provider", "is_active", "is_default", "updated_at")
    list_filter = ("is_active", "is_default", "provider")
    list_editable = ("is_active",)
    search_fields = ("name", "model", "description")


@admin.register(Thread)
class ThreadAdmin(admin.ModelAdmin):
    list_display = ("__str__", "graph", "user", "created_at", "updated_at")
    list_filter = ("graph",)
    search_fields = ("id", "title", "user__username")
    readonly_fields = ("id", "created_at", "updated_at", "checkpoint_count")
    raw_id_fields = ("user",)

    @admin.display(description="Checkpoints")
    def checkpoint_count(self, obj):
        return Checkpoint.objects.filter(thread_id=obj.thread_id).count()


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
