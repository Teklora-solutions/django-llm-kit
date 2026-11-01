from django.contrib import admin

from llm_kit.models import Checkpoint, Thread


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
