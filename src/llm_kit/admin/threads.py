from django.contrib import admin

from llm_kit.models import Checkpoint, Thread


@admin.register(Thread)
class ThreadAdmin(admin.ModelAdmin):
    list_display = ("__str__", "graph", "owner", "tenant", "created_at", "updated_at")
    list_filter = ("graph",)
    search_fields = ("id", "title", "user__username", "external_key", "tenant")
    readonly_fields = ("id", "session_key", "created_at", "updated_at", "checkpoint_count")

    @admin.display(description="Owner")
    def owner(self, obj):
        return str(obj.user) if obj.user_id else obj.owner_label

    raw_id_fields = ("user",)

    @admin.display(description="Checkpoints")
    def checkpoint_count(self, obj):
        return Checkpoint.objects.filter(thread_id=obj.thread_id).count()
