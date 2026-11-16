from django.contrib import admin, messages

from llm_kit.models import BackgroundRun


@admin.register(BackgroundRun)
class BackgroundRunAdmin(admin.ModelAdmin):
    list_display = ("id", "kind", "graph", "status", "user", "tenant", "executor", "created_at", "ended_at")
    list_filter = ("kind", "status", "graph", "executor")
    search_fields = ("id", "thread__id", "user__username", "tenant")
    raw_id_fields = ("thread", "user")
    readonly_fields = [f.name for f in BackgroundRun._meta.fields]
    actions = ["cancel_runs"]

    def has_add_permission(self, request):
        return False

    @admin.action(description="Cancel the selected runs")
    def cancel_runs(self, request, queryset):
        n = sum(1 for run in queryset if run.cancel())
        self.message_user(request, f"Asked {n} run(s) to stop.", messages.SUCCESS)
