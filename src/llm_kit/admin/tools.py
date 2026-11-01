from django.contrib import admin, messages
from django.core.exceptions import PermissionDenied

from llm_kit.models import ToolProposal


@admin.register(ToolProposal)
class ToolProposalAdmin(admin.ModelAdmin):
    list_display = ("id", "created_at", "status", "tool", "target", "changes", "user", "decided_by")
    list_filter = ("status", "model", "tool")
    search_fields = ("object_repr", "object_pk", "tool", "thread_id")
    date_hierarchy = "created_at"
    actions = ["approve_selected", "reject_selected"]
    readonly_fields = [f.name for f in ToolProposal._meta.fields]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    @admin.display(description="Record")
    def target(self, obj):
        return f"{obj.model} {obj.object_repr or obj.object_pk}" if obj.object_pk else f"new {obj.model}"

    @admin.display(description="Change")
    def changes(self, obj):
        return "; ".join(f"{k}: {obj.before.get(k, '—')} → {v}" for k, v in (obj.after or {}).items())

    def _decide(self, request, queryset, fn, verb):
        from llm_kit import tools

        done = 0
        for proposal in queryset.filter(status=ToolProposal.PENDING):
            try:
                result = fn(proposal, request.user)
            except (PermissionDenied, tools.ProposalError) as e:
                self.message_user(request, f"#{proposal.pk}: {e}", messages.ERROR)
                continue
            if result.status == ToolProposal.FAILED:
                self.message_user(request, f"#{proposal.pk} could not be applied: {result.reason}", messages.WARNING)
            else:
                done += 1
        if done:
            self.message_user(request, f"{verb} {done} proposal(s).", messages.SUCCESS)

    @admin.action(description="Approve and apply selected proposals")
    def approve_selected(self, request, queryset):
        from llm_kit import tools

        self._decide(request, queryset, lambda p, u: tools.approve(p, by=u), "Approved")

    @admin.action(description="Reject selected proposals")
    def reject_selected(self, request, queryset):
        from llm_kit import tools

        self._decide(request, queryset, lambda p, u: tools.reject(p, by=u, reason="Rejected in admin"), "Rejected")
