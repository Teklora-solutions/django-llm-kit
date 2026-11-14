from django.contrib import admin

from llm_kit.models import RateLimit


@admin.register(RateLimit)
class RateLimitAdmin(admin.ModelAdmin):
    list_display = ("model", "rpm", "tpm", "rpd", "used_now", "note", "updated_at")
    search_fields = ("model", "note")

    @admin.display(description="Used now (rpm / tpm / rpd)")
    def used_now(self, obj):
        if any(c in obj.model for c in "*?["):
            return "per matching model"
        from llm_kit import ratelimits

        current = ratelimits.limiter()
        if current is None:
            return "-"
        u = current.usage(obj.model)
        return f"{u['rpm_used']} / {u['tpm_used']} / {u['rpd_used']}"
