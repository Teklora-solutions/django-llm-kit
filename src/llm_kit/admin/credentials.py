from django import forms
from django.contrib import admin, messages

from llm_kit.models import ProviderKey


class ProviderKeyForm(forms.ModelForm):
    key = forms.CharField(
        widget=forms.PasswordInput(render_value=False),
        required=False,
        help_text="Stored encrypted. Leave blank to keep the current key.",
    )

    class Meta:
        model = ProviderKey
        fields = ("provider", "label", "key", "active", "scope", "scope_id")

    def clean(self):
        data = super().clean()
        if not self.instance.pk and not data.get("key"):
            self.add_error("key", "Required for a new key.")
        return data

    def save(self, commit=True):
        obj = super().save(commit=False)
        if self.cleaned_data.get("key"):
            obj.set_key(self.cleaned_data["key"])
        if commit:
            obj.save()
        return obj


@admin.register(ProviderKey)
class ProviderKeyAdmin(admin.ModelAdmin):
    form = ProviderKeyForm
    list_display = ("provider", "label", "scope", "scope_id", "fingerprint", "active", "last_check_ok", "updated_at")
    readonly_fields = ("last_checked_at", "last_check_ok", "last_check_error")
    actions = ["test_keys"]

    @admin.action(description="Test the selected keys (one tiny call each)")
    def test_keys(self, request, queryset):
        from llm_kit.credentials.check import check_stored

        for row in queryset:
            result = check_stored(row)
            level = messages.SUCCESS if result.ok else messages.ERROR
            self.message_user(request, f"{row}: {result.message}", level=level)

    list_filter = ("provider", "active", "scope")
    search_fields = ("provider", "label", "scope_id")
