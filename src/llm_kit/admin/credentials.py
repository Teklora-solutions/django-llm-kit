from django import forms
from django.contrib import admin

from llm_kit.models import ProviderKey


class ProviderKeyForm(forms.ModelForm):
    key = forms.CharField(
        widget=forms.PasswordInput(render_value=False),
        required=False,
        help_text="Stored encrypted. Leave blank to keep the current key.",
    )

    class Meta:
        model = ProviderKey
        fields = ("provider", "label", "key", "active")

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
    list_display = ("provider", "label", "fingerprint", "active", "updated_at")
    list_filter = ("provider", "active")
    search_fields = ("provider", "label")
