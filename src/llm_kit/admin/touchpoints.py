from django import forms
from django.contrib import admin
from django.utils.html import format_html

from llm_kit.models import TouchpointOverride


def _choices():
    from llm_kit.core.touchpoints import all_touchpoints

    return [(t.name, t.name) for t in all_touchpoints()]


class TouchpointOverrideForm(forms.ModelForm):
    class Meta:
        model = TouchpointOverride
        fields = "__all__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["name"] = forms.ChoiceField(choices=_choices(), help_text="Touchpoints declared in code.")


@admin.register(TouchpointOverride)
class TouchpointOverrideAdmin(admin.ModelAdmin):
    form = TouchpointOverrideForm
    list_display = ("name", "default", "disabled", "code_default", "resolves_to", "updated_at")
    list_filter = ("disabled",)
    search_fields = ("name", "default", "note")
    readonly_fields = ("code_policy",)

    def _tp(self, obj):
        from llm_kit.core.errors import UnknownTouchpoint
        from llm_kit.core.touchpoints import get_touchpoint

        try:
            return get_touchpoint(obj.name)
        except UnknownTouchpoint:
            return None

    @admin.display(description="Code default")
    def code_default(self, obj):
        tp = self._tp(obj)
        return tp.default if tp else "— (not declared in code)"

    @admin.display(description="Resolves to")
    def resolves_to(self, obj):
        tp = self._tp(obj)
        if tp is None or obj.disabled:
            return "—"
        try:
            r = tp.resolve()
        except Exception as e:  # show configuration errors instead of a 500
            return f"error: {e}"
        return " → ".join(r.candidates)

    @admin.display(description="Policy in code")
    def code_policy(self, obj):
        tp = self._tp(obj)
        if tp is None:
            return "Not declared in code."
        return format_html(
            "<pre style='margin:0'>default: {}\nfallbacks: {}\nparams: {}\nchoice: {}\nrequires: {}</pre>",
            tp.default,
            list(tp.fallbacks),
            tp.params,
            tp.choice,
            tp.requires,
        )
