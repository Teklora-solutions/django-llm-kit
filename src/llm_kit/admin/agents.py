from django import forms
from django.contrib import admin
from django.utils.html import format_html, format_html_join

from llm_kit.agents.registry import declared_tool_names
from llm_kit.models import AgentOverride


def _choices():
    from llm_kit.core.agents import all_agents

    return [(a.name, a.name) for a in all_agents()]


def _spec(name):
    from llm_kit.core.agents import UnknownAgent, get_agent

    try:
        return get_agent(name)
    except UnknownAgent:
        return None


_ROW = (
    "<tr><th style='text-align:left;padding-right:1em'>{}</th>"
    "<td><pre style='margin:0;white-space:pre-wrap'>{}</pre></td></tr>"
)


class AgentOverrideForm(forms.ModelForm):
    class Meta:
        model = AgentOverride
        fields = "__all__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["agent"] = forms.ChoiceField(choices=_choices(), help_text="Agents declared in code.")

    def clean(self):
        data = super().clean()
        spec = _spec(data.get("agent"))
        if spec is None:
            return data
        if spec.is_code and (data.get("prompt") or data.get("tools") or data.get("max_steps")):
            raise forms.ValidationError("A code agent runs Python: only pausing it applies.")
        known = declared_tool_names(spec)
        if data.get("tools") and known:
            unknown = [t for t in data["tools"] if t not in known]
            if unknown:
                raise forms.ValidationError({"tools": f"Not among this agent's tools: {', '.join(unknown)}."})
        return data


@admin.register(AgentOverride)
class AgentOverrideAdmin(admin.ModelAdmin):
    form = AgentOverrideForm
    list_display = ("agent", "state", "max_steps", "tools_shown", "updated_at")
    search_fields = ("agent", "prompt", "paused", "note")
    readonly_fields = ("in_code",)

    @admin.display(description="State")
    def state(self, obj):
        return f"paused: {obj.paused}" if obj.paused else "running"

    @admin.display(description="Tools")
    def tools_shown(self, obj):
        return ", ".join(obj.tools) if obj.tools else "all"

    @admin.display(description="Declared in code")
    def in_code(self, obj):
        spec = _spec(obj.agent) if obj and obj.agent else None
        if spec is None:
            return "Not declared in code."
        if spec.is_code:
            return format_html(
                "Code agent <code>{}</code>. Subagents: {}", spec.code.__qualname__, ", ".join(spec.subagents) or "none"
            )
        prompt = (
            spec.prompt
            if isinstance(spec.prompt, str)
            else f"(built by {getattr(spec.prompt, '__qualname__', 'a function')})"
        )
        rows = [
            ("Touchpoint", spec.touchpoint),
            ("Max steps", spec.max_steps),
            ("Tools", ", ".join(declared_tool_names(spec)) or ("chosen per run" if callable(spec.tools) else "none")),
            ("Subagents", ", ".join(spec.subagents) or "none"),
            ("Prompt", prompt),
        ]
        return format_html(
            "<table>{}</table>",
            format_html_join("", _ROW, rows),
        )
