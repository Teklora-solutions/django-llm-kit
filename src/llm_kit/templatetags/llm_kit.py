from django import template

register = template.Library()


@register.filter
def pct(value, digits=1):
    """0.0455 -> "4.5%"."""
    try:
        return f"{float(value) * 100:.{int(digits)}f}%"
    except (TypeError, ValueError):
        return "—"


@register.filter
def num(value):
    """2586505 -> "2,586,505"."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return value
