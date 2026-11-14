from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie


@ensure_csrf_cookie
def index(request):
    return render(request, "chat/index.html")


@login_required
def settings_page(request):
    """A user's AI settings, built from llm-kit's template tags."""
    return render(request, "chat/settings.html")
