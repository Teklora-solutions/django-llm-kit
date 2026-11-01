import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-not-secret")
DEBUG = True
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

INSTALLED_APPS = [
    "daphne",  # makes `runserver` serve ASGI, so SSE and WebSockets stream
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "channels",
    "llm_kit",
    "chat",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]
ROOT_URLCONF = "project.urls"
ASGI_APPLICATION = "project.asgi.application"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
        # Recommended for SQLite: LangGraph writes checkpoints from worker threads.
        "OPTIONS": {"transaction_mode": "IMMEDIATE", "timeout": 20},
    }
}
CHANNEL_LAYERS = {"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
STATIC_URL = "static/"
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LLM_KIT = {
    "GRAPHS": {"assistant": "chat.graphs:builder"},
    "DEFAULT_MODEL": os.environ.get("LLM_KIT_MODEL", "anthropic:claude-sonnet-5-5"),
    "MODELS": {
        "fast": {"model": "anthropic:claude-haiku-5-5", "temperature": 0},
        "smart": {"model": "anthropic:claude-opus-5-5"},
    },
    # Demo only: lets you try the page without logging in. Keep True in real apps.
    "REQUIRE_AUTH": False,
}
