import os
import tempfile

SECRET_KEY = "test"
DEBUG = True
USE_TZ = True
ROOT_URLCONF = "tests.urls"
# File-backed so LangGraph's worker threads share the test database.
_DB = os.path.join(tempfile.gettempdir(), f"llm_kit_test_{os.getpid()}.sqlite3")
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": _DB,
        "TEST": {"NAME": _DB},
        "OPTIONS": {"timeout": 20},
    }
}
INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.admin",
    "django.contrib.messages",
    "rest_framework",
    "channels",
    "llm_kit",
    "tests.testapp",
]
MIDDLEWARE = [
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "django.template.context_processors.request",
            ]
        },
    }
]
CHANNEL_LAYERS = {"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
LLM_KIT = {
    "GRAPHS": {
        "chat": "tests.graphs:chat_builder",
        "approval": "tests.graphs:make_approval_graph",
        "boom": "tests.graphs.boom_graph",
        "catalogue": {"graph": "tests.graphs:catalogue_builder", "touchpoint": "tests.chat"},
        "slow": "tests.graphs:slow_builder",
        "tools": "tests.graphs:make_tools_graph",
    },
    "DEFAULT_GRAPH": "chat",
    "MODEL_BACKEND": "tests.graphs:fake_backend",
    "ENCRYPTION_KEYS": ["test-encryption-passphrase"],
    # Recording is enabled per test (see tests/test_observability*.py); a background writer
    # thread inserting rows during unrelated tests would fight their transactions.
    "RECORD_SINKS": [],
    "TENANT_RESOLVER": "tests.testapp.tenancy:tenant_for",
}
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]

# Run the suite on Postgres with: LLM_KIT_TEST_PG=postgres://user:pw@host:port/db pytest
if _pg := os.environ.get("LLM_KIT_TEST_PG"):
    from urllib.parse import urlparse

    _u = urlparse(_pg)
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": _u.path.lstrip("/") or "postgres",
            "USER": _u.username,
            "PASSWORD": _u.password,
            "HOST": _u.hostname,
            "PORT": _u.port or 5432,
        }
    }
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Run the suite on LangGraph's Postgres checkpointer with LLM_KIT_TEST_CHECKPOINTER=postgres (needs LLM_KIT_TEST_PG).
if os.environ.get("LLM_KIT_TEST_CHECKPOINTER"):
    LLM_KIT["CHECKPOINTER"] = os.environ["LLM_KIT_TEST_CHECKPOINTER"]
