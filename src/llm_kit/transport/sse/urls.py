"""SSE streaming routes. Include with ``path("llm/", include("llm_kit.transport.sse.urls"))``.

For the REST API add ``path("llm/api/", include("llm_kit.transport.api.urls"))`` (needs DRF).
"""

from django.urls import path

from llm_kit.media import webhooks as media_webhooks
from llm_kit.monitoring.prometheus import metrics
from llm_kit.scoped import views as scoped_views
from llm_kit.transport.sse import views

app_name = "llm_kit"

urlpatterns = [
    path("stream/", views.stream, name="stream-new"),
    path("threads/<uuid:thread_id>/stream/", views.stream, name="stream"),
    path("runs/<uuid:run_id>/watch/", views.watch, name="run-watch"),
    path("metrics/", metrics, name="metrics"),
    path("media/webhooks/<str:provider>/", media_webhooks.receive, name="media-webhook"),
    # Form posts from the settings templates ({% llm_kit_settings %} and friends).
    path("settings/save/", scoped_views.save_setting, name="settings-save"),
    path("settings/keys/add/", scoped_views.add_key, name="settings-key-add"),
    path("settings/keys/<int:pk>/remove/", scoped_views.remove_key, name="settings-key-remove"),
    path("settings/keys/<int:pk>/test/", scoped_views.test_key, name="settings-key-test"),
    path("settings/budgets/save/", scoped_views.save_budget, name="settings-budget-save"),
]
