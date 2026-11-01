"""SSE streaming routes. Include with ``path("llm/", include("llm_kit.urls"))``.

For the REST API add ``path("llm/api/", include("llm_kit.api.urls"))`` (needs DRF).
"""

from django.urls import path

from llm_kit import views

app_name = "llm_kit"

urlpatterns = [
    path("stream/", views.stream, name="stream-new"),
    path("threads/<uuid:thread_id>/stream/", views.stream, name="stream"),
]
