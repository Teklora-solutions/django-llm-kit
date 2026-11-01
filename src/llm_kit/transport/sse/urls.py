"""SSE streaming routes. Include with ``path("llm/", include("llm_kit.transport.sse.urls"))``.

For the REST API add ``path("llm/api/", include("llm_kit.transport.api.urls"))`` (needs DRF).
"""

from django.urls import path

from llm_kit.transport.sse import views

app_name = "llm_kit"

urlpatterns = [
    path("stream/", views.stream, name="stream-new"),
    path("threads/<uuid:thread_id>/stream/", views.stream, name="stream"),
]
