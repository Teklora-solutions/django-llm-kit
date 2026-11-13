from chat.views import index
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("", index),
    path("admin/", admin.site.urls),
    path("llm/", include("llm_kit.transport.sse.urls")),  # SSE streaming
    path("llm/api/", include("llm_kit.transport.api.urls")),  # REST API
]
