from django.contrib import admin
from django.urls import include, path

from chat.views import index

urlpatterns = [
    path("", index),
    path("admin/", admin.site.urls),
    path("llm/", include("llm_kit.urls")),  # SSE streaming
    path("llm/api/", include("llm_kit.api.urls")),  # REST API
]
