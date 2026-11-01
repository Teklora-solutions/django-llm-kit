from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("llm/", include("llm_kit.transport.sse.urls")),
    path("llm/api/", include("llm_kit.transport.api.urls")),
]
