"""REST API. Include with ``path("llm/api/", include("llm_kit.transport.api.urls"))``."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from llm_kit.transport.api import views

app_name = "llm_kit_api"

router = DefaultRouter()
router.register("threads", views.ThreadViewSet, basename="thread")

urlpatterns = [
    path("models/", views.ModelListView.as_view(), name="models"),
    path("graphs/", views.GraphListView.as_view(), name="graphs"),
    path("", include(router.urls)),
]
