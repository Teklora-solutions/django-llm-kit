"""REST API. Include with ``path("llm/api/", include("llm_kit.transport.api.urls"))``."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from llm_kit.transport.api import monitoring, views

app_name = "llm_kit_api"

router = DefaultRouter()
router.register("threads", views.ThreadViewSet, basename="thread")

urlpatterns = [
    path("touchpoints/", views.TouchpointListView.as_view(), name="touchpoints"),
    path("touchpoints/<str:name>/models/", views.TouchpointModelsView.as_view(), name="touchpoint-models"),
    path("graphs/", views.GraphListView.as_view(), name="graphs"),
    path("monitoring/summary/", monitoring.SummaryView.as_view(), name="monitoring-summary"),
    path("monitoring/timeseries/", monitoring.TimeseriesView.as_view(), name="monitoring-timeseries"),
    path("monitoring/breakdown/", monitoring.BreakdownView.as_view(), name="monitoring-breakdown"),
    path("monitoring/errors/", monitoring.ErrorsView.as_view(), name="monitoring-errors"),
    path("monitoring/top/", monitoring.TopView.as_view(), name="monitoring-top"),
    path("monitoring/providers/", monitoring.ProvidersView.as_view(), name="monitoring-providers"),
    path("monitoring/alerts/", monitoring.AlertsView.as_view(), name="monitoring-alerts"),
    path("", include(router.urls)),
]
