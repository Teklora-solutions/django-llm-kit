import logging

from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied, ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

from llm_kit.core.errors import BudgetExceeded, UnknownTouchpoint
from llm_kit.core.touchpoints import all_touchpoints, get_touchpoint
from llm_kit.models import Thread
from llm_kit.runtime import access, runs
from llm_kit.runtime.graphs import get_graph, graph_names, touchpoint_name
from llm_kit.transport.api.permissions import ThreadAccess
from llm_kit.transport.api.serializers import (
    CatalogueModelSerializer,
    RunSerializer,
    ThreadSerializer,
    TouchpointSerializer,
)

logger = logging.getLogger("llm_kit")


class ThreadViewSet(viewsets.ModelViewSet):
    """CRUD for threads, plus their LangGraph state.

    GET  threads/<id>/state/     current state (values, next, interrupts)
    GET  threads/<id>/history/   checkpoints, newest first (?limit=20)
    POST threads/<id>/runs/      run to completion / next interrupt, return state;
                                 with "background": true, start it detached (202, the run);
                                 "queue": true waits its turn on a busy thread
    """

    serializer_class = ThreadSerializer
    permission_classes = [ThreadAccess]

    def get_queryset(self):
        try:
            return access.threads_for(access.owner_from_request(self.request))
        except access.NotAllowed as e:
            raise PermissionDenied(str(e)) from e

    def get_object(self):
        # Look up by id across all threads; ThreadAccess decides who may use it.
        obj = get_object_or_404(Thread, pk=self.kwargs["pk"])
        self.check_object_permissions(self.request, obj)
        return obj

    def perform_create(self, serializer):
        try:
            fields = access.owner_fields(access.owner_from_request(self.request))
        except access.NotAllowed as e:
            raise PermissionDenied(str(e)) from e
        serializer.save(**fields)

    @action(detail=True, methods=["get"])
    def state(self, request, pk=None):
        thread = self.get_object()
        return Response(runs.state_to_json(get_graph(thread.graph).get_state(thread.config)))

    @action(detail=True, methods=["get"])
    def history(self, request, pk=None):
        thread = self.get_object()
        try:
            limit = min(int(request.query_params.get("limit", 20)), 200)
        except ValueError:
            raise ValidationError({"limit": "Must be an integer."}) from None
        graph = get_graph(thread.graph)
        return Response([runs.state_to_json(s) for s in graph.get_state_history(thread.config, limit=limit)])

    @action(detail=True, methods=["post"], serializer_class=RunSerializer)
    def runs(self, request, pk=None):
        thread = self.get_object()
        body = RunSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        payload = dict(body.validated_data)
        owner = access.owner_from_request(request)
        queue = payload.pop("queue", None)
        if payload.pop("background", False) or queue:
            from llm_kit import background

            try:
                run = background.start(thread, payload, owner, queue=queue)
            except runs.BadRequest as e:
                raise ValidationError({"detail": str(e)}) from e
            except BudgetExceeded as e:
                return Response({"detail": str(e)}, status=status.HTTP_429_TOO_MANY_REQUESTS)
            except background.RunBusy as e:
                return _busy(e)
            return Response(run.to_json(), status=status.HTTP_202_ACCEPTED)
        if (busy := _active_run(thread)) is not None:
            return _busy(busy)
        try:
            runs.validate_choice(thread.graph, payload.get("model"))
            setup = runs.prepare(thread, payload, owner=owner)
        except runs.BadRequest as e:
            raise ValidationError({"detail": str(e)}) from e
        except BudgetExceeded as e:
            return Response({"detail": str(e)}, status=status.HTTP_429_TOO_MANY_REQUESTS)
        except runs.RunRefused as e:
            raise PermissionDenied(str(e)) from e
        except runs.HookFailed as e:
            logger.exception("Run hook failed for thread %s", thread.pk)
            return Response({"detail": runs.error_message(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        state = runs.invoke_run(get_graph(thread.graph), setup.input, setup.config, setup.contexts)
        thread.save(update_fields=["updated_at"])
        _advance(thread)
        return Response(state, status=status.HTTP_200_OK)


def _active_run(thread):
    from llm_kit.background.service import active_run

    return active_run(thread)


def _advance(thread):
    from llm_kit.background.service import advance

    advance(thread)


def _busy(e):
    run = getattr(e, "run", e)
    body = {"detail": "This thread already has a run in progress."}
    if run is not None:
        body["run"] = run.to_json()
    return Response(body, status=status.HTTP_409_CONFLICT)


class TouchpointListView(APIView):
    """Touchpoints declared in code, with who chooses their model."""

    permission_classes = [ThreadAccess]

    def get(self, request):
        return Response(TouchpointSerializer(all_touchpoints(), many=True).data)


class TouchpointModelsView(APIView):
    """Models a client may pick for one touchpoint (empty when the server chooses)."""

    permission_classes = [ThreadAccess]

    def get(self, request, name):
        from llm_kit.touchpoints import allowed_models

        try:
            tp = get_touchpoint(name)
        except UnknownTouchpoint:
            return Response({"detail": f"Unknown touchpoint {name!r}."}, status=status.HTTP_404_NOT_FOUND)
        resolution = tp.resolve()
        return Response(
            {
                "touchpoint": tp.name,
                "default": resolution.model,
                "models": CatalogueModelSerializer(allowed_models(tp), many=True).data,
            }
        )


class GraphListView(APIView):
    permission_classes = [ThreadAccess]

    def get(self, request):
        return Response([{"name": n, "touchpoint": touchpoint_name(n)} for n in graph_names()])
