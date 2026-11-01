from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

from llm_kit import access, catalogue, runs
from llm_kit.api.permissions import ThreadAccess
from llm_kit.api.serializers import ModelSpecSerializer, RunSerializer, ThreadSerializer
from llm_kit.graphs import get_graph, graph_names
from llm_kit.models import Thread


class ThreadViewSet(viewsets.ModelViewSet):
    """CRUD for threads, plus their LangGraph state.

    GET  threads/<id>/state/     current state (values, next, interrupts)
    GET  threads/<id>/history/   checkpoints, newest first (?limit=20)
    POST threads/<id>/runs/      run to completion / next interrupt, return state
    """

    serializer_class = ThreadSerializer
    permission_classes = [ThreadAccess]

    def get_queryset(self):
        return access.threads_for(self.request.user)

    def get_object(self):
        # Look up by id across all threads; ThreadAccess decides who may use it.
        obj = get_object_or_404(Thread, pk=self.kwargs["pk"])
        self.check_object_permissions(self.request, obj)
        return obj

    def perform_create(self, serializer):
        serializer.save(user=access.owner_for_new_thread(self.request.user))

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
        try:
            runs.validate_model(payload.get("model"))
            graph_input = runs.build_input(payload)
            config = runs.build_config(thread, payload, request.user)
        except runs.BadRequest as e:
            raise ValidationError({"detail": str(e)}) from e
        state = runs.invoke_run(get_graph(thread.graph), graph_input, config)
        thread.save(update_fields=["updated_at"])
        return Response(state, status=status.HTTP_200_OK)


class ModelListView(APIView):
    """Selectable chat models from the catalogue (DB + LLM_KIT['MODELS'])."""

    permission_classes = [ThreadAccess]

    def get(self, request):
        return Response(ModelSpecSerializer(catalogue.available(), many=True).data)


class GraphListView(APIView):
    permission_classes = [ThreadAccess]

    def get(self, request):
        return Response([{"name": n} for n in graph_names()])
