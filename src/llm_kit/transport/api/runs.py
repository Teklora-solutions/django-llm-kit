"""Background runs over REST.

    GET  runs/?status=running&thread=<id>   the requester's runs, newest first (up to 50)
    GET  runs/<id>/                         one run
    POST runs/<id>/cancel/                  ask it to stop

Start one with ``POST threads/<id>/runs/`` and ``"background": true``; watch it with the SSE
view ``llm/runs/<id>/watch/`` or the WebSocket ``{"type": "watch", "run": "<id>"}``.
"""

from __future__ import annotations

import uuid

from rest_framework.exceptions import NotFound, PermissionDenied
from rest_framework.response import Response
from rest_framework.views import APIView

from llm_kit import background
from llm_kit.models import BackgroundRun
from llm_kit.models.runs import STATUS
from llm_kit.runtime import access
from llm_kit.transport.api.permissions import ThreadAccess


def _owner(request):
    try:
        return access.owner_from_request(request)
    except access.NotAllowed as e:
        raise PermissionDenied(str(e)) from e


def _run(request, pk) -> BackgroundRun:
    try:
        return background.get_run(pk, _owner(request))
    except (BackgroundRun.DoesNotExist, access.NotAllowed):
        raise NotFound("Run not found.") from None


class RunListView(APIView):
    permission_classes = [ThreadAccess]

    def get(self, request):
        qs = background.runs_for(_owner(request))
        wanted = request.query_params.get("status")
        if wanted:
            if wanted not in dict(STATUS):
                return Response({"detail": f"status must be one of {', '.join(dict(STATUS))}."}, status=400)
            qs = qs.filter(status=wanted)
        if thread := request.query_params.get("thread"):
            try:
                qs = qs.filter(thread_id=uuid.UUID(thread))
            except ValueError:
                return Response({"detail": "thread must be a thread id."}, status=400)
        return Response([background.refresh(r).to_json() for r in qs[:50]])


class RunDetailView(APIView):
    permission_classes = [ThreadAccess]

    def get(self, request, pk):
        return Response(_run(request, pk).to_json())


class RunCancelView(APIView):
    permission_classes = [ThreadAccess]

    def post(self, request, pk):
        run = _run(request, pk)
        cancelled = run.cancel()
        return Response({**background.refresh(run).to_json(), "cancelled": cancelled}, status=202 if cancelled else 409)
