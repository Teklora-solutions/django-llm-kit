"""The tool catalogue and proposals, for your own UI.

GET  tools/                               tools the current user may use (?category=crm&kind=find)
GET  tools/proposals/                     pending proposals the current user may decide
POST tools/proposals/<id>/approve/
POST tools/proposals/<id>/reject/         {"reason": "..."}
"""

from __future__ import annotations

from django.core.exceptions import PermissionDenied as DjangoPermissionDenied
from django.shortcuts import get_object_or_404
from rest_framework.exceptions import PermissionDenied, ValidationError
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from llm_kit import tools
from llm_kit.models import ToolProposal


def proposal_json(p: ToolProposal) -> dict:
    return {
        "id": p.pk,
        "status": p.status,
        "tool": p.tool,
        "model": p.model,
        "action": p.action,
        "object_pk": p.object_pk or None,
        "object_repr": p.object_repr,
        "before": p.before,
        "after": p.after,
        "user": p.user_id,
        "thread_id": p.thread_id or None,
        "created_at": p.created_at,
        "decided_at": p.decided_at,
        "decided_by": p.decided_by_id,
        "reason": p.reason,
        "result_pk": p.result_pk or None,
    }


class ToolListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        categories = request.query_params.getlist("category") or None
        kinds = request.query_params.getlist("kind") or None
        out = []
        for entry in tools.entries_for(request.user, categories=categories, kinds=kinds):
            tool = entry.tool()
            out.append(
                {
                    "name": entry.name,
                    "description": entry.description,
                    "category": entry.category,
                    "kind": entry.kind,
                    "writes": entry.writes,
                    "model": entry.spec.label if entry.spec else None,
                    "parameters": tool.tool_call_schema.model_json_schema()
                    if hasattr(tool.tool_call_schema, "model_json_schema")
                    else tool.tool_call_schema,
                }
            )
        return Response(out)


class ProposalListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response([proposal_json(p) for p in tools.pending_for(request.user)])


class ProposalDecisionView(APIView):
    permission_classes = [IsAuthenticated]
    decision = "approve"

    def post(self, request, pk):
        proposal = get_object_or_404(ToolProposal, pk=pk)
        try:
            if self.decision == "approve":
                proposal = tools.approve(proposal, by=request.user)
            else:
                proposal = tools.reject(proposal, by=request.user, reason=str(request.data.get("reason", ""))[:2000])
        except DjangoPermissionDenied as e:
            raise PermissionDenied(str(e)) from e
        except tools.ProposalError as e:
            raise ValidationError({"detail": str(e)}) from e
        return Response(proposal_json(proposal))
