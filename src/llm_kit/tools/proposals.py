"""Approving or rejecting changes agents proposed (``writes = "approval"`` or ``approval_fields``).

    proposals = llm_kit.tools.pending_for(request.user)
    llm_kit.tools.approve(proposal, by=request.user)
    llm_kit.tools.reject(proposal, by=request.user, reason="wrong customer")

Who may decide: someone with the model's ``approve_permission`` (default: its change
permission) who is the requester, in the same tenant (tenant-scoped models), or a
superuser. Approving re-checks the requester's scope and refuses if the row changed
since it was proposed.
"""

from __future__ import annotations

from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.dispatch import Signal
from django.forms.models import model_to_dict
from django.utils import timezone

from llm_kit.models import ToolProposal
from llm_kit.tools.caller import Caller, resolve_tenant
from llm_kit.tools.ops import apply, build_form, form_errors, jsonable
from llm_kit.tools.scope import OutOfScope, scoped

# sender=ToolProposal, proposal=<ToolProposal>
proposal_created = Signal()
proposal_decided = Signal()


class ProposalError(Exception):
    pass


def propose(
    spec, caller: Caller, config, instance, given: dict, form, before: dict, object_repr: str = ""
) -> ToolProposal:
    names = list(given)
    configurable = (config or {}).get("configurable") or {}
    proposal = ToolProposal.objects.create(
        tool=spec.save_name,
        model=spec.label,
        object_pk=str(instance.pk) if instance is not None else "",
        object_repr=object_repr,
        data=jsonable(given),
        before=before,
        after=jsonable({k: form.cleaned_data[k] for k in names}),
        user=caller.user if caller.is_authenticated else None,
        tenant="" if caller.tenant_id is None else str(caller.tenant_id),
        thread_id=str(configurable.get("thread_id") or ""),
    )
    proposal_created.send(sender=ToolProposal, proposal=proposal)
    return proposal


def _spec(proposal):
    from llm_kit.tools import registry

    spec = registry.spec_for_label(proposal.model)
    if spec is None:
        raise ProposalError(f"{proposal.model} is no longer exposed to agents.")
    return spec


def may_decide(proposal: ToolProposal, user, spec=None) -> bool:
    spec = spec or _spec(proposal)
    who = Caller(user)
    if not who.has_perm(spec.approve_permission):
        return False
    if user.is_superuser or spec.unscoped is not None or proposal.user_id == user.pk:
        return True
    if spec.tenant_field and not spec.owner_field:
        tenant = resolve_tenant(user)
        return tenant is not None and str(getattr(tenant, "pk", tenant)) == proposal.tenant
    return False


def pending_for(user) -> list[ToolProposal]:
    """Pending proposals ``user`` may decide."""
    from llm_kit.tools import registry

    out = []
    for p in ToolProposal.objects.filter(status=ToolProposal.PENDING).select_related("user"):
        spec = registry.spec_for_label(p.model)
        if spec is not None and may_decide(p, user, spec):
            out.append(p)
    return out


def _decide(proposal, by, status, reason="", result_pk=""):
    proposal.status = status
    proposal.decided_by = by
    proposal.decided_at = timezone.now()
    proposal.reason = reason
    proposal.result_pk = result_pk
    proposal.save(update_fields=["status", "decided_by", "decided_at", "reason", "result_pk"])


def _locked(proposal, by):
    p = ToolProposal.objects.select_for_update().get(pk=proposal.pk)
    if p.status != ToolProposal.PENDING:
        raise ProposalError(f"Proposal #{p.pk} was already decided ({p.status}).")
    spec = _spec(p)
    if not may_decide(p, by, spec):
        raise PermissionDenied(f"You may not decide proposal #{p.pk}.")
    return p, spec


def approve(proposal: ToolProposal, by) -> ToolProposal:
    """Apply the change as the requester. Status becomes approved, or failed with a reason."""
    with transaction.atomic():
        p, spec = _locked(proposal, by)
        problem, result_pk = _apply(p, spec)
        _decide(p, by, ToolProposal.FAILED if problem else ToolProposal.APPROVED, problem, result_pk)
    proposal_decided.send(sender=ToolProposal, proposal=p)
    return p


def _apply(p: ToolProposal, spec) -> tuple[str, str]:
    """(problem, result_pk): an empty problem means it was applied."""
    if p.user is None:
        return "The user it was proposed for no longer exists.", ""
    requester = Caller(p.user, tenant=p.tenant or None)
    instance = None
    if p.object_pk:
        instance = scoped(spec, requester).filter(pk=p.object_pk).first()
        if instance is None:
            return "The record no longer exists, or the requester can't see it.", ""
        if jsonable(model_to_dict(instance, fields=list(p.data))) != p.before:
            return "The record changed after this was proposed; ask again.", ""
    form = build_form(spec, requester, instance, p.data)
    if not form.is_valid():
        return f"No longer valid: {form_errors(form)}", ""
    try:
        obj = apply(spec, requester, form, creating=instance is None)
    except OutOfScope:
        return "It would put the record outside the requester's records.", ""
    return "", str(obj.pk)


def reject(proposal: ToolProposal, by, reason: str = "") -> ToolProposal:
    with transaction.atomic():
        p, _ = _locked(proposal, by)
        _decide(p, by, ToolProposal.REJECTED, reason)
    proposal_decided.send(sender=ToolProposal, proposal=p)
    return p
