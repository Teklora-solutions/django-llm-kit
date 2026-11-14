from tests.testapp.models import Member


def tenant_for(user):
    return Member.objects.filter(user=user).values_list("organisation_id", flat=True).first()


# user pk -> extra organisation pks they belong to (tests fill this in)
ALSO_MEMBER: dict = {}


def workspace_for(user, request=None, scope=None):
    """A request-aware resolver: the X-Workspace-Id header picks one of the user's organisations."""
    from django.core.exceptions import PermissionDenied

    picked = None
    if request is not None:
        picked = request.headers.get("X-Workspace-Id")
    elif scope is not None:
        picked = dict(scope.get("headers") or []).get(b"x-workspace-id", b"").decode() or None
    if not picked:
        return tenant_for(user)
    mine = {tenant_for(user), *ALSO_MEMBER.get(user.pk, ())}
    if int(picked) not in mine:
        raise PermissionDenied("Not a member of that workspace.")
    return int(picked)


def can_manage(user, tenant):
    """Who may change a tenant's scoped settings, for tests: alice."""
    return user.username == "alice"
