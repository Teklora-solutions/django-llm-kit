from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import BasePermission

from llm_kit.runtime import access


class ThreadAccess(BasePermission):
    def has_permission(self, request, view):
        if not access.can_use_api(request.user):
            return False
        try:  # a TENANT_RESOLVER that refuses this request (a tenant the user isn't in)
            access.owner_from_request(request)
        except access.NotAllowed as e:
            raise PermissionDenied(str(e)) from e
        return True

    def has_object_permission(self, request, view, obj):
        try:
            access.check_thread(obj, access.owner_from_request(request))
        except access.NotAllowed:
            return False
        return True
