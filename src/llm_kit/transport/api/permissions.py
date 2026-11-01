from rest_framework.permissions import BasePermission

from llm_kit.runtime import access


class ThreadAccess(BasePermission):
    def has_permission(self, request, view):
        return access.can_use_api(request.user)

    def has_object_permission(self, request, view, obj):
        try:
            access.check_thread(obj, access.owner_from_request(request))
        except access.NotAllowed:
            return False
        return True
