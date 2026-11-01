from rest_framework.permissions import BasePermission

from llm_kit import access


class ThreadAccess(BasePermission):
    def has_permission(self, request, view):
        return access.can_use_api(request.user)

    def has_object_permission(self, request, view, obj):
        try:
            access.check_thread(obj, request.user)
        except access.NotAllowed:
            return False
        return True
