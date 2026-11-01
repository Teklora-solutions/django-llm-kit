from django.conf import settings
from django.db import models


class Note(models.Model):
    """The assistant can find and write the signed-in user's notes; pinning one needs approval."""

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    text = models.TextField()
    pinned = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    class AgentAccess:
        description = "the signed-in user's notes"
        fields = ["text", "pinned", "created_at"]
        search_fields = ["text"]
        filter_fields = ["pinned", "created_at"]
        owner_field = "owner"  # each user's agent sees only their notes; the server sets the owner
        writable_fields = ["text"]
        approval_fields = ["pinned"]  # proposed by the agent, applied when a person approves

    def __str__(self):
        return self.text[:50]
