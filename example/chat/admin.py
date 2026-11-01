from django.contrib import admin

from chat.models import Note

admin.site.register(Note, list_display=("text", "owner", "pinned", "created_at"))
