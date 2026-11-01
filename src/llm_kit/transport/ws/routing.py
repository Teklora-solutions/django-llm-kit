"""WebSocket routes. In asgi.py::

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from llm_kit.transport.ws.routing import websocket_urlpatterns

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(URLRouter(websocket_urlpatterns)),
})
"""

from django.urls import path

from llm_kit.transport.ws.consumers import ThreadConsumer

websocket_urlpatterns = [
    path("ws/llm/threads/", ThreadConsumer.as_asgi(), name="llm_kit_ws_new"),
    path("ws/llm/threads/<uuid:thread_id>/", ThreadConsumer.as_asgi(), name="llm_kit_ws"),
]
