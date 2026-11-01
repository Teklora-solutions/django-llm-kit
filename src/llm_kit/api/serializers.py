from rest_framework import serializers

from llm_kit.graphs import default_graph_name, graph_names
from llm_kit.models import Thread


class ThreadSerializer(serializers.ModelSerializer):
    graph = serializers.CharField(required=False)

    class Meta:
        model = Thread
        fields = ["id", "graph", "title", "metadata", "created_at", "updated_at"]
        read_only_fields = ["id", "created_at", "updated_at"]

    def validate_graph(self, value):
        if self.instance is not None and value != self.instance.graph:
            raise serializers.ValidationError("A thread's graph cannot be changed.")
        if value not in graph_names():
            raise serializers.ValidationError(f"Unknown graph. Choose from: {', '.join(graph_names())}.")
        return value

    def validate(self, attrs):
        if self.instance is None and not attrs.get("graph"):
            default = default_graph_name()
            if not default:
                raise serializers.ValidationError({"graph": "This field is required."})
            attrs["graph"] = default
        return attrs


class RunSerializer(serializers.Serializer):
    """Body of POST /threads/<id>/runs/. See llm_kit.runs for semantics."""

    message = serializers.JSONField(required=False)
    input = serializers.JSONField(required=False, allow_null=True)
    resume = serializers.JSONField(required=False, allow_null=True)
    model = serializers.CharField(required=False)
    config = serializers.DictField(required=False)


class ModelSpecSerializer(serializers.Serializer):
    name = serializers.CharField()
    model = serializers.CharField()
    description = serializers.CharField()
    source = serializers.CharField()
