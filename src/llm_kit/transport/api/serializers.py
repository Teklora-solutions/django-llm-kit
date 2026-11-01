from rest_framework import serializers

from llm_kit.models import Thread
from llm_kit.runtime.graphs import default_graph_name, graph_names


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
    """Body of POST /threads/<id>/runs/. See llm_kit.runtime.runs for semantics."""

    message = serializers.JSONField(required=False)
    input = serializers.JSONField(required=False, allow_null=True)
    resume = serializers.JSONField(required=False, allow_null=True)
    model = serializers.CharField(required=False)
    config = serializers.DictField(required=False)


class TouchpointSerializer(serializers.Serializer):
    name = serializers.CharField()
    description = serializers.CharField()
    choice = serializers.SerializerMethodField()
    default = serializers.CharField()

    def get_choice(self, tp) -> str:
        from llm_kit.core.touchpoints import ClientChoice

        return "client" if isinstance(tp.choice, ClientChoice) else "server"


class CatalogueModelSerializer(serializers.Serializer):
    ref = serializers.CharField()
    name = serializers.CharField()
    provider = serializers.CharField()
    kind = serializers.CharField()
    context_window = serializers.IntegerField(allow_null=True)
    max_output = serializers.IntegerField(allow_null=True)
    tool_call = serializers.BooleanField(allow_null=True)
    reasoning = serializers.BooleanField(allow_null=True)
    input_modalities = serializers.ListField(source="input_modality_list")
    price_input = serializers.DecimalField(max_digits=20, decimal_places=8, allow_null=True, coerce_to_string=True)
    price_output = serializers.DecimalField(max_digits=20, decimal_places=8, allow_null=True, coerce_to_string=True)
