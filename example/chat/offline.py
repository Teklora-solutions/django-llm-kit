"""MODEL_BACKEND used when LLM_KIT_DEMO_OFFLINE=1: an echo model, so the UI works with no API key."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class EchoModel(GenericFakeChatModel):
    model_ref: str = ""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.messages = iter([AIMessage(content=f"[{self.model_ref}] You said: {messages[-1].content}")])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return self  # never calls tools


def backend(ref, params, api_keys):
    return EchoModel(messages=iter([]), model_ref=ref)
