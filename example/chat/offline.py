from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class EchoModel(GenericFakeChatModel):
    """Streams back 'You said: ...' word by word. For trying the UI offline."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.messages = iter([AIMessage(content=f"You said: {messages[-1].content}")])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
