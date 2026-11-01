"""MODEL_BACKEND used when LLM_KIT_DEMO_OFFLINE=1, so the page works with no API key.

A tiny scripted model: it echoes, and to show the harness it hands messages containing
"research" to the researcher, which looks up notes and reports what it found.
"""

import json
import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class OfflineModel(BaseChatModel):
    model_ref: str = ""

    @property
    def _llm_type(self) -> str:
        return "llm-kit-offline"

    def bind_tools(self, tools, **kwargs):
        from langchain_core.utils.function_calling import convert_to_openai_tool

        return self.bind(tools=[convert_to_openai_tool(t) for t in tools])

    def _reply(self, messages, tools) -> AIMessage:
        names = [t["function"]["name"] for t in tools or []]
        last = messages[-1]
        text = last.content if isinstance(last.content, str) else str(last.content)
        if last.type == "human":
            if "delegate" in names and "research" in text.lower():
                args = {"agent": "chat.researcher", "task": text}
                return AIMessage(content="", tool_calls=[{"name": "delegate", "args": args, "id": _id()}])
            if "find_notes" in names and "delegate" not in names:  # the researcher
                return AIMessage(content="", tool_calls=[{"name": "find_notes", "args": {}, "id": _id()}])
        if last.type == "tool":
            return AIMessage(content=f"[{self.model_ref}] From {last.name}: {text[:400]}")
        return AIMessage(content=f"[{self.model_ref}] You said: {text}")

    def _generate(self, messages, stop=None, run_manager=None, tools=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages, tools))])

    def _stream(self, messages, stop=None, run_manager=None, tools=None, **kwargs):
        reply = self._reply(messages, tools)
        if reply.tool_calls:
            chunks = [
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                for i, c in enumerate(reply.tool_calls)
            ]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
            return
        for word in reply.content.split(" "):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


def _id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def backend(ref, params, api_keys):
    return OfflineModel(model_ref=ref)
