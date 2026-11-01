"""Every place this app calls an LLM. Discovered automatically at startup."""

from llm_kit import ClientChoice, touchpoint

assistant = touchpoint(
    "chat.assistant",
    description="The chat page's tool-calling assistant.",
    default="anthropic/claude-sonnet-5-5",
    fallbacks=["gemini/gemini-3.8-flash"],
    params={"max_tokens": 2048},
    requires={"tool_call": True},
    choice=ClientChoice(
        allow=("anthropic/*", "gemini/*", "openai/*", "openrouter/*"),
        max_price_input=5,  # USD per 1M input tokens
    ),
)
