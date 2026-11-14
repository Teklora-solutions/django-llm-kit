"""Every place this app calls an LLM. Discovered automatically at startup."""

from llm_kit import ClientChoice, KeyPolicy, Param, Settable, touchpoint

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
    # What signed-in users may change on /settings/ (or through llm/api/settings/).
    settable=Settable(
        model=("user",),
        params={"temperature": Param("number", min=0, max=1, scopes=("user",), help="Lower is more predictable.")},
        keys=KeyPolicy.OWN_THEN_PLATFORM,  # a user's own key first, then the demo's
    ),
)

researcher = touchpoint(
    "chat.researcher",
    description="The read-only researcher the assistant hands questions to.",
    default="gemini/gemini-3.8-flash",
    fallbacks=["anthropic/claude-haiku-5-5"],
    params={"max_tokens": 1024},
    requires={"tool_call": True},
)
