from django.contrib.auth import get_user_model
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from llm_kit import tools

tools.expose(get_user_model(), fields=["username", "first_name"], owner_field="pk", name="account", plural="accounts")


@tool
def whoami(config: RunnableConfig) -> str:
    """Say who the tool is acting for."""
    caller = tools.caller_from(config)
    return f"user={caller.user.username} tenant={caller.tenant_id}"


tools.register(whoami, permission=tools.AUTHENTICATED, category="testapp", guidance="- `whoami` says who you are.")
