"""Agents for the test-suite. Their touchpoints use scripted models (tests.graphs.SCRIPTS)."""

from pydantic import BaseModel

from llm_kit import agent, tools, touchpoint

touchpoint("tests.assistant", default="ollama/script-assistant")
touchpoint("tests.researcher", default="ollama/script-researcher")
touchpoint("tests.helper", default="ollama/script-helper")


class Report(BaseModel):
    answer: str
    ids: list[int] = []


agent(
    "tests.assistant",
    description="Answers staff questions about customers and invoices.",
    touchpoint="tests.assistant",
    prompt='You help {user} of tenant {tenant}. Today is {date}. Keep {braces} and {"json": 1} as written.',
    tools=tools.select(categories=["testapp"]),
    subagents=["tests.researcher", "tests.newsroom"],
    max_steps=4,
)

agent(
    "tests.researcher",
    description="Looks invoices up and reports facts with ids; changes nothing.",
    touchpoint="tests.researcher",
    prompt="Research the task.",
    tools=tools.select(kinds=["find"]),
    subagents=["tests.helper"],
    output=Report,
)

agent(
    "tests.helper",
    description="Helps the researcher.",
    touchpoint="tests.helper",
    prompt="Help.",
    tools=["whoami"],
)


@agent.code("tests.newsroom", description="Researches, then writes a headline.", subagents=["tests.researcher"])
def newsroom(task, ctx):
    report = ctx.run("tests.researcher", task)
    return {"headline": report.output.answer.upper(), "user": ctx.caller.user.username, "thread": ctx.thread_id}
