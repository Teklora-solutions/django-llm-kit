---
description: Declare agents and subagents in code; one harness runs them all, with delegation, limits, history windows, deferred tools, admin overrides and a recorded run tree.
---

# Agents and subagents

One harness runs every agent. An agent is a declaration: its model (a [touchpoint](touchpoints.md)), its prompt and its tools. A subagent is the same harness with a different declaration, the way Claude Code's subagents are Claude Code with another model, prompt and tool set.

## Declaring agents

Put declarations in `<app>/agents.py`; it is imported at startup.

```python
# myapp/agents.py
from llm_kit import agent, tools

agent(
    "crm.assistant",
    description="Answers staff questions about customers and invoices.",
    touchpoint="crm.assistant",                       # model policy, client choice, fallbacks, overrides
    prompt="You help {user}. Today is {date}.",       # template ({date} {now} {user} {tenant} {agent}) or fn(ctx)
    tools=[current_time, tools.select(categories=["crm"])],
    subagents=["crm.researcher"],
    max_steps=12,
)

agent(
    "crm.researcher",
    description="Looks through invoices to answer one question; reports facts with ids. Changes nothing.",
    touchpoint="crm.cheap",                           # another model
    prompt="Answer the task from the records. Quote ids.",
    tools=tools.select(kinds=["find"]),               # read-only
    output=ResearchReport,                            # optional pydantic shape for the answer
)

@agent.code("crm.weekly_report", description="Builds the weekly report.", subagents=["crm.researcher"])
def weekly_report(task, ctx):                         # Python instead of a model loop
    facts = ctx.run("crm.researcher", "What was invoiced this week?")
    return facts.output
```

The `description` is what a parent agent sees when it decides whom to hand work to, so write it for the model.

## Running agents

**Over HTTP:** every agent is a graph by its name. `POST /llm/stream/` with `{"graph": "crm.assistant", "message": "…"}`, threads, the WebSocket and the model picker work without listing the agent in `GRAPHS`.

**From code:**

```python
from llm_kit import agents

result = agents.run("crm.researcher", "Which invoices are overdue?", user=request.user)
result.text, result.output, result.steps, result.stopped   # plus token counts

result = await agents.arun(...)                 # async
async for event in agents.stream(...):          # token / message / update / end / error events
    ...
```

- Pass `thread=` (a `Thread`) or `thread_id=` to continue a conversation.
- Without either, the run is **throwaway**: it and its subagents keep their state in memory, never in the database, and it is freed when the run ends. The recorded calls and runs are kept. This also means a throwaway run works inside a test transaction (Django's `TestCase`).
- For threads your code owns (a WhatsApp number, a ticket), see [keyed threads](access.md#threads-your-code-keys).
- `retry=` sets how long this run's model calls wait out a short rate limit (see [rate limits and fallbacks](touchpoints.md#rate-limits-and-fallbacks)).

**Inside another agent's run** (from a tool, a code agent, or an orchestrator they call), `agents.run` **joins that request** instead of starting a new one:

- it acts for the same caller, unless you pass `user=` or `tenant=`;
- it shares the request's limits (depth and timeout; it doesn't use up a hand-off, because your code chose it, not the model);
- it runs on its own thread under the parent's, `<thread>/<agent>/<id>`;
- it is recorded as a subagent run in the request's tree, so the request's cost includes it;
- its messages stream to the client like a subagent's.

Pass `join=False` to start a separate request. `agents.stream` always starts one.

A touchpoint called there without a `config` (`tp.complete(...)`, or `tp.model().invoke(..., tp.run_config())`) joins the request too: the call is recorded in its run tree, counts towards its cost, and uses its `retry` policy. It doesn't use the request's model choice, which is for the agent's own touchpoint.

## The loop

model → tools → model, until the model answers or reaches `max_steps`. On its last step the model is told to answer. Each step reads the declaration and admin overrides again, so a change in the admin applies to runs already in progress.

## Tools

- Tools are always limited to what the **caller** may use. See [Model tools](tools.md) for permissions and scoping.
- The tool node refuses any call outside the agent's own set, so a parent's tools never reach a subagent.
- `tools.select(names=…, categories=…, kinds=…)` picks registered tools lazily, per caller.
- **Public tools:** a tool registered with `permission=tools.PUBLIC` is offered to anonymous visitors and to runs with no user (a Celery task). Use it only for tools that read public data. Tools generated from your models can't be public.
- **Loading tools on demand:** see [many tools](#many-tools) below.

### Many tools

Every bound tool's name, description and argument schema goes into every model call. With dozens of tools that crowds the context and makes the model choose worse. So an agent that can use more than `DEFER_TOOLS_ABOVE` tools (default 20) **defers** them:

- The model gets its `essential` tools, a `find_tools` tool, and a one-line overview of the rest by category, such as `34 more tools are available (billing (12), crm (9), reports (8), other (5))`.
- `find_tools(query, category=None)` searches the deferred tools and loads the best matches, which the model can call from its next step. The search covers names, categories, descriptions and argument names, and words that few tools share count most. With a `category` and no query, it loads that category's first tools. When nothing matches, the reply lists the categories again.
- Found tools stay loaded for the thread. Only the `MAX_LOADED_TOOLS` most recently found stay bound (default 15), so a long conversation doesn't build the context back up.

```python
agent("ops.assistant", ..., tools=tools.select(), essential=["whoami"])   # deferred once over the threshold
agent("crm.writer", ..., defer_tools=True)                                # always deferred
agent("crm.quick", ..., defer_tools=False)                                # never deferred
```

To search by meaning as well as by words, so that "money customers owe" finds `find_invoices`, point `TOOL_SEARCH_TOUCHPOINT` at an [embedding touchpoint](touchpoints.md#embeddings). Each tool is embedded once per process, and each search embeds the query. If the embedding call fails, the search falls back to words.

Deferral keeps the prompt small. It doesn't change which tools the agent may use.
- **Long tool results:** `tool_result_chars=12000` cuts each tool result to that many characters, with a note saying how much was left out, so one fetched page can't fill the context window.

## Handing work over

- An agent with `subagents` gets one `delegate(agent, task)` tool.
- The subagent runs as the same user, on its own checkpointed thread `<thread>/<agent>/<call id>`. It sees only the task, or the parent's conversation with `fork=True`.
- Its answer (or its `output` as JSON) is the tool result.
- Its messages stream to the client as `token`/`message` events with `agent` and `depth` set, so a UI can show them nested.

## Limits per request

| Setting | Default | Meaning |
|---|---|---|
| `AGENT_MAX_DEPTH` | `2` | how deep subagents may nest (`0`: no delegation) |
| `AGENT_MAX_DELEGATIONS` | `8` | hand-offs per request, across all its agents |
| `AGENT_TIMEOUT` | `300` | seconds after which a request's agents start no new model call |
| `AGENT_BUDGET` | `None` | USD a request may spend before its agents start no new model call |

No agent can hand work to itself or to an agent above it. The timeout is cooperative: it is checked before each model call and hand-off, and doesn't kill a call in progress.

## Budgets

A request keeps a running total of what its calls cost, its subagents' included, as each call ends. Before each step, an agent checks it against the request's budget. Once the budget is reached, the agent makes no more model calls and runs no more tools, and the run ends with `stopped="budget"`:

```python
result = agents.run("crm.researcher", task, user=user, budget=0.50)   # USD; default AGENT_BUDGET
if result.stopped == "budget":
    ...
```

- A call already under way finishes. A request can go over its budget by about one call.
- Prices come from the [catalogue](catalogue.md). A model with no known price adds nothing.
- The total needs recording on (`OBSERVABILITY`, the default). It doesn't need a sink, so `RECORD_SINKS: []` is fine.
- A run inside another agent's request shares that request's budget. Passing `budget=` there raises `ConfigError`.

Daily limits per user and per tenant come from recorded calls:

```python
LLM_KIT = {"DAILY_BUDGETS": {"user": 5, "tenant": 50}}   # USD since local midnight
```

A touchpoint can have a daily limit too, across every call made through it, whoever makes them:

```python
digest = touchpoint("crm.digest", default="anthropic/claude-haiku-5-5", daily_budget=20)
LLM_KIT = {"DAILY_BUDGETS": {"touchpoints": {"crm.digest": 10}}}   # sets or overrides it without a deploy
```

Once the touchpoint has reached its limit, `model()`, `complete()` and `embed()` raise `BudgetExceeded`, and an agent using it stops with `stopped="budget"`. The check runs before every call without querying the database each time. Each process reads the period's recorded spend, at most every 30 seconds, and adds what has been spent since:

- With a shared cache (Redis or Memcached as Django's default cache), every process adds each priced call to counters in the cache, so all workers see each other's spend at once. This is `SPEND_COUNTER = "auto"`, the default. Use `"cache"` to force it, or `None` to turn it off.
- With a per-process cache (Django's default `LocMemCache`), each process adds only its own calls, and sees other processes' spend at the next read.

Either way a limit can be passed by about one call: a call already under way finishes.

A request from a user or tenant that has reached its limit is refused before it starts. `agents.run` raises `BudgetExceeded`, the HTTP views answer 429, and the WebSocket consumer sends an `error` event. Calls are written in the background, so the day's total can miss the last few seconds of spend. `DAILY_BUDGETS` needs the `database` sink.

## History window

`history=2000` sends each model call only the newest ~2000 tokens of the conversation, starting at a user message. The checkpoint still keeps everything.

## Admin overrides

*Agent overrides* in the admin can append to or replace the prompt, lower `max_steps`, narrow the tools, or pause an agent. They narrow the declaration in code, never widen it. To change an agent's model, override its touchpoint.

## Recording

Every call and run carries its agent. A subagent's run links to its parent (`parent_run_id`, `depth`), and the request's run counts every call under it, so its cost includes its subagents. The admin shows the run tree, and [monitoring](monitoring.md) filters and breaks down by `agent`.

## Checked at startup

Unknown subagents, touchpoints or tool names, agents that delegate in a circle, and names that clash with `GRAPHS` raise at startup. `python manage.py llm_kit_agents [--user name]` lists agents, their model, tools, subagents and limits, and the tools a given user gets.
