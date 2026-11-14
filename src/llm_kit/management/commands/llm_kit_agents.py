from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from llm_kit.agents.overrides import override_for
from llm_kit.agents.registry import check, declared_tool_names
from llm_kit.core.agents import all_agents
from llm_kit.core.agents.delegate import effective
from llm_kit.core.agents.env import default_env


class Command(BaseCommand):
    help = "List agents: model, tools, subagents, limits and admin overrides. With --user, the tools that user gets."

    def add_arguments(self, parser):
        parser.add_argument("--user", help="Show the tools each agent has when this username runs it.")

    def handle(self, *args, user=None, **options):
        account = None
        if user:
            try:
                account = get_user_model()._default_manager.get_by_natural_key(user)
            except get_user_model().DoesNotExist as e:
                raise CommandError(f"No user {user!r}") from e
        specs = all_agents()
        if not specs:
            self.stdout.write('No agents. Declare them in <app>/agents.py with llm_kit.agent("app.name", ...).')
        env = default_env()
        for declared in specs:
            spec = effective(declared, env)
            kind = "code" if spec.is_code else f"touchpoint {spec.touchpoint}"
            self.stdout.write(self.style.MIGRATE_HEADING(spec.name) + f"  [{kind}]")
            self.stdout.write(f"  {spec.description}")
            if spec.paused:
                self.stdout.write(self.style.WARNING(f"  paused: {spec.paused}"))
            if not spec.is_code:
                limits = f"  max steps: {spec.max_steps}"
                if spec.output is not None:
                    limits += f"   output: {spec.output.__name__}"
                if spec.fork:
                    limits += "   sees the parent's conversation"
                self.stdout.write(limits)
                self.stdout.write(f"  tools: {self._tools(spec, account)}")
            if spec.subagents:
                self.stdout.write(f"  subagents: {', '.join(spec.subagents)}")
            override = override_for(spec.name)
            if override is not None:
                changed = [
                    name
                    for name, on in (
                        ("prompt", bool(override.prompt.strip())),
                        ("max_steps", bool(override.max_steps)),
                        ("tools", override.tools is not None),
                        ("paused", bool(override.paused)),
                    )
                    if on
                ]
                self.stdout.write(f"  admin override: {', '.join(changed) or 'none'}")
        for problem in check():
            self.stdout.write(self.style.ERROR(f"problem: {problem}"))

    def _tools(self, spec, account) -> str:
        from llm_kit.core.agents.harness import resolve_tools
        from llm_kit.tools.registry import ToolSelection

        if account is not None:
            from llm_kit.tools.caller import CONFIG_KEY, caller_for

            config = {"configurable": {CONFIG_KEY: caller_for(account)}}
            names = [t.name for t in resolve_tools(spec, config, default_env())]
            text = ", ".join(names) or "none"
        elif isinstance(spec.tools, (list, tuple, ToolSelection)):
            text = ", ".join(declared_tool_names(spec)) or "none"
        else:
            text = "chosen per run" if spec.tools else "none"
        if spec.defer_tools:
            text += f"  (loaded on demand; from the start: {', '.join(spec.essential) or 'none'})"
        elif spec.defer_tools is None:
            text += "  (loaded on demand when there are many)"
        return text
