from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from llm_kit.tools import registry


def _perm(p):
    return getattr(p, "__name__", p) if callable(p) else p


class Command(BaseCommand):
    help = "List agent tools (generated from models and hand-written), their scope and permissions."

    def add_arguments(self, parser):
        parser.add_argument("--user", help="Only the tools this username may use (tenant from TENANT_RESOLVER).")

    def handle(self, *args, user=None, **options):
        # Declarations are checked when the app starts; reaching here means they all passed.
        if user:
            try:
                account = get_user_model()._default_manager.get_by_natural_key(user)
            except get_user_model().DoesNotExist as e:
                raise CommandError(f"No user {user!r}") from e
            entries = registry.entries_for(account)
            self.stdout.write(f"{len(entries)} tool(s) for {user}:")
        else:
            entries = registry.catalogue()
        if not entries:
            self.stdout.write(
                "No agent tools. Add `class AgentAccess:` to a model, or expose()/register() in <app>/agent_tools.py."
            )
        for e in entries:
            self.stdout.write(self.style.MIGRATE_HEADING(e.name) + f"  [{e.category}, {e.kind}]")
            if e.spec is not None:
                self.stdout.write(f"  model: {e.spec.label}   scope: {e.spec.scope_text()}")
                if e.kind == "save":
                    approval = f", approval for {e.spec.approval_fields}" if e.spec.approval_fields else ""
                    self.stdout.write(f"  writes: {e.spec.writes} {e.spec.saved_fields}{approval}")
            self.stdout.write(f"  permission: {' or '.join(str(_perm(p)) for p in e.permissions)}")
        paused = [s for s in registry.specs().values() if s.paused]
        for s in paused:
            self.stdout.write(self.style.WARNING(f"paused: {s.label}: {s.paused}"))
