from django.core.management.base import BaseCommand, CommandError

from llm_kit.core.touchpoints import ClientChoice, all_touchpoints, problems
from llm_kit.touchpoints.env import DjangoEnv


class Command(BaseCommand):
    help = "List touchpoints declared in code, how they resolve, and configuration problems."

    def add_arguments(self, parser):
        parser.add_argument("--check", action="store_true", help="Exit with an error if any touchpoint has problems.")

    def handle(self, *args, check=False, **options):
        env = DjangoEnv()
        tps = all_touchpoints()
        if not tps:
            self.stdout.write("No touchpoints registered. Declare them in <app>/touchpoints.py.")
            return
        total = 0
        for tp in tps:
            choice = "client" if isinstance(tp.choice, ClientChoice) else "server"
            self.stdout.write(self.style.MIGRATE_HEADING(f"{tp.name}") + f"  ({choice} chooses)")
            try:
                r = tp.resolve(env=env)
                self.stdout.write(f"  resolves to: {' → '.join(r.candidates)}  [{r.chosen_by}]")
                for note in r.notes:
                    self.stdout.write(f"  note: {note}")
                for model, why in r.skipped:
                    self.stdout.write(f"  skipped {model}: {why}")
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  cannot resolve: {e}"))
                total += 1
            for p in problems(tp, env):
                self.stdout.write(self.style.WARNING(f"  problem: {p}"))
                total += 1
        if check and total:
            raise CommandError(f"{total} touchpoint problem(s).")
