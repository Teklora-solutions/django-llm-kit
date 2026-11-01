from django.core.management.base import BaseCommand

from llm_kit.monitoring.alerts import check_all


class Command(BaseCommand):
    help = "Evaluate enabled alert rules now and deliver the ones that fire."

    def handle(self, *args, **options):
        fired = check_all()
        for e in fired:
            self.stdout.write(self.style.WARNING(f"FIRED {e.rule.name}: {e.message}"))
        self.stdout.write(f"{len(fired)} alert(s) fired.")
