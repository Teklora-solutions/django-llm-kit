from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from llm_kit import conf
from llm_kit.models import AlertEvent, CatalogueChange, LLMCall, LLMRun, Thread, ToolCall


class Command(BaseCommand):
    help = "Delete old calls, runs, tool calls, captured content, catalogue changes and alerts (threads: opt-in)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--calls", type=int, help="Days to keep calls/runs/tool calls (default CALL_RETENTION_DAYS)."
        )
        parser.add_argument(
            "--content", type=int, help="Days to keep captured prompts (default CONTENT_RETENTION_DAYS)."
        )
        parser.add_argument("--changes", type=int, default=180, help="Days to keep catalogue changes and alerts.")
        parser.add_argument(
            "--threads",
            type=int,
            help="Also delete threads (and their graph state) idle for this many days. Off unless given.",
        )
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, calls=None, content=None, changes=180, threads=None, dry_run=False, **options):
        now = timezone.now()
        calls = calls if calls is not None else conf.get("CALL_RETENTION_DAYS")
        content = content if content is not None else conf.get("CONTENT_RETENTION_DAYS")
        plan = [
            ("calls", LLMCall.objects.filter(started_at__lt=now - timedelta(days=calls))),
            ("runs", LLMRun.objects.filter(started_at__lt=now - timedelta(days=calls))),
            ("tool calls", ToolCall.objects.filter(started_at__lt=now - timedelta(days=calls))),
            ("catalogue changes", CatalogueChange.objects.filter(synced_at__lt=now - timedelta(days=changes))),
            ("alerts", AlertEvent.objects.filter(fired_at__lt=now - timedelta(days=changes))),
        ]
        if threads is not None:
            plan.append(("threads", Thread.objects.filter(updated_at__lt=now - timedelta(days=threads))))
        content_qs = LLMCall.objects.filter(started_at__lt=now - timedelta(days=content), prompt__isnull=False)
        verb = "would delete" if dry_run else "deleted"
        for label, qs in plan:
            n = qs.count()
            if not dry_run and n:
                if label == "threads":
                    for t in qs.iterator():  # one by one so their checkpoints are deleted too
                        t.delete()
                else:
                    qs.delete()
            self.stdout.write(f"{verb} {n} {label}")
        n = content_qs.count()
        if not dry_run and n:
            content_qs.update(prompt=None, completion=None)
        self.stdout.write(f"{'would clear' if dry_run else 'cleared'} captured content on {n} calls")
