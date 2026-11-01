from django.core.management.base import BaseCommand

from llm_kit import catalogue


class Command(BaseCommand):
    help = "Refresh the model catalogue from models.dev, OpenRouter and LiteLLM (conditional requests)."

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="Ignore ETags and re-parse every source.")
        parser.add_argument("--offline", action="store_true", help="Load the bundled snapshot; no network.")
        parser.add_argument("--source", action="append", dest="sources", help="Only these sources (repeatable).")

    def handle(self, *args, force=False, offline=False, sources=None, **options):
        selected = None
        if sources:
            by_name = {s.name: s for s in catalogue.configured_sources()}
            unknown = [s for s in sources if s not in by_name]
            if unknown:
                self.stderr.write(f"Unknown source(s): {', '.join(unknown)}. Known: {', '.join(by_name)}")
                return
            selected = [by_name[s] for s in sources]
        report = catalogue.sync(force=force, offline=offline, sources=selected)
        self.stdout.write(report.summary())
        notable = [c for c in report.changes if c.kind in ("removed", "deprecated", "price_changed")]
        for c in notable[:20]:
            detail = f" {c.field}: {c.old} -> {c.new}" if c.field else ""
            self.stdout.write(f"  {c.kind:<14} {c.provider}/{c.model_id}{detail}")
        if len(notable) > 20:
            self.stdout.write(f"  … and {len(notable) - 20} more (see admin: Catalogue changes)")
        if any(s.last_status == "failed" for s in report.sources.values()):
            self.stderr.write(self.style.WARNING("Some sources failed; their previous data was kept."))
