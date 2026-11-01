from django.core.management.base import BaseCommand, CommandError

from llm_kit.monitoring.health import report


class Command(BaseCommand):
    help = "Health summary: catalogue freshness, circuits, touchpoint problems, last 24 h. Exits 1 when unhealthy."

    def add_arguments(self, parser):
        parser.add_argument("--no-fail", action="store_true", help="Always exit 0 (report only).")

    def handle(self, *args, no_fail=False, **options):
        r = report()
        w = self.stdout.write
        w(self.style.MIGRATE_HEADING("Catalogue"))
        w(f"  last synced: {r['catalogue_last_synced'] or 'never'}")
        d = r["last_24h"]
        w(self.style.MIGRATE_HEADING("Last 24 h"))
        w(f"  calls {d['calls']}, errors {d['errors']} ({d['error_rate']:.1%}), fallbacks {d['fallback_rate']:.1%}")
        w(f"  cost ${d['cost'].normalize():f}, unpriced calls {d['unpriced']}")
        lat = r["latency_ms_24h"]
        w(f"  latency p50 {lat['p50']} ms, p95 {lat['p95']} ms")
        w(self.style.MIGRATE_HEADING("Providers"))
        w("  all circuits closed" if not r["open_circuits"] else "")
        for p, s in r["open_circuits"].items():
            w(self.style.ERROR(f"  {p}: OPEN ({s['reason']}), retry in {s['retry_in_s']} s"))
        if r["alerts_24h"]:
            w(self.style.MIGRATE_HEADING("Alerts (24 h)"))
            for a in r["alerts_24h"]:
                w(self.style.WARNING(f"  {a}"))
        if r["healthy"]:
            w(self.style.SUCCESS("Healthy."))
            return
        w(self.style.MIGRATE_HEADING("Issues"))
        for issue in r["issues"]:
            w(self.style.ERROR(f"  {issue}"))
        if not no_fail:
            raise CommandError(f"{len(r['issues'])} issue(s).")
