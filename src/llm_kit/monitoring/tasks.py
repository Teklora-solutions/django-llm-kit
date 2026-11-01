"""Schedule alert checks every few minutes (Celery Beat: ``llm_kit.check_alerts``; Django-Q2/cron: the function)."""

from __future__ import annotations


def run_alert_checks() -> str:
    from llm_kit.monitoring.alerts import check_all

    fired = check_all()
    return f"{len(fired)} alert(s) fired"


try:
    from celery import shared_task
except ImportError:  # pragma: no cover
    pass
else:
    check_alerts = shared_task(name="llm_kit.check_alerts")(run_alert_checks)
