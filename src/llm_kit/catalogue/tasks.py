"""Background jobs. Schedule the catalogue sync with whatever worker you use.

Celery Beat::

    CELERY_BEAT_SCHEDULE = {
        "llm-kit-catalogue": {"task": "llm_kit.sync_catalogue", "schedule": 6 * 60 * 60},
    }

Django-Q2::

    from django_q.tasks import schedule
    schedule("llm_kit.tasks.run_catalogue_sync", schedule_type="I", minutes=360)

Cron::

    0 */6 * * *  python manage.py llm_kit_sync_catalogue
"""

from __future__ import annotations


def run_catalogue_sync(force: bool = False) -> str:
    from llm_kit import catalogue

    return catalogue.sync(force=force).summary()


try:  # Celery is optional
    from celery import shared_task
except ImportError:  # pragma: no cover
    pass
else:
    sync_catalogue = shared_task(name="llm_kit.sync_catalogue", ignore_result=False)(run_catalogue_sync)
