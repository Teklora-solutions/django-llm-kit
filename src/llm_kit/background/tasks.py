"""Worker tasks for background runs (``RUN_EXECUTOR="celery"`` or ``"django_q"``)."""

from __future__ import annotations

__all__ = ["run_in_background", "run_in_background_task"]


def run_in_background(run_id: str) -> str:
    """Run one queued background run to its end."""
    from llm_kit.background.service import execute

    execute(str(run_id))
    return str(run_id)


try:  # Celery is optional
    from celery import shared_task
except ImportError:  # pragma: no cover
    run_in_background_task = None
else:
    run_in_background_task = shared_task(name="llm_kit.run_in_background", ignore_result=True)(run_in_background)
