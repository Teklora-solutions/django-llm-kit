"""Signals for background runs."""

from django.dispatch import Signal

run_finished = Signal()
"""Sent once when a background run reaches a final status: ``done``, ``interrupted``, ``error``,
``cancelled`` (also a waiting or queued run cancelled before it started) or ``lost``.

Arguments: ``sender`` (``BackgroundRun``), ``run``, ``status``, ``result`` (the ``end`` event's
data: ``next``, ``interrupts``; ``None`` unless done or interrupted) and ``error``. It is sent by
whichever process notices: the one that ran it, the one that cancelled it, or the one that found
it lost, and only by the one whose change took effect, so exactly once per run. It is sent after
the change is committed. A receiver that raises is logged and doesn't stop the others::

    from django.dispatch import receiver
    from llm_kit.background import run_finished

    @receiver(run_finished)
    def tell_them(sender, run, status, result, error, **kwargs):
        notify(run.user, f"Your task is {status}.")
"""
