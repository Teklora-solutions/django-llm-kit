"""Signals of background media jobs."""

from django.dispatch import Signal

# Sent once when a background media job ends (done, error, cancelled or lost), from whichever
# process noticed, after ``run_finished``. Arguments: ``run`` (the BackgroundRun), ``status``,
# ``result`` (saved media's names and URLs, text, units, cost, model; None when it failed) and
# ``error``. Use it to tell your users (a push, an email) or call your own webhooks.
media_job_finished = Signal()
