"""run_finished: sent once for every background run that reaches a final status."""

from datetime import timedelta

import pytest
from django.utils import timezone

from llm_kit import background
from llm_kit.background import executors, run_finished
from llm_kit.models import BackgroundRun, Thread
from tests.graphs import GATE
from tests.test_background import finish, wait_for

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture(autouse=True)
def gate():
    GATE.clear()
    yield GATE
    GATE.set()


@pytest.fixture
def heard():
    got = []

    def receiver(sender, run, status, result, error, **kwargs):
        assert sender is BackgroundRun
        got.append((run.pk, status, result, error))

    run_finished.connect(receiver)
    yield got
    run_finished.disconnect(receiver)


@pytest.fixture
def no_worker(settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "RUN_EXECUTOR": "tests.test_background:noop"}


def test_sent_once_when_a_run_is_done(user, heard):
    thread = Thread.objects.create(graph="chat", user=user)
    run = finish(background.start(thread, {"message": "hi"}, user=user))
    assert heard == [(run.pk, "done", {"next": [], "interrupts": []}, "")]


def test_sent_for_interrupted_and_error(user, heard):
    asking = finish(
        background.start(
            Thread.objects.create(graph="approval", user=user), {"input": {"request": "x", "log": []}}, user=user
        )
    )
    failing = finish(background.start(Thread.objects.create(graph="boom", user=user), {"message": "x"}, user=user))
    assert [(pk, status) for pk, status, *_ in heard] == [(asking.pk, "interrupted"), (failing.pk, "error")]
    assert heard[0][2]["interrupts"] and heard[1][3]


def test_sent_for_cancelled_runs_started_or_not(user, heard):
    thread = Thread.objects.create(graph="slow", user=user)
    running = wait_for(background.start(thread, {"message": "one"}, user=user), "running")
    waiting = background.start(thread, {"message": "two"}, user=user, queue=True)
    assert waiting.cancel()
    assert running.cancel()
    finish(running)
    assert sorted(status for _, status, *_ in heard) == ["cancelled", "cancelled"]
    assert {pk for pk, *_ in heard} == {running.pk, waiting.pk}


def test_a_lost_run_is_announced_once_wherever_it_is_noticed(user, heard, no_worker):
    thread = Thread.objects.create(graph="chat", user=user)
    run = background.start(thread, {"message": "x"}, user=user)
    stale = timezone.now() - timedelta(minutes=5)
    BackgroundRun.objects.filter(pk=run.pk).update(status="running", heartbeat_at=stale)
    one, two = BackgroundRun.objects.get(pk=run.pk), BackgroundRun.objects.get(pk=run.pk)  # two processes
    background.refresh(one)
    background.refresh(two)
    background.cancel(BackgroundRun.objects.get(pk=run.pk))  # too late: it's over
    assert heard == [(run.pk, "lost", None, "")]


def test_a_run_that_finishes_after_being_marked_lost_is_not_announced_again(user, heard, gate):
    thread = Thread.objects.create(graph="gated", user=user)
    run = wait_for(background.start(thread, {"message": "x"}, user=user), "running")
    BackgroundRun.objects.filter(pk=run.pk).update(heartbeat_at=timezone.now() - timedelta(minutes=5))
    assert background.refresh(run).status == "lost"
    gate.set()
    assert executors.join(run.run_id, 10)
    run.refresh_from_db()
    assert run.status == "lost" and heard == [(run.pk, "lost", None, "")]


def _finished(run, timeout=10):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run.refresh_from_db()
        if not run.active:
            return run
        finish(run, timeout=1) if run.status == "running" else time.sleep(0.02)
    raise AssertionError(f"run is still {run.status}")


def test_a_failing_receiver_does_not_stop_the_others_or_the_queue(user, heard, gate, caplog):
    def broken(**kwargs):
        raise RuntimeError("push service down")

    run_finished.connect(broken)
    try:
        thread = Thread.objects.create(graph="gated", user=user)
        first = wait_for(background.start(thread, {"message": "one"}, user=user), "running")
        second = background.start(thread, {"message": "two"}, user=user, queue=True)
        gate.set()
        finish(first)
        second = _finished(second)
    finally:
        run_finished.disconnect(broken)
    assert second.status == "done" and [s for _, s, *_ in heard] == ["done", "done"]
    assert "push service down" in caplog.text
