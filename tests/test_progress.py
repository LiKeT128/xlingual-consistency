import time

from xlingual.progress import Progress, format_elapsed, render


def test_format_elapsed():
    assert format_elapsed(9) == "9s"
    assert format_elapsed(75) == "1m15s"
    assert format_elapsed(600) == "10m00s"


def test_the_line_shows_activity_before_anything_completes():
    # The frozen-screen case: nothing done yet, but the user must still see
    # that workers are alive.
    line = render(done=0, total=200, failed=0, in_flight=6, waiting=0, rate=15, elapsed=3.0)
    assert "0/200" in line
    assert "6 in flight" in line
    assert "15 req/min" in line
    assert "estimating" in line


def test_waiting_workers_are_named_as_waiting_not_missing():
    line = render(done=10, total=200, failed=0, in_flight=6, waiting=4, rate=8, elapsed=30.0)
    assert "4 waiting on rate limit" in line


def test_failures_appear_only_when_there_are_some():
    clean = render(done=10, total=20, failed=0, in_flight=2, waiting=0, rate=None, elapsed=5.0)
    assert "failed" not in clean
    dirty = render(done=10, total=20, failed=3, in_flight=2, waiting=0, rate=None, elapsed=5.0)
    assert "failed 3" in dirty


def test_eta_is_extrapolated_from_observed_pace():
    line = render(done=50, total=200, failed=0, in_flight=6, waiting=0, rate=15, elapsed=60.0)
    assert "~3m00s left" in line


def test_counters_track_the_pool(capsys):
    with Progress(total=3, interval=10) as progress:  # slow tick: no redraws
        progress.task_started()
        progress.task_started()
        assert progress.in_flight == 2
        progress.task_waiting(True)
        assert progress.waiting == 1
        progress.task_waiting(False)
        assert progress.waiting == 0
        progress.task_finished(ok=True)
        progress.task_finished(ok=False)
        assert progress.done == 2
        assert progress.failed == 1
        assert progress.in_flight == 0
    assert "2/3" in capsys.readouterr().out


def test_the_ticker_redraws_without_any_completions(capsys):
    with Progress(total=5, interval=0.05):
        time.sleep(0.2)
    out = capsys.readouterr().out
    # Several carriage-returned redraws, none of them triggered by a result.
    assert out.count("\r") >= 2
