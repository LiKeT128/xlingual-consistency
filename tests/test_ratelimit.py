import threading
import time

from xlingual.ratelimit import RateLimiter, estimate_seconds, format_duration


def test_calls_within_budget_do_not_block():
    limiter = RateLimiter(requests_per_minute=10)
    start = time.monotonic()
    for _ in range(10):
        limiter.acquire()
    assert time.monotonic() - start < 0.2


def test_the_budget_blocks_once_it_is_spent():
    # Four calls into a two-per-window budget: the last two must wait for the
    # window to roll, which is the behaviour that keeps a free tier from 429ing.
    limiter = RateLimiter(requests_per_minute=2, window=0.4)
    start = time.monotonic()
    for _ in range(4):
        limiter.acquire()
    assert time.monotonic() - start >= 0.4


def test_the_budget_is_shared_across_threads():
    limiter = RateLimiter(requests_per_minute=4, window=0.5)
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        limiter.acquire()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    start = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # Eight calls against a budget of four per window need a second window.
    assert time.monotonic() - start >= 0.5


def test_estimate_is_bounded_by_the_quota_not_the_workers():
    # 200 requests at 15 rpm cannot finish faster than 800 s no matter how many
    # workers are thrown at it - concurrency does not beat a provider's cap.
    assert estimate_seconds(200, 15, latency=2.0, workers=32) == 800.0


def test_estimate_is_bounded_by_the_workers_when_the_quota_is_generous():
    assert estimate_seconds(200, 6000, latency=2.0, workers=4) == 100.0


def test_format_duration():
    assert format_duration(45) == "45 sec"
    assert format_duration(800) == "13 min"
