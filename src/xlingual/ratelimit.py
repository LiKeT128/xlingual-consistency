"""A sliding-window rate limiter shared by every worker thread.

A fixed sleep between calls caps throughput at one request per interval even
when the provider would happily take several at once. This limiter counts the
calls made in the trailing sixty seconds instead, so workers run in parallel up
to the real per-minute budget and only block when that budget is actually spent.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    """Sliding-window limiter that learns the provider's real budget.

    A configured RPM is a guess. Set it too high and every excess call is
    rejected, retried and eventually dropped - which is slower than the correct
    rate and loses data as well. So a 429 halves the working budget and a run of
    clean responses walks it back up. The tool converges on the true limit
    instead of requiring the user to know it in advance.
    """

    RECOVERY_STREAK = 20

    def __init__(self, requests_per_minute: int, window: float = 60.0,
                 min_capacity: int = 2) -> None:
        self.max_capacity = max(int(requests_per_minute), 1)
        self.min_capacity = min(min_capacity, self.max_capacity)
        self.capacity = self.max_capacity
        self.window = window
        self.throttled = False
        self._calls: deque[float] = deque()
        self._streak = 0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until another call fits inside the trailing window."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._evict(now)
                if len(self._calls) < self.capacity:
                    self._calls.append(now)
                    return
                wait = self._calls[0] + self.window - now
            time.sleep(max(wait, 0.01))

    def penalize(self) -> int:
        """The provider said no. Halve the budget and report the new one."""
        with self._lock:
            self.capacity = max(self.min_capacity, self.capacity // 2)
            self.throttled = True
            self._streak = 0
            return self.capacity

    def reward(self) -> None:
        """A clean response. Creep back towards the configured budget."""
        with self._lock:
            if self.capacity >= self.max_capacity:
                return
            self._streak += 1
            if self._streak >= self.RECOVERY_STREAK:
                self.capacity += 1
                self._streak = 0

    def _evict(self, now: float) -> None:
        cutoff = now - self.window
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()


def estimate_seconds(requests: int, requests_per_minute: int, latency: float = 2.0,
                     workers: int = 1) -> float:
    """Wall time for a batch: whichever of the two ceilings binds first.

    Either the per-minute budget limits you, or the workers do. Concurrency
    cannot beat a provider's RPM cap - it only removes the idle waiting that a
    serial client adds on top of it.
    """
    by_quota = requests / max(requests_per_minute, 1) * 60.0
    by_workers = requests * latency / max(workers, 1)
    return max(by_quota, by_workers)


def format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} sec"
    return f"{seconds / 60:.0f} min"
