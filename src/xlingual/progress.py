"""A status line that ticks on a timer, not on completion.

Printing progress only when a request finishes leaves the screen frozen while
workers sit in rate-limit backoff - exactly the moment the user most needs to
know the tool is alive. This refreshes on its own clock and shows what the
workers are doing, including waiting.
"""

from __future__ import annotations

import sys
import threading
import time


def format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def render(done: int, total: int, failed: int, in_flight: int, waiting: int,
           rate: int | None, elapsed: float) -> str:
    """Build the status line. Pure, so its formatting is testable."""
    parts = [f"{done}/{total}"]

    pct = (done / total * 100) if total else 0.0
    parts.append(f"{pct:3.0f}%")

    if failed:
        parts.append(f"failed {failed}")

    activity = f"{in_flight} in flight"
    if waiting:
        activity += f", {waiting} waiting on rate limit"
    parts.append(activity)

    if rate is not None:
        parts.append(f"{rate} req/min")

    parts.append(format_elapsed(elapsed))

    if done and elapsed > 0:
        remaining = (total - done) * (elapsed / done)
        parts.append(f"~{format_elapsed(remaining)} left")
    else:
        parts.append("estimating")

    return "  " + "  |  ".join(parts)


class Progress:
    """Live status line driven by a background thread."""

    def __init__(self, total: int, limiter=None, interval: float = 0.5) -> None:
        self.total = total
        self.limiter = limiter
        self.interval = interval
        self.done = 0
        self.failed = 0
        self.in_flight = 0
        self.waiting = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = time.monotonic()
        self._width = 0

    def __enter__(self) -> "Progress":
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._draw()
        sys.stdout.write("\n")
        sys.stdout.flush()

    def task_started(self) -> None:
        with self._lock:
            self.in_flight += 1

    def task_waiting(self, waiting: bool) -> None:
        """Flag a worker as parked in rate-limit backoff rather than working."""
        with self._lock:
            self.waiting += 1 if waiting else -1
            self.waiting = max(self.waiting, 0)

    def task_finished(self, ok: bool) -> None:
        with self._lock:
            self.in_flight = max(self.in_flight - 1, 0)
            self.done += 1
            if not ok:
                self.failed += 1

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._draw()

    def _draw(self) -> None:
        with self._lock:
            rate = self.limiter.capacity if self.limiter is not None else None
            line = render(
                self.done, self.total, self.failed, self.in_flight,
                self.waiting, rate, time.monotonic() - self._started,
            )
        padding = " " * max(self._width - len(line), 0)
        self._width = max(self._width, len(line))
        sys.stdout.write("\r" + line + padding)
        sys.stdout.flush()
