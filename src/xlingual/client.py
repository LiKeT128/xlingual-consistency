"""A small OpenAI-compatible chat client built for free tiers.

Free tiers rate-limit hard and fail often, so the client paces itself to a
requests-per-minute budget and retries 429 and 5xx with exponential backoff,
honouring Retry-After when the server sends one.
"""

from __future__ import annotations

import random
import re
import threading
import time
from typing import Callable

import requests

from .config import Config
from .ratelimit import RateLimiter

# Google returns the wait as "retryDelay": "27s" inside the error body.
_RETRY_DELAY = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


class ProviderError(RuntimeError):
    """Raised when the provider refuses the request in a way retrying cannot fix."""


class ChatClient:
    """Thread-safe: several workers may share one instance.

    Each thread gets its own requests.Session, and all of them draw from a
    single RateLimiter so the per-minute budget is respected across the pool
    rather than per worker.
    """

    # Waiting out a rate limit is the correct response to it, so 429 gets its
    # own generous allowance rather than sharing the one for real failures.
    RATE_LIMIT_RETRIES = 12

    def __init__(self, config: Config) -> None:
        self.config = config
        self._limiter = RateLimiter(config.requests_per_minute)
        self._local = threading.local()
        # Set by the runner so the status line can show workers parked in
        # backoff instead of looking frozen.
        self.on_wait: Callable[[bool], None] | None = None

    @property
    def limiter(self) -> RateLimiter:
        return self._limiter

    @property
    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def _throttle(self) -> None:
        self._limiter.acquire()

    def complete(self, prompt: str, *, model: str | None = None) -> str:
        """Send one user message and return the assistant's text."""
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.config.base_url}/chat/completions"

        last_error = ""
        errors = 0        # 5xx and network faults - may genuinely not recover
        throttles = 0     # 429 - not a failure, just "later"
        while errors < self.config.max_retries and throttles < self.RATE_LIMIT_RETRIES:
            self._throttle()
            try:
                response = self._session.post(
                    url, json=payload, headers=headers, timeout=self.config.timeout
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                time.sleep(min(2**errors + random.random(), 60.0))
                errors += 1
                continue

            if response.status_code == 200:
                self._limiter.reward()
                return self._extract(response.json())

            if response.status_code == 429:
                # A 429 means our budget is wrong, not that the request is bad.
                # Shrink the budget and keep waiting, instead of spending the
                # retry allowance that real failures need.
                last_error = f"HTTP 429: {response.text[:200]}"
                self._limiter.penalize()
                self._sleep_visibly(self._retry_delay(response, throttles))
                throttles += 1
                continue

            if response.status_code in (500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                time.sleep(self._retry_delay(response, errors))
                errors += 1
                continue

            raise ProviderError(f"HTTP {response.status_code}: {response.text[:400]}")

        raise ProviderError(
            f"gave up after {errors} errors and {throttles} rate limits: {last_error}"
        )

    def _sleep_visibly(self, seconds: float) -> None:
        """Sleep, but let the caller show that this worker is waiting, not stuck."""
        if self.on_wait is None:
            time.sleep(seconds)
            return
        self.on_wait(True)
        try:
            time.sleep(seconds)
        finally:
            self.on_wait(False)

    def _retry_delay(self, response: requests.Response, attempt: int) -> float:
        """Prefer the wait the server asked for, in whichever form it sent it."""
        header = response.headers.get("Retry-After")
        if header:
            try:
                return min(float(header), 120.0)
            except ValueError:
                pass

        # Google puts the wait inside the error body rather than in a header.
        match = _RETRY_DELAY.search(response.text or "")
        if match:
            try:
                return min(float(match.group(1)), 120.0)
            except ValueError:
                pass

        return min(2**attempt + random.random(), 60.0)

    def list_models(self) -> list[str]:
        """Ask the provider which model ids this key can actually call.

        Provider docs go stale and free tiers expose a different set from the
        paid ones, so the endpoint is the only reliable answer to "what can I
        run".
        """
        response = self._session.get(
            f"{self.config.base_url}/models",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=self.config.timeout,
        )
        if response.status_code != 200:
            raise ProviderError(f"HTTP {response.status_code}: {response.text[:400]}")

        body = response.json()
        entries = body.get("data", body.get("models", []))
        ids = []
        for entry in entries:
            name = entry.get("id") or entry.get("name") or ""
            ids.append(name.rsplit("/", 1)[-1] if name else "")
        return sorted(i for i in ids if i)

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(2**attempt + random.random(), 60))

    @staticmethod
    def _extract(body: dict) -> str:
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {body}") from exc
