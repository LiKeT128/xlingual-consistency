"""A small OpenAI-compatible chat client built for free tiers.

Free tiers rate-limit hard and fail often, so the client paces itself to a
requests-per-minute budget and retries 429 and 5xx with exponential backoff,
honouring Retry-After when the server sends one.
"""

from __future__ import annotations

import random
import time

import requests

from .config import Config


class ProviderError(RuntimeError):
    """Raised when the provider refuses the request in a way retrying cannot fix."""


class ChatClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._session = requests.Session()
        self._min_interval = 60.0 / max(config.requests_per_minute, 1)
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

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
        for attempt in range(self.config.max_retries):
            self._throttle()
            try:
                response = self._session.post(
                    url, json=payload, headers=headers, timeout=self.config.timeout
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                self._backoff(attempt)
                continue

            if response.status_code == 200:
                return self._extract(response.json())

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(min(float(retry_after), 120))
                        continue
                    except ValueError:
                        pass
                self._backoff(attempt)
                continue

            raise ProviderError(f"HTTP {response.status_code}: {response.text[:400]}")

        raise ProviderError(f"gave up after {self.config.max_retries} attempts: {last_error}")

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(2**attempt + random.random(), 60))

    @staticmethod
    def _extract(body: dict) -> str:
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {body}") from exc
