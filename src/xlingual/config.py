"""Provider configuration.

Every provider below speaks the OpenAI chat-completions protocol, so one HTTP
client covers all of them. Pick a provider with XLC_PROVIDER, drop the key in
XLC_API_KEY, and the base URL is filled in for you.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROVIDERS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openrouter": "https://openrouter.ai/api/v1",
    "mistral": "https://api.mistral.ai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "github": "https://models.github.ai/inference",
    "openai": "https://api.openai.com/v1",
    "custom": "",
    # Offline: answers are generated locally, no key and no network. Lets anyone
    # run the whole pipeline end to end before spending a single request.
    "mock": "",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "prompts.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader, so the project has no dependency just for this."""
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def env_str(name: str, default: str = "") -> str:
    """Treat an empty value as absent.

    'XLC_RPM=' in a .env file sets the variable to an empty string, so
    os.environ.get(name, default) returns '' rather than the default. Every
    numeric setting has to survive that, because deleting a value is exactly
    what someone does when they mean 'use the default'.
    """
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def env_int(name: str, default: int) -> int:
    raw = env_str(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(
            f"{name} must be a whole number, but .env has {name}={raw!r}. "
            f"Remove the line to use the default ({default})."
        ) from None


def env_float(name: str, default: float) -> float:
    raw = env_str(name, str(default))
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(
            f"{name} must be a number, but .env has {name}={raw!r}. "
            f"Remove the line to use the default ({default})."
        ) from None


@dataclass(frozen=True)
class Config:
    provider: str
    base_url: str
    api_key: str
    model: str
    judge_model: str
    temperature: float
    max_tokens: int
    requests_per_minute: int
    concurrency: int
    max_retries: int
    timeout: int
    system_prompt: str

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        provider = env_str("XLC_PROVIDER", "groq").lower()
        if provider not in PROVIDERS:
            raise SystemExit(
                f"Unknown XLC_PROVIDER '{provider}'. "
                f"Choose one of: {', '.join(sorted(PROVIDERS))}"
            )
        if provider == "mock":
            # No key, no endpoint, no quota - the point is to need nothing.
            base_url, api_key = "mock://local", "mock"
            model = env_str("XLC_MODEL", "mock")
        else:
            base_url = env_str("XLC_BASE_URL") or PROVIDERS[provider]
            if not base_url:
                raise SystemExit("XLC_PROVIDER=custom requires XLC_BASE_URL to be set.")
            api_key = env_str("XLC_API_KEY")
            if not api_key:
                raise SystemExit(
                    "XLC_API_KEY is not set. Copy .env.example to .env and fill it in."
                )
            model = env_str("XLC_MODEL")
            if not model:
                raise SystemExit(
                    "XLC_MODEL is not set (for example: llama-3.3-70b-versatile)."
                )
        return cls(
            provider=provider,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model,
            judge_model=env_str("XLC_JUDGE_MODEL") or model,
            temperature=env_float("XLC_TEMPERATURE", 0.0),
            max_tokens=env_int("XLC_MAX_TOKENS", 512),
            requests_per_minute=max(1, env_int("XLC_RPM", 15)),
            concurrency=max(1, env_int("XLC_CONCURRENCY", 6)),
            max_retries=max(1, env_int("XLC_MAX_RETRIES", 5)),
            timeout=max(1, env_int("XLC_TIMEOUT", 90)),
            system_prompt=os.environ.get("XLC_SYSTEM_PROMPT", ""),
        )
