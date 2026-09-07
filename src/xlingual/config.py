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
        provider = os.environ.get("XLC_PROVIDER", "groq").strip().lower()
        if provider not in PROVIDERS:
            raise SystemExit(
                f"Unknown XLC_PROVIDER '{provider}'. "
                f"Choose one of: {', '.join(sorted(PROVIDERS))}"
            )
        base_url = os.environ.get("XLC_BASE_URL") or PROVIDERS[provider]
        if not base_url:
            raise SystemExit("XLC_PROVIDER=custom requires XLC_BASE_URL to be set.")
        api_key = os.environ.get("XLC_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "XLC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        model = os.environ.get("XLC_MODEL", "").strip()
        if not model:
            raise SystemExit("XLC_MODEL is not set (for example: llama-3.3-70b-versatile).")
        return cls(
            provider=provider,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model,
            judge_model=os.environ.get("XLC_JUDGE_MODEL", "").strip() or model,
            temperature=float(os.environ.get("XLC_TEMPERATURE", "0")),
            max_tokens=int(os.environ.get("XLC_MAX_TOKENS", "512")),
            requests_per_minute=int(os.environ.get("XLC_RPM", "15")),
            concurrency=max(1, int(os.environ.get("XLC_CONCURRENCY", "6"))),
            max_retries=int(os.environ.get("XLC_MAX_RETRIES", "5")),
            timeout=int(os.environ.get("XLC_TIMEOUT", "90")),
            system_prompt=os.environ.get("XLC_SYSTEM_PROMPT", ""),
        )
