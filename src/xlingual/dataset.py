"""Loading and validating the evaluation set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import LANGUAGES
from .config import DATA_FILE

CHECK_TYPES = {
    "contains_any",
    "exact",
    "starts_with",
    "number",
    "digits_only",
    "word_count",
    "line_count",
    "sentence_count",
    "json_keys",
    "json_array_len",
    "refusal",
    "judge",
}

JUDGED_CHECKS = {"refusal", "judge"}


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    check: dict
    prompts: dict[str, str]

    @property
    def needs_judge(self) -> bool:
        return self.check.get("type") in JUDGED_CHECKS


class DatasetError(ValueError):
    pass


def load_items(path: Path | None = None) -> list[Item]:
    source = path or DATA_FILE
    if not source.exists():
        raise DatasetError(f"dataset not found: {source}")

    items: list[Item] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{source.name}:{line_no} is not valid JSON: {exc}") from exc
        item = _build(record, source.name, line_no)
        if item.id in seen:
            raise DatasetError(f"{source.name}:{line_no} duplicate id '{item.id}'")
        seen.add(item.id)
        items.append(item)

    if not items:
        raise DatasetError(f"{source} contains no items")
    return items


def _build(record: dict, filename: str, line_no: int) -> Item:
    where = f"{filename}:{line_no}"
    for field in ("id", "category", "check", "prompts"):
        if field not in record:
            raise DatasetError(f"{where} is missing '{field}'")

    check = record["check"]
    if not isinstance(check, dict) or "type" not in check:
        raise DatasetError(f"{where} has a malformed 'check'")
    if check["type"] not in CHECK_TYPES:
        raise DatasetError(
            f"{where} uses unknown check type '{check['type']}'. "
            f"Known types: {', '.join(sorted(CHECK_TYPES))}"
        )

    prompts = record["prompts"]
    missing = [lang for lang in LANGUAGES if not prompts.get(lang, "").strip()]
    if missing:
        raise DatasetError(f"{where} ('{record['id']}') is missing prompts for: {', '.join(missing)}")

    return Item(
        id=record["id"],
        category=record["category"],
        check=check,
        prompts={lang: prompts[lang] for lang in LANGUAGES},
    )


def categories(items: list[Item]) -> list[str]:
    return sorted({item.category for item in items})
