"""Collect model answers and grade them.

Both stages append to JSONL files under results/ and skip work that is already
recorded, so an interrupted run - which on a free tier is the normal case, not
the exception - is resumed by running the same command again.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import LANGUAGES
from .client import ChatClient, ProviderError
from .config import Config, RESULTS_DIR
from .dataset import Item
from .graders import ERROR, NEEDS_JUDGE, grade
from .judge import judge_answer

ANSWERS_FILE = "answers.jsonl"
GRADES_FILE = "grades.jsonl"


@dataclass
class Answer:
    item_id: str
    language: str
    model: str
    prompt: str
    answer: str
    error: str = ""


@dataclass
class Grade:
    item_id: str
    language: str
    category: str
    model: str
    status: str
    detail: str
    judged: bool


def _results_dir(config: Config) -> Path:
    slug = "".join(c if c.isalnum() or c in "-._" else "_" for c in config.model)
    path = RESULTS_DIR / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append(path: Path, record) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def collect_answers(config: Config, items: list[Item], *, languages=LANGUAGES) -> Path:
    """Ask the model every prompt in every language, skipping what we already have."""
    out_dir = _results_dir(config)
    path = out_dir / ANSWERS_FILE
    done = {(row["item_id"], row["language"]) for row in _read_jsonl(path) if not row.get("error")}

    client = ChatClient(config)
    pending = [(item, lang) for item in items for lang in languages if (item.id, lang) not in done]
    if not pending:
        print(f"All {len(items) * len(languages)} answers already collected.")
        return path

    print(f"Collecting {len(pending)} answers with {config.model} ({config.provider})...")
    started = time.monotonic()
    for index, (item, language) in enumerate(pending, 1):
        prompt = item.prompts[language]
        try:
            text = client.complete(prompt)
            record = Answer(item.id, language, config.model, prompt, text)
        except ProviderError as exc:
            record = Answer(item.id, language, config.model, prompt, "", str(exc))
        _append(path, record)
        _progress(index, len(pending), item.id, language, started)
    print()
    return path


def grade_answers(config: Config, items: list[Item], *, use_judge: bool = True) -> Path:
    """Grade collected answers, calling the judge only where code cannot decide."""
    out_dir = _results_dir(config)
    answers = _read_jsonl(out_dir / ANSWERS_FILE)
    if not answers:
        raise SystemExit("No answers found. Run 'xlc run' first.")

    grades_path = out_dir / GRADES_FILE
    done = {(row["item_id"], row["language"]) for row in _read_jsonl(grades_path)}
    by_id = {item.id: item for item in items}

    client = ChatClient(config) if use_judge else None
    pending = [row for row in answers if (row["item_id"], row["language"]) not in done]
    if not pending:
        print(f"All {len(answers)} answers already graded.")
        return grades_path

    print(f"Grading {len(pending)} answers...")
    started = time.monotonic()
    for index, row in enumerate(pending, 1):
        item = by_id.get(row["item_id"])
        if item is None:
            continue
        if row.get("error"):
            verdict = type("V", (), {"status": ERROR, "detail": row["error"]})()
            judged = False
        else:
            verdict = grade(item, row["answer"])
            judged = False
            if verdict.status == NEEDS_JUDGE:
                if client is None:
                    continue
                verdict = judge_answer(client, item, row["language"], row["answer"])
                judged = True
        _append(
            grades_path,
            Grade(
                item_id=item.id,
                language=row["language"],
                category=item.category,
                model=row["model"],
                status=verdict.status,
                detail=verdict.detail,
                judged=judged,
            ),
        )
        _progress(index, len(pending), item.id, row["language"], started)
    print()
    return grades_path


def _progress(index: int, total: int, item_id: str, language: str, started: float) -> None:
    elapsed = time.monotonic() - started
    rate = index / elapsed if elapsed else 0
    remaining = (total - index) / rate if rate else 0
    sys.stdout.write(
        f"\r  [{index}/{total}] {item_id:<12} {language}  "
        f"~{remaining / 60:.1f} min left    "
    )
    sys.stdout.flush()


def load_grades(config: Config) -> list[dict]:
    return _read_jsonl(_results_dir(config) / GRADES_FILE)


def load_answers(config: Config) -> list[dict]:
    return _read_jsonl(_results_dir(config) / ANSWERS_FILE)


def results_dir(config: Config) -> Path:
    return _results_dir(config)
